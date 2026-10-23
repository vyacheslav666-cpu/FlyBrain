"""«Живой» мозг (C9, C10): один повтор модели в отдельном потоке, кадры спайков ~25 раз в секунду (D-015, D-033, D-034).

Поток считает движок блоками по 18 шагов (D-008) и примерно каждые 40 мс настоящего времени собирает кадр:
какие нейроны выстрелили с прошлого кадра и сколько раз. Кадры и сообщения о состоянии уходят подписчикам
(сервер передаёт их страницам по WebSocket). Команды (пауза, продолжить, сброс, стимулы, замедление) приходят
из других потоков и выполняются между блоками: движок трогает только этот поток.

Замедление показа (C10) только делает паузы между блоками: каждый блок и каждый шаг по 0,1 мс считается,
ничего не пропускается.

Панель «Психоделик (эксперимент)» (P2, D-064…D-066; модуль `flybrain.psylive`): доза, режим и шум мембраны меняются
на ходу теми же средствами движка, что в опытах P1/P1b; индикатор «сеть зажглась» считается по каждому кадру.
"""

import json
import math
import queue
import struct
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import torch

from flybrain.engine import Engine
from flybrain.psylive import DOSE_MAX, MODES, SIGMA_MAX, IgnitionMeter, PsyLive
from flybrain.stimuli import DEFAULT_HZ, DEFAULT_SIDE, MAX_HZ

FRAME_FORMAT = 1
# Заголовок кадра: версия, номер «эпохи» (растёт при сбросе), начало и конец кадра в мс модели,
# скорость (во сколько раз медленнее реального времени), число нейронов в кадре. Дальше uint32 номера
# нейронов и uint16 число их спайков за кадр. Всё little-endian; должно совпадать с web/live.js.
FRAME_HEADER = struct.Struct("<IIddfI")
FRAME_SECONDS = 0.04  # ~25 кадров в секунду
MAX_SPEED = 1.0  # быстрее реального времени не считаем: живой режим показывает время мухи как есть
MAX_SLEEP_SECONDS = 0.05  # дольше без проверки команд страницы не спим
SPEED_WINDOW_SECONDS = 2.0  # скорость — среднее за последние 2 с счёта
# Ползунок замедления: цель — от SLOW_MIN до 1× реального времени; «без замедления» — цель не задана.
SLOW_MIN = 0.02
SLOW_SLIDER_MAX = 0.4  # последнее деление ползунка перед «максимум» (на этой машине модель и так идёт ~0,48×)
PSY_READING_FRAMES = 10  # показания индикатора «сеть зажглась» — примерно раз в 0,4 с (и сразу, если он переключился)


def encode_frame(epoch: int, start_ms: float, end_ms: float, speed: float, neurons: np.ndarray,
                 spikes: np.ndarray) -> bytes:
    neurons = np.asarray(neurons, dtype="<u4")
    spikes = np.minimum(np.asarray(spikes), 65535).astype("<u2")
    return FRAME_HEADER.pack(FRAME_FORMAT, epoch, start_ms, end_ms, speed, len(neurons)) + neurons.tobytes() \
        + spikes.tobytes()


def decode_frame(blob: bytes) -> dict:
    version, epoch, start_ms, end_ms, speed, count = FRAME_HEADER.unpack_from(blob)
    offset = FRAME_HEADER.size
    neurons = np.frombuffer(blob, dtype="<u4", count=count, offset=offset)
    spikes = np.frombuffer(blob, dtype="<u2", count=count, offset=offset + 4 * count)
    return {"version": version, "epoch": epoch, "start_ms": start_ms, "end_ms": end_ms, "speed": speed,
            "neurons": neurons, "spikes": spikes}


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"видеокарта {torch.cuda.get_device_name(device)}"
    return "процессор (видеокарта не найдена; это в десятки раз медленнее)"


class LiveSim:
    """Один повтор модели Shiu, который считается, пока открыта хотя бы одна страница.

    `make_engine` строит движок (это долго, поэтому — в потоке и только когда пришла первая страница).
    `stimuli` — кнопки-стимулы: ключ → сторона мухи → номера нейронов (`flybrain.stimuli.build_catalog`).
    Подписчик — функция, которой поток отдаёт кадры (`bytes`) и состояние (`str` с JSON).
    """

    def __init__(self, make_engine: Callable[[], Engine], stimuli: dict[str, dict[str, np.ndarray]],
                 frame_seconds: float = FRAME_SECONDS, max_speed: float | None = MAX_SPEED,
                 psy: PsyLive | None = None):
        self._make_engine = make_engine
        self._stimuli = {key: {side: np.asarray(ids, dtype=np.int64) for side, ids in sides.items()}
                         for key, sides in stimuli.items()}
        # Что сейчас включено: ключ → {on, hz, side}. Меняет только поток расчёта.
        self._stimulus_state = {key: {"on": False, "hz": DEFAULT_HZ, "side": DEFAULT_SIDE} for key in self._stimuli}
        self._target_speed: float | None = None  # None — без замедления
        self._anchor: tuple[float, float] | None = None  # (настоящее время, время модели), от которых считаем темп
        self._block_ms: float | None = None  # сколько мс модели в одном блоке (узнаём по первому блоку)
        self._frame_seconds = frame_seconds
        self._max_speed = max_speed
        self._commands: queue.Queue = queue.Queue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._subscribers: set[Callable] = set()
        self._thread: threading.Thread | None = None
        self.engine: Engine | None = None
        self._prev_counts: torch.Tensor | None = None
        self._speed_samples: deque = deque()
        self._state = "waiting"  # waiting → starting → running / paused; error — движок не построился
        self._paused = False
        self._epoch = 0
        self._speed = 0.0
        self._device = ""
        self._message = ""
        # Панель «Психоделик» (P2): доза, режим, шум; индикатор считается, только если данные панели есть.
        self._psy = psy if psy is not None and psy.available else None
        self._psy_state = {"dose": 0.0, "mode": "p1", "sigma": 0.0}
        self._meter = IgnitionMeter(len(self._psy.alln_exc)) if self._psy else None
        self._ignited = False
        self._frames_since_reading = 0
        self._stim_index: torch.Tensor | None = None  # стимулируемые сейчас нейроны (их спайки индикатор не считает)
        self._alln_index: torch.Tensor | None = None

    # ---------- из других потоков ----------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="flybrain-live", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def subscribe(self, send: Callable) -> None:
        with self._lock:
            self._subscribers.add(send)
        send(self.status_json())
        self._wake.set()

    def unsubscribe(self, send: Callable) -> None:
        with self._lock:
            self._subscribers.discard(send)

    def command(self, message: dict) -> None:
        """Команда страницы (кривые команды молча отбрасываются):

        - {"cmd": "pause" | "resume" | "reset"};
        - {"cmd": "stimulus", "key": "sugar", "on": true, "hz": 100, "side": "left"}: любое из on, hz, side можно не указывать;
        - {"cmd": "speed", "value": 0.05}: показ в 0,05× реального времени; "value": null — без замедления;
        - {"cmd": "psy", "dose": 2.5, "mode": "p1" | "kc_inhibit", "sigma": 1.0}: любое можно не указывать (P2).
        """
        cmd = message.get("cmd")
        if cmd in ("pause", "resume", "reset"):
            clean = {"cmd": cmd}
        elif cmd == "stimulus":
            clean = self._clean_stimulus(message)
        elif cmd == "speed":
            clean = self._clean_speed(message)
        elif cmd == "psy" and self._psy is not None:
            clean = self._clean_psy(message)
        else:
            clean = None
        if clean is not None:
            self._commands.put(clean)
            self._wake.set()

    def _clean_stimulus(self, message: dict) -> dict | None:
        key = message.get("key")
        sides = self._stimuli.get(key) if isinstance(key, str) else None
        if sides is None:
            return None
        clean: dict = {"cmd": "stimulus", "key": key}
        if "on" in message:
            if not isinstance(message["on"], bool):
                return None
            clean["on"] = message["on"]
        if "hz" in message:
            hz = message["hz"]
            if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not math.isfinite(hz):
                return None
            clean["hz"] = min(max(float(hz), 0.0), MAX_HZ)
        if "side" in message:
            if message["side"] not in sides:  # стороны без списка из статьи у стимула нет, выбрать её нельзя (D-034)
                return None
            clean["side"] = message["side"]
        return clean

    @staticmethod
    def _clean_speed(message: dict) -> dict | None:
        value = message.get("value")
        if value is None:
            return {"cmd": "speed", "value": None}
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return {"cmd": "speed", "value": min(max(float(value), SLOW_MIN), 1.0)}

    @staticmethod
    def _clean_psy(message: dict) -> dict | None:
        clean: dict = {"cmd": "psy"}
        for field, top in (("dose", DOSE_MAX), ("sigma", SIGMA_MAX)):
            if field in message:
                value = message[field]
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    return None
                clean[field] = min(max(float(value), 0.0), top)
        if "mode" in message:
            if message["mode"] not in MODES:
                return None
            clean["mode"] = message["mode"]
        return clean

    def status(self) -> dict:
        return {"type": "status", "state": self._state, "paused": self._paused,
                "stimuli": {key: dict(state) for key, state in self._stimulus_state.items()},
                "target_speed": self._target_speed, "slow_min": SLOW_MIN, "slow_slider_max": SLOW_SLIDER_MAX,
                "epoch": self._epoch, "time_ms": self.engine.time_ms if self.engine else 0.0,
                "speed": self._speed, "max_speed": self._max_speed, "device": self._device,
                "message": self._message,
                "psy": {"available": self._psy is not None, **self._psy_state, "ignited": self._ignited}}

    def status_json(self) -> str:
        return json.dumps(self.status(), ensure_ascii=False)

    # ---------- поток расчёта ----------

    def _broadcast(self, item) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for send in subscribers:
            send(item)

    def _has_subscribers(self) -> bool:
        with self._lock:
            return bool(self._subscribers)

    def _set_state(self, state: str, message: str = "") -> None:
        self._state, self._message = state, message
        self._broadcast(self.status_json())

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._has_subscribers():
                # Никто не смотрит — видеокарту не греем. Состояние модели сохраняется до следующей страницы.
                self._speed_samples.clear()
                self._anchor = None
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            if self.engine is None and not self._build_engine():
                return
            changed = self._apply_commands()
            if changed:
                self._broadcast(self.status_json())
            if self._paused:
                self._speed_samples.clear()
                self._anchor = None
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            self._run_frame()

    def _build_engine(self) -> bool:
        self._set_state("starting", "готовлю движок…")
        try:
            engine = self._make_engine()
            engine.prepare()
        except Exception as exc:  # движок не построился — сообщаем странице, сервер работает дальше
            self._set_state("error", f"движок не запустился: {str(exc).splitlines()[0]}")
            return False
        self.engine = engine
        self._prev_counts = engine.counts[:, 0].clone()
        if self._psy is not None:
            self._alln_index = torch.as_tensor(self._psy.alln_exc, device=engine.device)
            self._stim_index = torch.zeros(0, dtype=torch.int64, device=engine.device)
        self._device = device_label(engine.device)
        self._set_state("paused" if self._paused else "running")
        return True

    def _apply_commands(self) -> bool:
        changed = False
        while True:
            try:
                message = self._commands.get_nowait()
            except queue.Empty:
                return changed
            cmd = message["cmd"]
            if cmd == "pause":
                self._paused = True
            elif cmd == "resume":
                self._paused = False
                self._anchor = None
            elif cmd == "reset":
                # Все нейроны в покой, время 0. Включённые стимулы остаются включёнными (как в движке).
                self.engine.reset()
                self._prev_counts.zero_()
                self._epoch += 1
                self._speed_samples.clear()
                self._anchor = None
                if self._meter is not None:  # сброс гасит «зажжённую» сеть: индикатор — заново
                    self._meter.reset()
                    self._ignited = False
            elif cmd == "stimulus":
                self._apply_stimulus(message)
            elif cmd == "psy":
                self._apply_psy(message)
            elif cmd == "speed":
                self._target_speed = message["value"]
                self._speed_samples.clear()  # счётчик скорости должен показывать новый темп, а не смесь со старым
                self._anchor = None
            self._state = "paused" if self._paused else "running"
            changed = True

    def _apply_stimulus(self, message: dict) -> None:
        key = message["key"]
        state = self._stimulus_state[key]
        for field in ("on", "hz", "side"):
            if field in message:
                state[field] = message[field]
        if state["on"]:
            self.engine.set_stimulus(key, self._stimuli[key][state["side"]], state["hz"])
        else:
            self.engine.remove_stimulus(key)
        if self._psy is not None:
            on = [self._stimuli[k][s["side"]] for k, s in self._stimulus_state.items() if s["on"]]
            index = np.unique(np.concatenate(on)) if on else np.zeros(0, dtype=np.int64)
            self._stim_index = torch.as_tensor(index, device=self.engine.device)

    def _apply_psy(self, message: dict) -> None:
        """Доза, режим, шум — на ходу, теми же средствами движка, что в P1/P1b; при дозе 0 и σ = 0 — прежний расчёт."""
        state = self._psy_state
        for field in ("dose", "mode", "sigma"):
            if field in message:
                state[field] = message[field]
        self.engine.set_threshold_shift(self._psy.shift(state["dose"], state["mode"]))
        self.engine.set_noise(state["sigma"])

    def _speed_limit(self) -> float | None:
        """Быстрее какого темпа идти нельзя: цель ползунка и «не быстрее реального времени» (D-017, D-033)."""
        limits = [x for x in (self._target_speed, self._max_speed) if x]
        return min(limits) if limits else None

    def _seconds_until_next_block(self, limit: float, now: float) -> float:
        """Сколько ждать до следующего блока, чтобы время модели не обгоняло настоящее, умноженное на `limit`."""
        engine = self.engine
        if self._anchor is None:
            self._anchor = (now, engine.time_ms)
        wall0, model0 = self._anchor
        behind_ms = model0 + (now - wall0) * 1000.0 * limit - engine.time_ms
        if behind_ms > (self._block_ms or 0.0):
            # Видеокарта не успевала (или была пауза): долг не копим, иначе потом был бы рывок быстрее лимита.
            self._anchor = (now, engine.time_ms)
            return 0.0
        return max(0.0, -behind_ms / (1000.0 * limit))

    def _run_frame(self) -> None:
        engine = self.engine
        started = time.perf_counter()
        start_ms = engine.time_ms
        limit = self._speed_limit()
        while True:
            now = time.perf_counter()
            wait = self._seconds_until_next_block(limit, now) if limit else 0.0
            if wait > 0:
                # Впереди графика. Если кадр уже что-то посчитал и кончится раньше, чем понадобится следующий блок,
                # ждём до конца кадра и отправляем его; иначе ждём блок.
                remaining = self._frame_seconds - (now - started)
                last = engine.time_ms > start_ms and remaining <= wait
                # sleep, а не Event.wait: на Windows тот ждёт кратно ~15 мс, а блоки идут чаще. Спим кусками,
                # чтобы команда страницы (пауза, сброс, ползунок) не ждала дольше MAX_SLEEP_SECONDS.
                time.sleep(max(0.0, min(remaining if last else wait, MAX_SLEEP_SECONDS)))
                if last or not self._commands.empty():
                    break
                continue
            before = engine.time_ms
            engine.run_steps(engine.block_steps)
            engine.synchronize()  # иначе процессор убежит вперёд и кадр не будет соответствовать времени
            self._block_ms = engine.time_ms - before
            if time.perf_counter() - started >= self._frame_seconds:
                break
        if engine.time_ms == start_ms:
            return  # ничего не посчитано (пришла команда): пустой кадр не отправляем
        counts = engine.counts[:, 0]
        delta = counts - self._prev_counts
        fired = torch.nonzero(delta).flatten()
        spikes = delta[fired]
        self._prev_counts.copy_(counts)
        self._update_speed(time.perf_counter(), engine.time_ms)
        self._broadcast(encode_frame(self._epoch, start_ms, engine.time_ms, self._speed,
                                     fired.cpu().numpy(), spikes.cpu().numpy()))
        if self._meter is not None:
            self._measure(start_ms, engine.time_ms, delta)

    def _measure(self, start_ms: float, end_ms: float, delta: torch.Tensor) -> None:
        """Индикатор «сеть зажглась» (D-064): спайки не стимулируемых нейронов и петля усиковой доли за кадр."""
        total = int(delta.sum())
        if len(self._stim_index):
            total -= int(delta[self._stim_index].sum())
        self._meter.add(start_ms, end_ms, total, (delta[self._alln_index] > 0).cpu().numpy())
        ignited = self._meter.ignited
        self._frames_since_reading += 1
        switched = ignited != self._ignited
        if switched:
            self._ignited = ignited
            self._broadcast(self.status_json())
        if switched or self._frames_since_reading >= PSY_READING_FRAMES:
            self._frames_since_reading = 0
            self._broadcast(json.dumps(self._meter.reading(), ensure_ascii=False))

    def _update_speed(self, now: float, time_ms: float) -> None:
        """Скорость (D-017): мс модели за последние ~2 с, делённые на эти 2 с настоящего времени."""
        samples = self._speed_samples
        samples.append((now, time_ms))
        while len(samples) > 2 and now - samples[1][0] >= SPEED_WINDOW_SECONDS:
            samples.popleft()
        if len(samples) >= 2:
            (wall0, sim0), (wall1, sim1) = samples[0], samples[-1]
            if wall1 > wall0:
                self._speed = (sim1 - sim0) / 1000.0 / (wall1 - wall0)
