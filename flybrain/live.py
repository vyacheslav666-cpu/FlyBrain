"""«Живой» мозг (C9): один повтор модели в отдельном потоке, кадры спайков ~25 раз в секунду (D-015, D-033).

Поток считает движок блоками по 18 шагов (D-008) и примерно каждые 40 мс настоящего времени собирает кадр:
какие нейроны выстрелили с прошлого кадра и сколько раз. Кадры и сообщения о состоянии уходят подписчикам
(сервер передаёт их страницам по WebSocket). Команды (пауза, продолжить, сброс, тест) приходят из других
потоков и выполняются между блоками: движок трогает только этот поток.
"""

import json
import queue
import struct
import threading
import time
from collections import deque
from collections.abc import Callable

import numpy as np
import torch

from flybrain.engine import Engine

FRAME_FORMAT = 1
# Заголовок кадра: версия, номер «эпохи» (растёт при сбросе), начало и конец кадра в мс модели,
# скорость (во сколько раз медленнее реального времени), число нейронов в кадре. Дальше uint32 номера
# нейронов и uint16 число их спайков за кадр. Всё little-endian; должно совпадать с web/live.js.
FRAME_HEADER = struct.Struct("<IIddfI")
FRAME_SECONDS = 0.04  # ~25 кадров в секунду
MAX_SPEED = 1.0  # быстрее реального времени не считаем: живой режим показывает время мухи как есть
SPEED_WINDOW_SECONDS = 2.0  # скорость — среднее за последние 2 с счёта
TEST_STIMULUS = "test_sugar"
TEST_HZ = 100.0


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
    `test_neurons` — нейроны временной кнопки «тест» (сахар, C9).
    Подписчик — функция, которой поток отдаёт кадры (`bytes`) и состояние (`str` с JSON).
    """

    def __init__(self, make_engine: Callable[[], Engine], test_neurons, test_label: str = "сахар",
                 test_about: str = "", frame_seconds: float = FRAME_SECONDS, max_speed: float | None = MAX_SPEED):
        self._make_engine = make_engine
        self._test_neurons = np.asarray(test_neurons, dtype=np.int64)
        self._test_label = test_label
        self._test_about = test_about
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
        self._test_on = False
        self._epoch = 0
        self._speed = 0.0
        self._device = ""
        self._message = ""

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
        """Команда страницы: {"cmd": "pause" | "resume" | "reset"} или {"cmd": "test", "on": true/false}."""
        if message.get("cmd") in ("pause", "resume", "reset", "test"):
            self._commands.put(message)
            self._wake.set()

    def status(self) -> dict:
        return {"type": "status", "state": self._state, "paused": self._paused, "test": self._test_on,
                "test_label": self._test_label, "test_about": self._test_about,
                "test_count": len(self._test_neurons), "test_hz": TEST_HZ,
                "epoch": self._epoch, "time_ms": self.engine.time_ms if self.engine else 0.0,
                "speed": self._speed, "max_speed": self._max_speed, "device": self._device,
                "message": self._message}

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
            elif cmd == "reset":
                # Все нейроны в покой, время 0. Включённый тест остаётся включённым (как стимулы в движке).
                self.engine.reset()
                self._prev_counts.zero_()
                self._epoch += 1
                self._speed_samples.clear()
            elif cmd == "test":
                self._test_on = bool(message.get("on"))
                if self._test_on:
                    self.engine.set_stimulus(TEST_STIMULUS, self._test_neurons, TEST_HZ)
                else:
                    self.engine.remove_stimulus(TEST_STIMULUS)
            self._state = "paused" if self._paused else "running"
            changed = True

    def _run_frame(self) -> None:
        engine = self.engine
        started = time.perf_counter()
        start_ms = engine.time_ms
        frame_ms = self._frame_seconds * 1000.0
        limit_ms = frame_ms * self._max_speed if self._max_speed else None
        while True:
            engine.run_steps(engine.block_steps)
            engine.synchronize()  # иначе процессор убежит вперёд и кадр не будет соответствовать времени
            if time.perf_counter() - started >= self._frame_seconds:
                break
            if limit_ms is not None and engine.time_ms - start_ms >= limit_ms:
                # Быстрее реального времени не идём: ждём до конца кадра.
                time.sleep(max(0.0, self._frame_seconds - (time.perf_counter() - started)))
                break
        counts = engine.counts[:, 0]
        delta = counts - self._prev_counts
        fired = torch.nonzero(delta).flatten()
        spikes = delta[fired]
        self._prev_counts.copy_(counts)
        self._update_speed(time.perf_counter(), engine.time_ms)
        self._broadcast(encode_frame(self._epoch, start_ms, engine.time_ms, self._speed,
                                     fired.cpu().numpy(), spikes.cpu().numpy()))

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
