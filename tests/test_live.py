"""Тесты «живого» мозга (C9, C10): поток кадров спайков, пауза, сброс, стимулы, замедление, WebSocket сервера."""

import json
import threading
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from flybrain import live, processed
from flybrain.engine import Engine
from flybrain.paths import PROCESSED_DIR
from flybrain.server import create_app


def _tiny_engine() -> Engine:
    # Нейрон 0 — «вкусовой» (его включают стимулы), нейрон 1 получает от него сильную возбуждающую связь.
    return Engine(np.array([0, 0, 1]), np.array([0]), np.array([1000.0]), n_trials=1, device="cpu", seed=7)


# Кнопки в тестах: «сахар» — на левой стороне нейрон 0, на правой нейрон 1; «горечь» — только левая сторона (нейрон 0).
STIMULI = {"sugar": {"left": [0], "right": [1]}, "bitter": {"left": [0]}}


class Collector:
    """Подписчик для тестов: складывает кадры и состояния."""

    def __init__(self):
        self.frames: list[dict] = []
        self.statuses: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, item) -> None:
        with self.lock:
            if isinstance(item, bytes):
                self.frames.append(live.decode_frame(item))
            else:
                self.statuses.append(json.loads(item))

    def wait(self, condition, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if condition(self):
                    return
            time.sleep(0.01)
        raise AssertionError("не дождались")


@pytest.fixture
def sim():
    simulation = live.LiveSim(_tiny_engine, STIMULI, frame_seconds=0.02, max_speed=None)
    simulation.start()
    yield simulation
    simulation.stop()


def test_frame_round_trip():
    blob = live.encode_frame(3, 1.8, 21.6, 0.48, np.array([5, 138_000]), np.array([1, 70_000]))
    frame = live.decode_frame(blob)
    assert (frame["version"], frame["epoch"]) == (live.FRAME_FORMAT, 3)
    assert (frame["start_ms"], frame["end_ms"]) == (1.8, 21.6)
    assert frame["speed"] == pytest.approx(0.48)
    assert frame["neurons"].tolist() == [5, 138_000]
    assert frame["spikes"].tolist() == [1, 65535]  # больше 65535 спайков за кадр не бывает; на всякий случай обрезаем
    assert live.FRAME_HEADER.size == 32  # web/live.js читает заголовок как 32 байта
    assert len(blob) == 32 + 2 * 4 + 2 * 2


def test_engine_is_built_only_for_a_page():
    simulation = live.LiveSim(_tiny_engine, STIMULI, frame_seconds=0.02)
    simulation.start()
    try:
        time.sleep(0.3)
        assert simulation.engine is None  # никто не смотрит — видеокарту не занимаем
        collector = Collector()
        simulation.subscribe(collector)
        collector.wait(lambda c: c.frames)
        assert collector.statuses[0]["state"] == "waiting"
        assert [s["state"] for s in collector.statuses][-1] == "running"
    finally:
        simulation.stop()


def test_frames_carry_every_spike_and_follow_each_other(sim):
    collector = Collector()
    sim.subscribe(collector)
    collector.wait(lambda c: len(c.frames) >= 3)
    with collector.lock:
        assert all(len(f["neurons"]) == 0 for f in collector.frames)  # в покое модель молчит (D-019)
    sim.command({"cmd": "stimulus", "key": "sugar", "on": True})
    collector.wait(lambda c: sum(int(f["spikes"].sum()) for f in c.frames) > 200)
    sim.command({"cmd": "pause"})
    collector.wait(lambda c: c.statuses[-1]["paused"])
    time.sleep(0.1)  # кадр, который считался в момент команды, успевает прийти
    with collector.lock:
        frames = list(collector.frames)
        test_on = [s for s in collector.statuses if s["stimuli"]["sugar"]["on"]]
    assert test_on
    # Кадры идут встык: конец одного — начало следующего, ничего не пропало.
    for before, after in zip(frames, frames[1:]):
        assert after["start_ms"] == pytest.approx(before["end_ms"])
    totals = np.zeros(2, dtype=np.int64)
    for frame in frames:
        np.add.at(totals, frame["neurons"].astype(np.int64), frame["spikes"])
    assert totals.tolist() == sim.engine.counts[:, 0].tolist()
    assert totals[0] > 0 and totals[1] > 0  # стимул включил нейрон 0, а он — соседа


def test_pause_stops_frames_and_resume_continues(sim):
    collector = Collector()
    sim.subscribe(collector)
    collector.wait(lambda c: len(c.frames) >= 2)
    sim.command({"cmd": "pause"})
    collector.wait(lambda c: c.statuses[-1]["state"] == "paused")
    time.sleep(0.1)
    with collector.lock:
        count = len(collector.frames)
    time_ms = sim.engine.time_ms
    time.sleep(0.3)
    assert len(collector.frames) == count and sim.engine.time_ms == time_ms
    sim.command({"cmd": "resume"})
    collector.wait(lambda c: len(c.frames) > count)
    assert collector.statuses[-1]["state"] == "running"


def test_reset_starts_model_time_over_with_new_epoch(sim):
    collector = Collector()
    sim.subscribe(collector)
    sim.command({"cmd": "stimulus", "key": "sugar", "on": True})
    collector.wait(lambda c: c.frames and c.frames[-1]["end_ms"] > 50)
    sim.command({"cmd": "reset"})
    collector.wait(lambda c: c.frames and c.frames[-1]["epoch"] == 1)
    with collector.lock:
        first_new = next(f for f in collector.frames if f["epoch"] == 1)
        status = next(s for s in collector.statuses if s["epoch"] == 1)
    assert first_new["start_ms"] == 0.0
    assert status["stimuli"]["sugar"]["on"] is True  # сброс не выключает стимулы


class InstantEngine:
    """Движок, который считает мгновенно: проверяем, что живой режим всё равно не обгоняет реальное время."""

    block_steps = 18
    device = torch.device("cpu")

    def __init__(self):
        self.counts = torch.zeros((3, 1), dtype=torch.int32)
        self.time_ms = 0.0

    def prepare(self):
        pass

    def synchronize(self):
        pass

    def run_steps(self, steps: int):
        self.time_ms += steps * 0.1

    def reset(self):
        self.time_ms = 0.0

    def set_stimulus(self, name, indices, hz):
        pass

    def remove_stimulus(self, name):
        pass


def test_speed_is_not_faster_than_real_time():
    simulation = live.LiveSim(InstantEngine, STIMULI, frame_seconds=0.02)  # MAX_SPEED = 1
    simulation.start()
    try:
        collector = Collector()
        simulation.subscribe(collector)
        collector.wait(lambda c: len(c.frames) >= 60)
        with collector.lock:
            speed = collector.frames[-1]["speed"]
            frame_ms = [f["end_ms"] - f["start_ms"] for f in collector.frames]
        assert 0.7 < speed <= 1.1  # без ограничения было бы в тысячи раз быстрее
        assert max(frame_ms) <= 20 + 1.8 + 1e-9  # кадр — не больше 20 мс модели (с точностью до блока 1,8 мс)
    finally:
        simulation.stop()


def _queued(simulation: live.LiveSim) -> list[dict]:
    items = []
    while not simulation._commands.empty():
        items.append(simulation._commands.get_nowait())
    return items


def test_bad_commands_are_dropped_and_good_ones_are_cleaned():
    simulation = live.LiveSim(_tiny_engine, STIMULI)  # поток не запускаем: смотрим только, что попало в очередь
    bad = [
        {"cmd": "test", "on": True},  # старая команда C9 больше не существует
        {"cmd": "stimulus", "key": "nope", "on": True},
        {"cmd": "stimulus", "key": 5},
        {"cmd": "stimulus", "key": "bitter", "side": "right"},  # у горечи нет списка для правой стороны (D-034)
        {"cmd": "stimulus", "key": "sugar", "side": "up"},
        {"cmd": "stimulus", "key": "sugar", "on": "yes"},
        {"cmd": "stimulus", "key": "sugar", "hz": "100"},
        {"cmd": "stimulus", "key": "sugar", "hz": True},
        {"cmd": "stimulus", "key": "sugar", "hz": float("nan")},
        {"cmd": "speed", "value": "fast"},
        {"cmd": "speed", "value": float("inf")},
        {"cmd": "reboot"},
    ]
    for message in bad:
        simulation.command(message)
    assert _queued(simulation) == []
    simulation.command({"cmd": "stimulus", "key": "sugar", "on": True, "hz": 999, "side": "right", "extra": 1})
    simulation.command({"cmd": "stimulus", "key": "sugar", "hz": -5})
    simulation.command({"cmd": "speed", "value": 0.0001})
    simulation.command({"cmd": "speed", "value": None})
    assert _queued(simulation) == [
        {"cmd": "stimulus", "key": "sugar", "on": True, "hz": 200.0, "side": "right"},  # частота зажата в 0…200
        {"cmd": "stimulus", "key": "sugar", "hz": 0.0},
        {"cmd": "speed", "value": live.SLOW_MIN},
        {"cmd": "speed", "value": None},
    ]


def test_stimuli_switch_on_off_change_hz_and_side_and_combine(sim):
    collector = Collector()
    sim.subscribe(collector)
    collector.wait(lambda c: c.frames)
    assert collector.statuses[-1]["stimuli"]["sugar"] == {"on": False, "hz": 100.0, "side": "left"}  # как в плане C10

    def last_stimuli():
        return collector.statuses[-1]["stimuli"]

    sim.command({"cmd": "stimulus", "key": "sugar", "on": True})
    collector.wait(lambda c: last_stimuli()["sugar"]["on"])
    assert sim.engine._stimuli["sugar"][1] == 100.0 and sim.engine._stimuli["sugar"][0].tolist() == [0]

    sim.command({"cmd": "stimulus", "key": "sugar", "hz": 40})
    sim.command({"cmd": "stimulus", "key": "sugar", "side": "right"})
    collector.wait(lambda c: last_stimuli()["sugar"]["side"] == "right")
    indices, hz = sim.engine._stimuli["sugar"]
    assert indices.tolist() == [1] and hz == 40.0  # сторона и частота сменились на ходу, стимул остался включённым

    sim.command({"cmd": "stimulus", "key": "bitter", "on": True, "hz": 200})  # сахар + горечь вместе (опыт Б3)
    collector.wait(lambda c: last_stimuli()["bitter"]["on"])
    assert set(sim.engine._stimuli) == {"sugar", "bitter"}

    sim.command({"cmd": "stimulus", "key": "sugar", "on": False})
    collector.wait(lambda c: not last_stimuli()["sugar"]["on"])
    assert set(sim.engine._stimuli) == {"bitter"}
    assert last_stimuli()["sugar"]["hz"] == 40.0 and last_stimuli()["sugar"]["side"] == "right"  # настройки помнятся


def _frames_after_speed(simulation, value, count):
    """Ставит цель замедления и возвращает `count` кадров уже при новой цели и последнее состояние."""
    collector = Collector()
    simulation.subscribe(collector)
    simulation.command({"cmd": "speed", "value": value})
    collector.wait(lambda c: c.statuses[-1]["target_speed"] == value)
    time.sleep(0.3)  # кадр, считавшийся в момент команды, успевает прийти
    with collector.lock:
        skip = len(collector.frames)
    collector.wait(lambda c: len(c.frames) >= skip + count, timeout=30)
    with collector.lock:
        return collector.frames[skip:skip + count], collector.statuses[-1]


def test_slowdown_sets_the_speed_and_skips_no_steps():
    simulation = live.LiveSim(InstantEngine, STIMULI, frame_seconds=0.02, max_speed=None)
    simulation.start()
    try:
        fast, _ = _frames_after_speed(simulation, None, 20)
        assert fast[-1]["speed"] > 5  # мгновенный движок без замедления считает намного быстрее реального времени
        started = time.monotonic()
        frames, status = _frames_after_speed(simulation, 0.1, 40)
        assert status["target_speed"] == 0.1
        assert frames[-1]["speed"] == pytest.approx(0.1, rel=0.2)  # счётчик показывает настоящий темп
        # Кадры идут встык, и каждый — целое число блоков по 1,8 мс: паузы между блоками, но ни одного шага не потеряно.
        for before, after in zip(frames, frames[1:]):
            assert after["start_ms"] == pytest.approx(before["end_ms"])
        for frame in frames:
            blocks = (frame["end_ms"] - frame["start_ms"]) / 1.8
            assert blocks == pytest.approx(round(blocks)) and round(blocks) >= 1
        # Модельное время за кадры соответствует настоящему, умноженному на цель.
        modelled = frames[-1]["end_ms"] - frames[0]["start_ms"]
        assert modelled / 1000 / (time.monotonic() - started) < 0.2
    finally:
        simulation.stop()


def test_slow_motion_gives_exactly_the_same_model_as_full_speed():
    """Замедление — только паузы между блоками: при том же зерне и том же времени модели спайки те же."""

    def stimulated_engine() -> Engine:
        engine = _tiny_engine()
        engine.set_stimulus("always", [0], 150.0)
        return engine

    simulation = live.LiveSim(stimulated_engine, STIMULI, frame_seconds=0.02, max_speed=None)
    simulation.start()
    try:
        collector = Collector()
        simulation.subscribe(collector)
        simulation.command({"cmd": "speed", "value": 0.2})
        collector.wait(lambda c: c.frames and c.frames[-1]["end_ms"] > 40, timeout=30)
        simulation.command({"cmd": "pause"})
        collector.wait(lambda c: c.statuses[-1]["paused"])
        time.sleep(0.2)
        time_ms = simulation.engine.time_ms
        counts = simulation.engine.counts[:, 0].tolist()
        with collector.lock:
            streamed = np.zeros(2, dtype=np.int64)
            for frame in collector.frames:
                np.add.at(streamed, frame["neurons"].astype(np.int64), frame["spikes"])
        assert streamed.tolist() == counts  # в кадры попало всё
    finally:
        simulation.stop()
    reference = stimulated_engine()
    reference.run_steps(int(round(time_ms / 0.1)))
    assert reference.counts[:, 0].tolist() == counts and counts[0] > 0 and counts[1] > 0


def test_slowdown_survives_pause_resume_and_reset():
    simulation = live.LiveSim(InstantEngine, STIMULI, frame_seconds=0.02, max_speed=None)
    simulation.start()
    try:
        _frames_after_speed(simulation, 0.5, 10)
        simulation.command({"cmd": "pause"})
        time.sleep(0.3)
        simulation.command({"cmd": "resume"})  # после паузы нет «рывка вдогонку»: время модели не обгоняет цель
        collector = Collector()
        simulation.subscribe(collector)
        collector.wait(lambda c: len(c.frames) >= 15, timeout=30)
        with collector.lock:
            speed = collector.frames[-1]["speed"]
            longest = max(f["end_ms"] - f["start_ms"] for f in collector.frames)
        assert speed < 0.8 and longest < 30  # кадр по 20 мс при 0,5× — это ~10 мс модели (плюс блок 1,8)
        simulation.command({"cmd": "reset"})
        collector.wait(lambda c: c.frames[-1]["epoch"] == 1, timeout=30)
        assert collector.statuses[-1]["target_speed"] == 0.5  # сброс не отменяет замедление
    finally:
        simulation.stop()


def test_engine_failure_is_reported_not_hidden():
    def broken() -> Engine:
        raise RuntimeError("CUDA out of memory\nподробности")

    simulation = live.LiveSim(broken, STIMULI)
    simulation.start()
    try:
        collector = Collector()
        simulation.subscribe(collector)
        collector.wait(lambda c: c.statuses and c.statuses[-1]["state"] == "error")
        assert "CUDA out of memory" in collector.statuses[-1]["message"]
    finally:
        simulation.stop()


needs_data = pytest.mark.skipif(
    not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists(), reason="нет data/processed — запустите build-data"
)


@needs_data
def test_websocket_streams_sugar_on_real_connectome():
    sugar = set(processed.load_groups()["stimuli"]["sugar"]["indices"])
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/live") as socket:
            status = socket.receive_json()
            assert status["type"] == "status" and set(status["stimuli"]) == {"sugar", "bitter", "antenna", "shadow"}
            assert not any(state["on"] for state in status["stimuli"].values())
            while status["state"] != "running":
                message = socket.receive()
                if message.get("text"):
                    status = json.loads(message["text"])
                    assert status["state"] != "error", status["message"]
            socket.send_json({"cmd": "stimulus", "key": "sugar", "on": True, "hz": 100, "side": "left"})
            fired: set[int] = set()
            for _ in range(400):  # ~16 с кадров с запасом; обычно хватает пары секунд
                message = socket.receive()
                if message.get("bytes"):
                    frame = live.decode_frame(message["bytes"])
                    fired.update(frame["neurons"].tolist())
                    if sugar <= fired and len(fired - sugar) > 20:
                        break
            assert sugar <= fired  # все 20 вкусовых нейронов сахара выстрелили
            assert len(fired - sugar) > 20  # и от них загорелись другие нейроны
            socket.send_json({"cmd": "stimulus", "key": "sugar", "on": False})


@needs_data
def test_right_side_sugar_is_the_paper_list_for_the_right_side():
    groups = processed.load_groups()["stimuli"]
    with TestClient(create_app()) as client:
        panel = client.get("/api/stimuli").json()
    sugar = next(b for b in panel["buttons"] if b["key"] == "sugar")
    assert [s["side"] for s in sugar["sides"]] == ["left", "right"] and not sugar["side_locked"]
    assert [s["count"] for s in sugar["sides"]] == [20, 9]
    assert groups["sugar_right"]["paper"] == "figures.ipynb: neu_sugar_left"
