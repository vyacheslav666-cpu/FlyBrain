"""Тесты «живого» мозга (коммит C9): поток кадров спайков, пауза, сброс, тест, скорость, WebSocket сервера."""

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
    # Нейрон 0 — «вкусовой» (его включает тест), нейрон 1 получает от него сильную возбуждающую связь.
    return Engine(np.array([0, 0, 1]), np.array([0]), np.array([1000.0]), n_trials=1, device="cpu", seed=7)


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
    simulation = live.LiveSim(_tiny_engine, test_neurons=[0], frame_seconds=0.02, max_speed=None)
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
    simulation = live.LiveSim(_tiny_engine, test_neurons=[0], frame_seconds=0.02)
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
    sim.command({"cmd": "test", "on": True})
    collector.wait(lambda c: sum(int(f["spikes"].sum()) for f in c.frames) > 200)
    sim.command({"cmd": "pause"})
    collector.wait(lambda c: c.statuses[-1]["paused"])
    time.sleep(0.1)  # кадр, который считался в момент команды, успевает прийти
    with collector.lock:
        frames = list(collector.frames)
        test_on = [s for s in collector.statuses if s["test"]]
    assert test_on and test_on[0]["test_count"] == 1
    # Кадры идут встык: конец одного — начало следующего, ничего не пропало.
    for before, after in zip(frames, frames[1:]):
        assert after["start_ms"] == pytest.approx(before["end_ms"])
    totals = np.zeros(2, dtype=np.int64)
    for frame in frames:
        np.add.at(totals, frame["neurons"].astype(np.int64), frame["spikes"])
    assert totals.tolist() == sim.engine.counts[:, 0].tolist()
    assert totals[0] > 0 and totals[1] > 0  # тест включил нейрон 0, а он — соседа


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
    sim.command({"cmd": "test", "on": True})
    collector.wait(lambda c: c.frames and c.frames[-1]["end_ms"] > 50)
    sim.command({"cmd": "reset"})
    collector.wait(lambda c: c.frames and c.frames[-1]["epoch"] == 1)
    with collector.lock:
        first_new = next(f for f in collector.frames if f["epoch"] == 1)
        status = next(s for s in collector.statuses if s["epoch"] == 1)
    assert first_new["start_ms"] == 0.0
    assert status["test"] is True  # сброс не выключает тест


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


def test_speed_is_not_faster_than_real_time():
    simulation = live.LiveSim(InstantEngine, test_neurons=[0], frame_seconds=0.02)  # MAX_SPEED = 1
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


def test_engine_failure_is_reported_not_hidden():
    def broken() -> Engine:
        raise RuntimeError("CUDA out of memory\nподробности")

    simulation = live.LiveSim(broken, test_neurons=[0])
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
def test_websocket_streams_sugar_test_on_real_connectome():
    sugar = set(processed.load_groups()["stimuli"]["sugar"]["indices"])
    with TestClient(create_app()) as client:
        with client.websocket_connect("/ws/live") as socket:
            status = socket.receive_json()
            assert status["type"] == "status" and status["test_count"] == 20
            while status["state"] != "running":
                message = socket.receive()
                if message.get("text"):
                    status = json.loads(message["text"])
                    assert status["state"] != "error", status["message"]
            socket.send_json({"cmd": "test", "on": True})
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
            socket.send_json({"cmd": "test", "on": False})
