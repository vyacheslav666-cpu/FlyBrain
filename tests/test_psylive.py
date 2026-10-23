"""Панель «Психоделик (эксперимент)» в живом режиме (P2, D-064…D-066): доза, режим, шум, индикатор «зажглась»."""

import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from flybrain import live, processed, psylive
from flybrain.engine import Engine
from flybrain.paths import PROCESSED_DIR, RAW_DIR
from flybrain.psylive import IgnitionMeter, PsyLive
from flybrain.server import create_app
from tests.test_live import Collector

HAVE_DATA = (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists() and \
    (RAW_DIR / "flywire" / "Supplemental_file1_neuron_annotations.tsv").exists()


def _tiny_psy() -> PsyLive:
    # Нейрон 0 — стимул; 1 получает 5 серотониновых синапсов; 2 — клетка Кеньона с 10; 1 — «петля усиковой доли».
    return PsyLive(True, "", synapses=np.array([0, 5, 10]), serotonin=np.array([0]),
                   kc=np.array([False, False, True]), alln_exc=np.array([1]), delta_v_mv=1.0, s_half=5.0)


def _tiny_engine() -> Engine:
    # 0 → 1 (сильная связь) и 0 → 2; стимул на 0.
    return Engine(np.array([0, 0, 1, 2]), np.array([0, 0]), np.array([1000.0, 1000.0]), n_trials=1, device="cpu",
                  seed=7)


def test_meter_needs_a_full_window_and_the_p1b_threshold():
    meter = IgnitionMeter(al_total=4)
    # Кадры по 20 мс, по 3 000 спайков: 150 тыс. спайков в секунду — выше порога 100 тыс.
    for k in range(4):
        meter.add(20.0 * k, 20.0 * (k + 1), 3000, np.array([k == 0, False, k == 1, False]))
        assert not meter.ignited  # окно ещё меньше 100 мс
    meter.add(80.0, 100.0, 3000, np.zeros(4, dtype=bool))
    assert meter.span_ms == pytest.approx(100.0) and meter.rate_hz == pytest.approx(150_000)
    assert meter.ignited and meter.al_active == 2
    # Окно сдвигается: старые кадры уходят, частота падает ниже порога.
    for k in range(5, 11):
        meter.add(20.0 * k, 20.0 * (k + 1), 1000, np.zeros(4, dtype=bool))
    assert meter.rate_hz == pytest.approx(50_000) and not meter.ignited and meter.al_active == 0
    # Разрыв во времени (сброс модели) — окно заново.
    meter.add(0.0, 20.0, 9000, np.ones(4, dtype=bool))
    assert meter.span_ms == pytest.approx(20.0) and not meter.ignited
    reading = meter.reading()
    assert reading["type"] == "psy" and reading["al_active"] == 4 and reading["al_total"] == 4


def test_shift_follows_p1_and_p1b_and_is_none_at_dose_zero():
    psy = _tiny_psy()
    assert psy.shift(0.0, "p1") is None and psy.shift(0.0, "kc_inhibit") is None  # движок идёт прежним путём
    assert psy.shift(2.0, "p1") == pytest.approx([0.0, 1.0, 2 * 10 / 15])
    assert psy.shift(2.0, "kc_inhibit") == pytest.approx([0.0, 1.0, -2 * 10 / 15])  # у KC порог выше
    units = psy.mv_per_dose()
    assert units["median_s"] == 7.5 and units["median_mv"] == pytest.approx(7.5 / 12.5)


@pytest.fixture
def sim():
    simulation = live.LiveSim(_tiny_engine, {"sugar": {"left": [0]}}, frame_seconds=0.02, max_speed=None,
                              psy=_tiny_psy())
    simulation.start()
    yield simulation
    simulation.stop()


def test_dose_mode_and_noise_change_on_the_fly_and_zero_restores_the_old_engine(sim):
    collector = Collector()
    sim.subscribe(collector)
    collector.wait(lambda c: c.frames)
    sim.command({"cmd": "psy", "dose": 2, "mode": "kc_inhibit", "sigma": 1.5})
    collector.wait(lambda c: c.statuses[-1]["psy"]["dose"] == 2)
    assert collector.statuses[-1]["psy"] == {"available": True, "dose": 2.0, "mode": "kc_inhibit", "sigma": 1.5,
                                             "ignited": False}
    v_th = sim.engine._v_th[:, 0].tolist()
    assert v_th == pytest.approx([-45.0, -46.0, -45.0 + 2 * 10 / 15])
    assert sim.engine.noise_sigma == 1.5
    sim.command({"cmd": "psy", "dose": 0, "sigma": 0})
    collector.wait(lambda c: c.statuses[-1]["psy"]["dose"] == 0 and c.statuses[-1]["psy"]["sigma"] == 0)
    assert sim.engine._v_th is None and sim.engine.noise_sigma == 0  # те же операции, что без панели (D-055)


def test_bad_psy_commands_are_dropped_and_values_clamped(sim):
    collector = Collector()
    sim.subscribe(collector)
    collector.wait(lambda c: c.frames)
    for bad in ({"cmd": "psy", "dose": "много"}, {"cmd": "psy", "dose": float("nan")}, {"cmd": "psy", "mode": "x"},
                {"cmd": "psy", "sigma": True}):
        sim.command(bad)
    sim.command({"cmd": "psy", "dose": 99, "sigma": -1})
    collector.wait(lambda c: c.statuses[-1]["psy"]["dose"] == psylive.DOSE_MAX)
    assert collector.statuses[-1]["psy"]["sigma"] == 0 and collector.statuses[-1]["psy"]["mode"] == "p1"
    # Без данных панели команда «psy» не принимается, а в состоянии так и написано.
    plain = live.LiveSim(_tiny_engine, {"sugar": {"left": [0]}}, frame_seconds=0.02, max_speed=None)
    assert plain.status()["psy"]["available"] is False
    plain.command({"cmd": "psy", "dose": 3})
    assert plain._commands.empty()


def test_indicator_lights_up_and_reset_puts_it_out(sim):
    collector = Collector()
    sim.subscribe(collector)
    collector.wait(lambda c: c.frames)
    sim._meter.threshold_hz = 100.0  # крошечная сеть (~300 спайков/с): порог понижен, логика та же
    sim.command({"cmd": "stimulus", "key": "sugar", "on": True, "hz": 10_000})
    last_status = lambda c: [s for s in c.statuses if s["type"] == "status"][-1]  # noqa: E731
    collector.wait(lambda c: last_status(c)["psy"]["ignited"])
    readings = [s for s in collector.statuses if s.get("type") == "psy"]
    assert readings and readings[-1]["al_total"] == 1
    sim.command({"cmd": "stimulus", "key": "sugar", "on": False})
    sim.command({"cmd": "reset"})
    collector.wait(lambda c: not last_status(c)["psy"]["ignited"])


@pytest.mark.skipif(not HAVE_DATA or not torch.cuda.is_available(), reason="нужны данные и видеокарта")
def test_real_brain_ignites_at_dose_5_with_shadow_and_reset_puts_it_out():
    """Сверка с P1b (D-061, D-063): доза 5 + тень — сеть зажглась; доза 0 — не гаснет; сброс — гаснет."""
    from flybrain import stimuli

    csr = processed.load_connectivity()
    _, indices = stimuli.build_catalog(processed.load_groups(), stimuli.load_buttons())
    simulation = live.LiveSim(lambda: Engine(csr["indptr"], csr["indices"], csr["weight"], n_trials=1), indices,
                              max_speed=None, psy=PsyLive.load())
    simulation.start()
    try:
        collector = Collector()
        simulation.subscribe(collector)
        collector.wait(lambda c: c.frames, timeout=60)

        def run_ms(ms):
            start = simulation.engine.time_ms
            deadline = time.monotonic() + 60
            while simulation.engine.time_ms - start < ms and time.monotonic() < deadline:
                time.sleep(0.05)

        simulation.command({"cmd": "stimulus", "key": "shadow", "on": True, "hz": 100})
        run_ms(400)
        assert not simulation.status()["psy"]["ignited"]  # без дозы тень сеть не зажигает
        simulation.command({"cmd": "psy", "dose": 5})
        run_ms(1500)
        assert simulation.status()["psy"]["ignited"]
        simulation.command({"cmd": "stimulus", "key": "shadow", "on": False})
        simulation.command({"cmd": "psy", "dose": 0})
        run_ms(800)
        assert simulation.status()["psy"]["ignited"]  # гистерезис: доза 0 сеть не гасит
        simulation.command({"cmd": "reset"})
        run_ms(300)
        assert not simulation.status()["psy"]["ignited"]
    finally:
        simulation.stop()


@pytest.mark.skipif(not HAVE_DATA, reason="нужны данные")
def test_server_gives_panel_settings_and_layer():
    with TestClient(create_app()) as client:
        info = client.get("/api/psy").json()
    assert info["available"] and info["dose_max"] == 6 and info["sigma_ignites"] == 1.75
    assert len(info["serotonin"]) == 62 and len(info["targets"]) == 21_619 == len(info["target_synapses"])
    assert info["median_mv"] == pytest.approx(2 / 7) and info["max_mv"] == pytest.approx(1074 / 1079)
    assert info["al_total"] == 166 and info["ignited_hz"] == 100_000
