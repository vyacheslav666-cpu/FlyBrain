"""Движок C5 на крошечных сетях, где ответ считается на бумаге, и сверка с неизменённым `model.py` Shiu."""

import importlib.util
import math

import numpy as np
import pandas as pd
import pytest
import torch

from flybrain import processed
from flybrain.engine import SHIU, Engine, EngineError
from flybrain.paths import PROCESSED_DIR, RAW_DIR

SHIU_MODEL = RAW_DIR / "shiu" / "model.py"
V0, VTH = SHIU.v_0, SHIU.v_th
A, B, E = SHIU.decay_factors()
EVERY_STEP_HZ = 1 / (SHIU.dt / 1000)  # 10 000 Гц: импульс стимула в каждом шаге, без случайности
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def csr(n: int, edges: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Связи (отправитель, получатель, синапсы со знаком) → CSR, как в data/processed (строка — получатель)."""
    edges = sorted(edges, key=lambda edge: (edge[1], edge[0]))
    indptr = np.zeros(n + 1, dtype=np.int64)
    for _, post, _ in edges:
        indptr[post + 1] += 1
    return (np.cumsum(indptr), np.array([pre for pre, _, _ in edges], dtype=np.int32),
            np.array([w for _, _, w in edges], dtype=np.int16))


def engine(n, edges, device="cpu", **kwargs) -> Engine:
    kwargs.setdefault("dtype", torch.float64)
    return Engine(*csr(n, edges), device=device, seed=0, record_spikes=True, **kwargs)


def spike_steps(eng: Engine, neuron: int, trial: int = 0) -> list[int]:
    events = eng.spikes()
    return [int(s) for s, i, k in events if i == neuron and k == trial]


def trace(eng: Engine, neuron: int, steps: int) -> tuple[list[float], list[float]]:
    """v и g нейрона после каждого шага (по одному шагу за вызов)."""
    v, g = [], []
    for _ in range(steps):
        eng.run_steps(1)
        v.append(float(eng.v[neuron, 0]))
        g.append(float(eng.g[neuron, 0]))
    return v, g


def test_parameters_are_shiu_defaults():
    # D-005: числа движка — те же, что default_params в model.py Shiu (Brian2 хранит вольты и секунды).
    if not SHIU_MODEL.exists():
        pytest.skip("нет data/raw/shiu/model.py — запустите fetch-data")
    spec = importlib.util.spec_from_file_location("shiu_model", SHIU_MODEL)
    shiu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shiu)
    import brian2

    p = shiu.default_params
    mv, ms = brian2.mV, brian2.ms
    assert (SHIU.v_0, SHIU.v_rst, SHIU.v_th) == pytest.approx((p["v_0"] / mv, p["v_rst"] / mv, p["v_th"] / mv))
    assert (SHIU.t_mbr, SHIU.tau, SHIU.t_rfc, SHIU.t_dly) == pytest.approx(
        (p["t_mbr"] / ms, p["tau"] / ms, p["t_rfc"] / ms, p["t_dly"] / ms))
    assert SHIU.w_syn == pytest.approx(p["w_syn"] / mv) and SHIU.f_poi == p["f_poi"]
    assert SHIU.dt == pytest.approx(brian2.defaultclock.dt / ms)
    assert (SHIU.delay_steps, SHIU.refractory_steps, SHIU.poisson_weight) == (18, 22, pytest.approx(68.75))


@pytest.mark.parametrize("device", DEVICES)
def test_neighbour_feels_spike_after_1_8_ms(device):
    # A выстреливает в шаге 0. Сигнал идёт 18 шагов: в шаге 18 у соседа B прыгает g (на 10 синапсов
    # × 0,275 = 2,75 мВ), а v начинает расти только со следующего шага 19 — на g·B.
    eng = engine(2, [(0, 1, 10)], device)
    eng.v[0] = -40.0  # выше порога: спайк в первом же шаге
    v, g = trace(eng, 1, 21)
    assert spike_steps(eng, 0) == [0]
    w = 10 * 0.275
    assert g[:18] == [0.0] * 18 and v[:19] == [V0] * 19
    assert g[18] == pytest.approx(w) and g[19] == pytest.approx(w * E)
    assert v[19] == pytest.approx(V0 + w * B)
    assert v[20] == pytest.approx(V0 + w * B * A + w * E * B)


@pytest.mark.parametrize("device", DEVICES)
def test_refractory_2_2_ms(device):
    # S1 и S2 (стимулы 10 кГц) стреляют через шаг со сдвигом, так что вход у A приходит в каждом шаге
    # начиная с 18. A выстреливает в шаге 0 и «отдыхает» 22 шага (2,2 мс): v и g заморожены, а входы
    # в шагах 18…21 теряются (как в Brian2). Первый принятый вход — в шаге 22, спайк — в шаге 23.
    # 6000 синапсов × 0,275 = 1650 мВ в g за шаг поднимают v на 1650·B ≈ 8,1 мВ — сразу выше порога (7 мВ).
    eng = engine(3, [(1, 0, 6000), (2, 0, 6000)], device)
    eng.set_stimulus("s", [1, 2], EVERY_STEP_HZ)
    eng.v[0] = -40.0
    eng.v[2] = -40.0  # S2 стреляет в чётных шагах, S1 — в нечётных
    v, g = trace(eng, 0, 24)
    assert spike_steps(eng, 1)[:3] == [1, 3, 5] and spike_steps(eng, 2)[:3] == [0, 2, 4]
    assert spike_steps(eng, 0) == [0, 23]
    assert v[:22] == [SHIU.v_rst] * 22 and g[:22] == [0.0] * 22
    assert g[22] == pytest.approx(6000 * 0.275)
    # Без «отдыха» (стимулируемый нейрон, rfc = 0, как в poi() Shiu): свой же вход из шага 18 принят.
    loop = engine(1, [(0, 0, 6000)], device)
    loop.set_stimulus("s", [0], 0.0)
    loop.v[0] = -40.0
    loop.run_steps(20)
    assert spike_steps(loop, 0) == [0, 19]


@pytest.mark.parametrize("device", DEVICES)
def test_inhibitory_link_inhibits(device):
    # S (10 кГц) возбуждает B через 15 синапсов; I (тоже 10 кГц, в тех же шагах) — тормозит B.
    # Без торможения B стреляет; при −15 синапсах сумма входов 0 — молчит; при −30 v уходит ниже покоя.
    def run(inhibition):
        edges = [(0, 1, 15)] + ([(2, 1, inhibition)] if inhibition else [])
        eng = engine(3, edges, device)
        eng.set_stimulus("s", [0, 2], EVERY_STEP_HZ)
        eng.run_steps(18 * 20)
        return eng
    assert len(spike_steps(run(0), 1)) > 0
    assert spike_steps(run(-15), 1) == []
    strong = run(-30)
    assert spike_steps(strong, 1) == [] and float(strong.v[1, 0]) < V0


@pytest.mark.parametrize("device", DEVICES)
def test_silenced_neuron_spikes_but_reaches_nobody(device):
    eng = engine(2, [(0, 1, 2000)], device)
    eng.set_stimulus("s", [0], EVERY_STEP_HZ)
    eng.set_silenced([0])
    eng.run_steps(100)
    assert len(spike_steps(eng, 0)) == 50 and spike_steps(eng, 1) == []
    eng.set_silenced([0], silenced=False)  # на ходу: спайки следующего блока уже доходят
    eng.run_steps(100)
    assert len(spike_steps(eng, 1)) > 0


@pytest.mark.parametrize("device", DEVICES)
def test_poisson_rate_and_change_on_the_fly(device):
    # Импульс приходит с вероятностью p = hz·dt за шаг; спайк — в следующем шаге, но импульс в шаг
    # самого спайка теряется. Доля шагов со спайком s = p·(1 − s), то есть s = p / (1 + p):
    # для 100 Гц это 99,01 Гц, а не 100.
    eng = Engine(*csr(1, []), n_trials=200, device=device, seed=1)
    eng.set_stimulus("s", [0], 100.0)
    eng.run_ms(1000)
    p = 100 * SHIU.dt / 1000
    expected = p / (1 + p) / (SHIU.dt / 1000)
    assert eng.rates_hz().mean() == pytest.approx(expected, rel=0.02)
    eng.set_stimulus("s", [0], 0.0)  # та же форма: на видеокарте записанный блок читает новую частоту
    eng.run_steps(1)  # импульс из последнего шага до выключения ещё даёт спайк — не больше одного на повтор
    before = eng.counts.sum().item()
    eng.run_ms(100)
    assert eng.counts.sum().item() == before
    eng.set_stimulus("s", [0], 200.0)
    eng.run_ms(500)
    assert (eng.counts.sum().item() - before) / (200 * 0.5) == pytest.approx(
        0.02 / 1.02 / (SHIU.dt / 1000), rel=0.03)


def test_bad_inputs_are_errors():
    eng = engine(2, [(0, 1, 1)])
    with pytest.raises(EngineError):
        eng.set_stimulus("s", [5], 100.0)
    with pytest.raises(EngineError):
        eng.set_stimulus("s", [0], 20000.0)
    with pytest.raises(EngineError):
        Engine(*csr(2, []), n_trials=0, device="cpu")


def _shiu_tiny_spikes(tmp_path, edges, n, exc, exc2, duration_ms):
    """Неизменённый run_trial Shiu на крошечной сети; Brian2 без компилятора (numpy), чтобы было быстро."""
    import brian2

    spec = importlib.util.spec_from_file_location("shiu_model", SHIU_MODEL)
    shiu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shiu)
    pd.DataFrame({"x": np.zeros(n)}, index=pd.Index(np.arange(n) + 10**17, name="root_id")).to_csv(
        tmp_path / "comp.csv")
    pd.DataFrame({"Presynaptic_Index": [e[0] for e in edges], "Postsynaptic_Index": [e[1] for e in edges],
                  "Excitatory x Connectivity": [e[2] for e in edges]}).to_parquet(tmp_path / "con.parquet")
    params = dict(shiu.default_params)
    params["r_poi"] = params["r_poi2"] = EVERY_STEP_HZ * brian2.Hz
    params["t_run"] = duration_ms * brian2.ms
    target = brian2.prefs.codegen.target
    brian2.prefs.codegen.target = "numpy"
    try:
        trains = shiu.run_trial(exc, exc2, [], tmp_path / "comp.csv", tmp_path / "con.parquet", params)
    finally:
        brian2.prefs.codegen.target = target
    return sorted((int(round(float(t / brian2.second) / (SHIU.dt / 1000))), int(i))
                  for i, times in trains.items() for t in times)


@pytest.mark.parametrize("device", DEVICES)
def test_matches_unmodified_shiu_spike_for_spike(tmp_path, device):
    # Стимулы 10 кГц — без случайности, поэтому Brian2 и движок обязаны совпасть спайк в спайк.
    # Сеть: возбуждение, торможение, петля на себя, нейрон-стимул во второй группе (r_poi2).
    if not SHIU_MODEL.exists():
        pytest.skip("нет data/raw/shiu/model.py — запустите fetch-data")
    edges = [(0, 1, 40), (1, 2, 30), (3, 2, -20), (0, 3, 50), (2, 2, 10), (4, 1, 25), (4, 2, 60), (1, 5, 300),
             (5, 0, -400)]
    n = 6
    reference = _shiu_tiny_spikes(tmp_path, edges, n, [0], [4], 60)
    for chunks in ([600], [7, 30, 100, 463]):  # целые блоки и куски, не кратные 18 шагам
        eng = Engine(*csr(n, edges), n_trials=2, device=device, seed=3, record_spikes=True)
        eng.set_stimulus("exc", [0], EVERY_STEP_HZ)
        eng.set_stimulus("exc2", [4], EVERY_STEP_HZ)
        for steps in chunks:
            eng.run_steps(steps)
        events = eng.spikes()
        for trial in range(2):
            mine = sorted((int(s), int(i)) for s, i, k in events if k == trial)
            assert mine == reference
    assert {i for _, i in reference} == {0, 1, 2, 3, 4, 5}


@pytest.mark.skipif(not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists() or not torch.cuda.is_available(),
                    reason="нужны data/processed и видеокарта")
def test_sugar_reaches_mn9_on_real_connectome():
    csr_data = processed.load_connectivity()
    groups = processed.load_groups()
    eng = Engine(csr_data["indptr"], csr_data["indices"], csr_data["weight"], n_trials=4, seed=5)
    eng.set_stimulus("sugar", groups["stimuli"]["sugar"]["indices"], 100.0)
    eng.run_ms(300)
    rates = eng.rates_hz().mean(axis=0)
    sugar = groups["stimuli"]["sugar"]["indices"]
    assert rates[sugar].mean() == pytest.approx(99.0, abs=5)
    assert rates[groups["readouts"]["MN9"]["indices"]].max() > 10
    assert math.isfinite(float(eng.v.sum()))
