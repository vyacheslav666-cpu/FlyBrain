"""P1b: откуда берётся «зажигание» клеток Кеньона и гаснет ли оно после дозы (D-058, D-059, D-061).

1. **Варианты связей** (`variant_csr`). [ЧЕСТНОСТЬ] Это не новая модель, а проверка свойства модели Shiu: копия
   матрицы связей, в которой одна группа синапсов убрана или усилена. Веса связей в основных расчётах не меняются.
   - `no_kc_kc` — убраны синапсы клеток Кеньона (KC) друг на друга;
   - `apl_x2`, `apl_x4` — все выходные синапсы APL (тормозный нейрон грибовидных тел) ×2 и ×4;
   - `no_dan_kc` — убраны синапсы дофаминовых нейронов (DAN) на KC: в модели Shiu дофамин — быстрое возбуждение;
   - `no_dpm_kc` — убраны синапсы DPM на KC (DPM в модели возбуждающий: предсказан дофамин, D-025);
   - `no_alln_exc` — убраны все выходные синапсы локальных нейронов усиковой доли (ALLN) со знаком +1 в модели;
   - `alln_gaba_fix` — те из них, у кого по литературе (`known_nt`) медиатор ГАМК, сделаны тормозными;
   - `no_alln_exc_no_kc_kc` — два изменения вместе: без возбуждающих ALLN и без KC → KC.
   Последние три добавлены, когда первые показали, что петля — не в грибовидных телах (D-058).
2. **Гистерезис** (`run_hysteresis`): доза до уровня разгона, «зажигание», затем доза обратно в 0; потом шум
   выключается, потом потенциалы сбрасываются в покой. Доза и шум меняются на ходу теми же средствами движка,
   что есть с P1 (`Engine.set_threshold_shift`, `Engine.set_noise`), — движок не менялся.
"""

import json
import time
import zlib

import numpy as np

from flybrain.psy import KC_CLASS, PSY_DIR, RESULT_FORMAT, RunSpec, Setup
from flybrain.serotonin import mentions
from flybrain.validate import engine_version

VARIANTS = {
    "base": "связи модели Shiu без изменений",
    "no_kc_kc": "без синапсов KC → KC",
    "apl_x2": "выход APL ×2",
    "apl_x4": "выход APL ×4",
    "no_dan_kc": "без синапсов DAN → KC",
    "no_dpm_kc": "без синапсов DPM → KC",
    "no_alln_exc": "без выходов возбуждающих локальных нейронов усиковой доли",
    "alln_gaba_fix": "локальные нейроны усиковой доли, по литературе ГАМК, — тормозные",
    "no_alln_exc_no_kc_kc": "без выходов возбуждающих локальных нейронов и без KC → KC",
}
IGNITED_HZ = 100_000  # «горит»: больше 100 тыс. спайков в секунду во всём мозге (обычный ответ — до ~40 тыс.)
QUIET_HZ = 1_000  # «погасла»: меньше 1 тыс. спайков в секунду и дальше не поднимается до конца фазы


def variant_description(variant: str) -> str:
    return f"{variant}: {VARIANTS[variant]}"


def groups_for_variants(setup: Setup) -> dict[str, np.ndarray]:
    neurons = setup.neurons
    alln = (neurons["cell_class"] == "ALLN").to_numpy()
    excitatory = (neurons["sign"] > 0).to_numpy()
    gaba = np.array([mentions(k, "gaba") for k in setup.sets.known_nt])
    return {
        "kc": (neurons["cell_class"] == KC_CLASS).to_numpy(),
        "apl": (neurons["cell_type"] == "APL").to_numpy(),
        "dan": (neurons["cell_class"] == "DAN").to_numpy(),
        "dpm": (neurons["cell_type"] == "DPM").to_numpy(),
        "alln": alln,
        "alln_exc": alln & excitatory,  # локальные нейроны усиковой доли со знаком +1 в модели Shiu
        "alln_gaba_exc": alln & excitatory & gaba,  # …у которых по литературе (known_nt) медиатор — ГАМК
    }


def variant_csr(setup: Setup, variant: str) -> dict:
    """Копия матрицы связей (CSR: строка — получатель) с изменённой группой синапсов."""
    csr = setup.csr
    g = groups_for_variants(setup)
    n = len(csr["indptr"]) - 1
    receiver = np.repeat(np.arange(n), np.diff(csr["indptr"]))
    sender = csr["indices"]
    weight = csr["weight"].astype(np.int32)  # ×4 не помещается в int16
    if variant == "no_kc_kc":
        weight[g["kc"][sender] & g["kc"][receiver]] = 0
    elif variant in ("apl_x2", "apl_x4"):
        weight[g["apl"][sender]] *= int(variant[-1])
    elif variant == "no_dan_kc":
        weight[g["dan"][sender] & g["kc"][receiver]] = 0
    elif variant == "no_dpm_kc":
        weight[g["dpm"][sender] & g["kc"][receiver]] = 0
    elif variant == "no_alln_exc":
        weight[g["alln_exc"][sender]] = 0
    elif variant == "alln_gaba_fix":
        weight[g["alln_gaba_exc"][sender]] *= -1
    elif variant == "no_alln_exc_no_kc_kc":
        weight[g["alln_exc"][sender]] = 0
        weight[g["kc"][sender] & g["kc"][receiver]] = 0
    else:
        raise ValueError(f"нет такого варианта связей: {variant}")
    return {"indptr": csr["indptr"], "indices": csr["indices"], "weight": weight}


def circuit_facts(setup: Setup) -> dict:
    """Числа о схеме грибовидных тел для отчёта: сколько синапсов в каждой петле и какие знаки."""
    csr, neurons = setup.csr, setup.neurons
    g = groups_for_variants(setup)
    n = len(csr["indptr"]) - 1
    receiver = np.repeat(np.arange(n), np.diff(csr["indptr"]))
    sender, weight = csr["indices"], csr["weight"].astype(np.int64)

    def link(a: str, b: str) -> tuple[int, int]:
        m = g[a][sender] & g[b][receiver]
        return int(m.sum()), int(weight[m].sum())

    kc = np.flatnonzero(g["kc"])
    per_kc = {}
    for name in ("kc", "apl", "dan", "dpm"):
        m = g[name][sender] & g["kc"][receiver]
        per_kc[name] = float(np.bincount(receiver[m], weights=weight[m], minlength=n)[kc].mean())
    pn = (neurons["cell_class"] == "ALPN").to_numpy()
    m = pn[sender] & g["kc"][receiver]
    per_kc["alpn"] = float(np.bincount(receiver[m], weights=weight[m], minlength=n)[kc].mean())
    apl = np.flatnonzero(g["apl"])
    return {
        "counts": {k: int(v.sum()) for k, v in g.items()},
        "apl": [{"side": str(neurons.at[i, "side"]), "nt": str(neurons.at[i, "nt"]), "sign": int(neurons.at[i, "sign"])}
                for i in apl],
        "dpm_sign": sorted({int(s) for s in neurons["sign"].to_numpy()[g["dpm"]]}),
        "dan_sign": {int(k): int(v) for k, v in zip(*np.unique(neurons["sign"].to_numpy()[g["dan"]],
                                                              return_counts=True))},
        "kc_nt": {str(k): int(v) for k, v in neurons["nt"][g["kc"]].value_counts().items()},
        "links": {f"{a}->{b}": link(a, b) for a, b in (("kc", "kc"), ("kc", "apl"), ("apl", "kc"), ("kc", "dan"),
                                                        ("dan", "kc"), ("kc", "dpm"), ("dpm", "kc"))},
        "per_kc_input": per_kc,
    }


# ---------- гистерезис ----------

def hysteresis_plan(setup: Setup) -> list[dict]:
    """Протоколы: для каждого режима — с шумом σ (без стимула) и без шума (зажигание стимулом)."""
    config, b = setup.config, setup.config.p1b
    doses = {"p1": config.runaway_dose, "kc_inhibit": b.kc_runaway_dose}
    plans = []
    for mode, dose in doses.items():
        if dose is None:
            continue
        plans.append({"mode": mode, "protocol": "noise", "dose": dose, "sigma": b.hyst_sigma_mv, "condition": None})
        plans.append({"mode": mode, "protocol": "stimulus", "dose": dose, "sigma": 0.0,
                      "condition": b.hyst_condition})
    return plans


def _key(p: dict, trials: int) -> str:
    return f"hyst__{p['mode']}__{p['protocol']}__d{p['dose']:g}__s{p['sigma']:g}__t{trials}"


def _meta(setup: Setup, p: dict) -> str:
    b = setup.config.p1b
    return json.dumps({"format": RESULT_FORMAT, "plan": p, "engine": engine_version(), "sha": setup.sha,
                       "delta_v_mv": setup.config.delta_v_mv, "s_half": setup.config.s_half,
                       "phases_ms": [b.hyst_on_ms, b.hyst_back_ms, b.hyst_quiet_ms, b.hyst_reset_ms],
                       "trials": b.hyst_trials, "bin_ms": setup.config.bin_ms}, sort_keys=True)


def _run_one(setup: Setup, p: dict, device: str) -> dict:
    """Фазы: A — доза (и стимул), B — доза 0, C — доза 0 и шум выключен, D — сброс потенциалов, затем шум снова."""
    import torch

    from flybrain.engine import Engine

    b, config = setup.config.p1b, setup.config
    started = time.monotonic()
    spec = RunSpec("hyst", p["condition"], 0.0, p["dose"], config.criterion, p["sigma"], b.hyst_trials, 0.0,
                   b.hyst_on_ms, 0.0, mode=p["mode"])
    csr = setup.csr
    engine = Engine(csr["indptr"], csr["indices"], csr["weight"], n_trials=b.hyst_trials, device=device,
                    seed=zlib.crc32(f"psy:{_key(p, b.hyst_trials)}".encode("utf-8")))
    stim = setup.stimulated(config.condition(p["condition"]).group if p["condition"] else None)
    stim_t = torch.as_tensor(stim, device=engine.device)
    dt = engine.params.dt
    chunk = int(round(config.bin_ms / dt))
    offset = np.zeros(b.hyst_trials, dtype=np.int64)  # спайки до сброса (сброс обнуляет счётчики движка)
    clock = [0]  # шаги с начала опыта (сброс обнуляет и время движка)
    marks: list[tuple[int, np.ndarray]] = [(0, np.zeros(b.hyst_trials, dtype=np.int64))]

    def others() -> np.ndarray:
        total = engine.counts.sum(0)
        if len(stim):
            total = total - engine.counts[stim_t].sum(0)
        return total.cpu().numpy().astype(np.int64) + offset

    def run(ms: float) -> None:
        until = engine.step + int(round(ms / dt))
        while engine.step < until:
            edge = min(until, (engine.step // chunk + 1) * chunk)
            taken = edge - engine.step
            engine.run_steps(taken)
            clock[0] += taken
            marks.append((clock[0], others()))

    phases = []
    engine.set_threshold_shift(setup.shift(spec))
    engine.set_noise(p["sigma"])
    if len(stim):
        engine.set_stimulus("excite", stim, config.condition(p["condition"]).hz)
    run(b.hyst_on_ms)
    phases.append(("A", clock[0]))
    if len(stim):
        engine.remove_stimulus("excite")
    engine.set_threshold_shift(None)  # доза 0
    run(b.hyst_back_ms)
    phases.append(("B", clock[0]))
    engine.set_noise(0.0)
    run(b.hyst_quiet_ms)
    phases.append(("C", clock[0]))
    offset[:] = others()
    engine.reset()  # сброс: все потенциалы в покое, синаптические токи и «отдых» обнулены
    engine.set_noise(p["sigma"])
    run(b.hyst_reset_ms)
    phases.append(("D", clock[0]))
    on_cuda = engine.device.type == "cuda"
    del engine
    if on_cuda:
        torch.cuda.empty_cache()
    steps = np.array([s for s, _ in marks])
    return {"edges_ms": steps * dt, "cumulative": np.stack([v for _, v in marks]),
            "phases": [(name, step * dt) for name, step in phases], "seconds": time.monotonic() - started}


def run_hysteresis(setup: Setup, device: str = "auto", rerun: bool = False, compute: bool = True,
                   log=print) -> list[dict]:
    """Готовые (или посчитанные сейчас) опыты гистерезиса: план + данные + разбор по повторам."""
    if setup.config.p1b is None:
        return []
    out = []
    trials = setup.config.p1b.hyst_trials
    for p in hysteresis_plan(setup):
        path = PSY_DIR / f"{_key(p, trials)}.npz"
        meta = _meta(setup, p)
        data = None
        if path.exists() and not rerun:
            with np.load(path, allow_pickle=False) as a:
                if str(a["meta"]) == meta:
                    data = {"edges_ms": a["edges_ms"], "cumulative": a["cumulative"],
                            "phases": json.loads(str(a["phases"])), "seconds": float(a["seconds"])}
        if data is None and compute:
            log(f"  гистерезис: {_key(p, trials)} …", end="", flush=True)
            data = _run_one(setup, p, device)
            PSY_DIR.mkdir(parents=True, exist_ok=True)
            part = path.with_name(path.name + ".part")
            with open(part, "wb") as handle:
                np.savez_compressed(handle, edges_ms=data["edges_ms"], cumulative=data["cumulative"],
                                    phases=np.array(json.dumps(data["phases"])), seconds=np.array(data["seconds"]),
                                    meta=np.array(meta))
            part.replace(path)
            log(f" {data['seconds']:.0f} с")
        if data is not None:
            out.append({"plan": p, **data, "summary": summarize(data)})
    return out


def rate_series(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Время конца каждого отрезка, мс, и спайки в секунду во всём мозге по повторам: (E−1,), (E−1, повторы)."""
    edges, cum = np.asarray(data["edges_ms"]), np.asarray(data["cumulative"])
    widths = np.diff(edges) / 1000
    return edges[1:], np.diff(cum, axis=0) / widths[:, None]


def summarize(data: dict) -> dict:
    """По повторам: горит ли в конце каждой фазы и когда погасла после возврата дозы в 0."""
    t, rate = rate_series(data)
    phases = dict((name, at) for name, at in data["phases"])
    bounds = [0.0] + [at for _, at in data["phases"]]
    names = [name for name, _ in data["phases"]]
    lit = {}
    for name, lo, hi in zip(names, bounds, bounds[1:]):
        tail = (t > hi - 100) & (t <= hi)  # последние 100 мс фазы
        lit[name] = (rate[tail].mean(0) > IGNITED_HZ).tolist()
    # Когда погасла после возврата дозы в 0 (фаза B): первый момент, после которого до конца фазы < QUIET_HZ.
    start, end = phases["A"], phases["B"]
    inside = (t > start) & (t <= end)
    times = t[inside]
    out_ms = []
    for k in range(rate.shape[1]):
        r = rate[inside, k]
        loud = np.flatnonzero(r >= QUIET_HZ)
        if len(loud) == 0:
            out_ms.append(0.0)
        elif loud[-1] == len(r) - 1:
            out_ms.append(None)  # горит до конца фазы
        else:
            out_ms.append(float(times[loud[-1]] - start))  # конец последнего «громкого» отрезка
    return {"lit": lit, "out_ms": out_ms}
