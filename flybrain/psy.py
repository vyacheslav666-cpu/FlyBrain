"""`flybrain psy`: эксперимент P1 «психоделик» в режиме «мозг в банке» (docs/EXPERIMENT_PSY.md, D-053…D-056).

[ЧЕСТНОСТЬ] Вычислительная гипотеза, а не модель того, что «чувствует» муха: нейроны, получающие синапсы от
серотониновых нейронов, становятся возбудимее (порог ниже), как от агониста рецептора 5-HT2A. Веса связей
и параметры модели Shiu (D-005) не меняются; при дозе 0 и без шума расчёт — тот же, что прошёл проверку C6.

Части опыта (`config/psy.yaml`):
1. `runaway` — поиск дозы разгона (сахар 40 Гц, дозы по возрастанию), по основному и по широкому критерию;
2. `main` — четыре стимула при дозах основного опыта и при дозе разгона;
3. `thresholds` — порог «исполнителя» по частоте стимула при каждой дозе;
4. `spont` — без стимула, с шумом мембраны σ;
5. `broad` — основной опыт с широким критерием серотониновых нейронов (проверка устойчивости).
Уточнение P1b (D-058…D-061, раздел `p1b` в `config/psy.yaml`):
6. `mech` — откуда «зажигание»: доза 0, шум σ на 1 с, затем шум выключен; связи модели меняются по вариантам
   (`flybrain.psymech`): без синапсов KC → KC, выход APL ×2 и ×4, без DAN → KC, без DPM → KC;
7. `kc_runaway`, `kc_main`, `kc_thresholds`, `kc_spont` — сетка P1 в режиме «KC тормозятся» (у клеток Кеньона
   порог от дозы не снижается, а повышается — как от рецептора 5-HT1A);
8. `hyst` — гистерезис: доза до разгона, затем обратно в 0 (`flybrain.psymech`).
Каждый расчёт сохраняется в `data/psy/` и при следующем запуске берётся оттуда, если ничего не изменилось.
Отчёты — `docs/psy/results.md` (P1, модуль `flybrain.psyreport`) и `docs/psy/p1b.md` (P1b, `flybrain.p1breport`).
"""

import json
import math
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flybrain import processed
from flybrain.modality import MODALITIES, input_shares
from flybrain.paths import CONFIG_DIR, PROCESSED_DIR, PROJECT_ROOT
from flybrain.serotonin import ANNOTATIONS_NAME, SerotoninError, SerotoninSets, read_annotations, \
    serotonin_sets, serotonin_synapses, threshold_shift
from flybrain.validate import engine_version

CONFIG_FILE = CONFIG_DIR / "psy.yaml"
PSY_DIR = PROJECT_ROOT / "data" / "psy"
MODALITY_CACHE = PSY_DIR / "modality.npz"
RESULT_FORMAT = 1
ACTIVE_HZ = 1.0  # «активный» — не меньше 1 Гц в среднем (D-018)
KEY_READOUTS = ("MN9", "aBN1", "GF", "aDN1", "aDN2")
VISUAL_SUPER_CLASSES = ("optic", "visual_projection", "visual_centrifugal")
P1_PARTS = ("runaway", "main", "thresholds", "spont", "broad")
P1B_PARTS = ("mech", "kc_runaway", "kc_main", "kc_thresholds", "kc_spont", "hyst")
PARTS = P1_PARTS + P1B_PARTS
MODES = ("p1", "kc_inhibit")  # P1: все мишени возбуждаются; P1b: клетки Кеньона тормозятся (D-060)
KC_CLASS = "Kenyon_Cell"


class PsyError(Exception):
    """Опыт нельзя запустить: нет данных, или настройки не сходятся."""


# ---------- настройки ----------

@dataclass(frozen=True)
class Condition:
    key: str
    name_ru: str
    group: str
    hz: float
    modality: str
    readout_name: str
    readout_root_id: str
    threshold_hz: tuple[float, ...]


@dataclass(frozen=True)
class PsyConfig:
    criterion: str
    sensitivity_criterion: str
    delta_v_mv: float
    s_half: float
    trials: int
    stimulus_ms: float
    after_ms: float
    bin_ms: float
    late_window_ms: tuple[float, float]
    after_window_ms: tuple[float, float]
    min_ratio: float
    runaway_condition: str
    runaway_trials: int
    runaway_doses: tuple[float, ...]
    doses: tuple[float, ...]
    runaway_dose: float | None
    conditions: tuple[Condition, ...]
    threshold_trials: int
    sigmas_mv: tuple[float, ...]
    spont_trials: int
    warmup_ms: float
    p1b: "P1bConfig | None" = None

    def condition(self, key: str) -> Condition:
        return next(c for c in self.conditions if c.key == key)

    @property
    def main_doses(self) -> tuple[float, ...]:
        """Дозы основного опыта и доза разгона (если найдена) — по возрастанию, без повторов."""
        extra = () if self.runaway_dose is None else (self.runaway_dose,)
        return tuple(sorted(set(self.doses) | set(extra)))


@dataclass(frozen=True)
class P1bConfig:
    """Уточнение P1b (D-058…D-061)."""
    mech_variants: tuple[str, ...]
    mech_sigmas_mv: tuple[float, ...]
    mech_trials: int
    mech_noise_ms: float
    mech_off_ms: float
    mech_dose_variant: str
    mech_dose_doses: tuple[float, ...]
    kc_runaway_doses: tuple[float, ...]
    kc_doses: tuple[float, ...]
    kc_runaway_dose: float | None
    kc_spont_sigma_mv: float
    hyst_trials: int
    hyst_sigma_mv: float
    hyst_condition: str
    hyst_on_ms: float
    hyst_back_ms: float
    hyst_quiet_ms: float
    hyst_reset_ms: float

    @property
    def kc_main_doses(self) -> tuple[float, ...]:
        extra = () if self.kc_runaway_dose is None else (self.kc_runaway_dose,)
        return tuple(sorted(set(self.kc_doses) | set(extra)))


def _load_p1b(raw: dict | None) -> P1bConfig | None:
    if not raw:
        return None
    mech, kc, hyst = raw["mechanism"], raw["kc_inhibit"], raw["hysteresis"]
    return P1bConfig(
        mech_variants=tuple(mech["variants"]), mech_sigmas_mv=tuple(float(s) for s in mech["sigmas_mv"]),
        mech_trials=int(mech["trials"]), mech_noise_ms=float(mech["noise_ms"]), mech_off_ms=float(mech["off_ms"]),
        mech_dose_variant=mech["dose_check"]["variant"],
        mech_dose_doses=tuple(float(d) for d in mech["dose_check"]["doses"]),
        kc_runaway_doses=tuple(float(d) for d in kc["runaway_doses"]), kc_doses=tuple(float(d) for d in kc["doses"]),
        kc_runaway_dose=None if kc.get("runaway_dose") is None else float(kc["runaway_dose"]),
        kc_spont_sigma_mv=float(kc["spont_sigma_mv"]), hyst_trials=int(hyst["trials"]),
        hyst_sigma_mv=float(hyst["sigma_mv"]), hyst_condition=hyst["condition"], hyst_on_ms=float(hyst["on_ms"]),
        hyst_back_ms=float(hyst["back_ms"]), hyst_quiet_ms=float(hyst["quiet_ms"]),
        hyst_reset_ms=float(hyst["reset_ms"]))


def load_config(path: Path = CONFIG_FILE) -> PsyConfig:
    with open(path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    conditions = tuple(Condition(
        key=c["key"], name_ru=c["name_ru"], group=c["group"], hz=float(c["hz"]), modality=c["modality"],
        readout_name=c["readout"]["name"], readout_root_id=str(c["readout"]["root_id"]),
        threshold_hz=tuple(float(f) for f in c["threshold_hz"])) for c in raw["conditions"])
    runaway = raw["runaway"]
    config = PsyConfig(
        criterion=raw["criterion"], sensitivity_criterion=raw["sensitivity_criterion"],
        delta_v_mv=float(raw["delta_v_mv"]), s_half=float(raw["s_half"]), trials=int(raw["trials"]),
        stimulus_ms=float(raw["stimulus_ms"]), after_ms=float(raw["after_ms"]), bin_ms=float(raw["bin_ms"]),
        late_window_ms=tuple(float(x) for x in runaway["late_window_ms"]),
        after_window_ms=tuple(float(x) for x in runaway["after_window_ms"]),
        min_ratio=float(runaway["min_ratio"]), runaway_condition=runaway["condition"],
        runaway_trials=int(runaway["trials"]), runaway_doses=tuple(float(d) for d in runaway["doses"]),
        doses=tuple(float(d) for d in raw.get("doses") or []),
        runaway_dose=None if raw.get("runaway_dose") is None else float(raw["runaway_dose"]),
        conditions=conditions, threshold_trials=int(raw["threshold_trials"]),
        sigmas_mv=tuple(float(s) for s in raw["spontaneous"]["sigmas_mv"]),
        spont_trials=int(raw["spontaneous"]["trials"]), warmup_ms=float(raw["spontaneous"]["warmup_ms"]),
        p1b=_load_p1b(raw.get("p1b")))
    keys = [c.key for c in conditions]
    if config.p1b and config.p1b.hyst_condition not in keys:
        raise PsyError("config/psy.yaml: стимул гистерезиса (p1b.hysteresis.condition) — из числа условий")
    if len(set(keys)) != len(keys) or config.runaway_condition not in keys:
        raise PsyError("config/psy.yaml: ключи условий должны быть разными, а условие поиска разгона — из их числа")
    if any(c.modality not in MODALITIES for c in conditions):
        raise PsyError(f"config/psy.yaml: модальность стимула — одна из {', '.join(MODALITIES)}")
    if any(d < 0 for d in (*config.doses, *config.runaway_doses)) or 0.0 not in config.runaway_doses:
        raise PsyError("config/psy.yaml: дозы не меньше 0, и в поиске разгона есть доза 0")
    if config.after_window_ms[1] > config.stimulus_ms + config.after_ms or \
            config.late_window_ms[1] > config.stimulus_ms:
        raise PsyError("config/psy.yaml: окна разгона не помещаются в протокол")
    return config


# ---------- данные ----------

@dataclass
class KeyNeuron:
    name: str
    side: str
    index: int
    root_id: str

    @property
    def label(self) -> str:
        return f"{self.name} {'Л' if self.side == 'left' else 'П'}"


@dataclass
class Setup:
    config: PsyConfig
    neurons: pd.DataFrame
    csr: dict
    groups: dict
    sets: SerotoninSets
    synapses: dict[str, np.ndarray]  # s_i по каждому критерию
    shares: np.ndarray  # (N, 5) доли модальностей входа
    dominant: np.ndarray  # ведущая модальность, −1 — нет
    keys: list[KeyNeuron]
    sha: dict[str, str]
    variant_csrs: dict = field(default_factory=dict)  # связи вариантов P1b (flybrain.psymech), по запросу

    @property
    def n(self) -> int:
        return len(self.neurons)

    @property
    def kc_mask(self) -> np.ndarray:
        """Клетки Кеньона (cell_class = Kenyon_Cell): у них в режиме «KC тормозятся» эффект дозы обратный."""
        return (self.neurons["cell_class"] == KC_CLASS).to_numpy()

    def shift(self, spec: "RunSpec") -> np.ndarray:
        """Снижение порога каждого нейрона, мВ, для дозы, критерия и режима расчёта (D-055, D-060)."""
        inverted = self.kc_mask if spec.mode == "kc_inhibit" else None
        return threshold_shift(self.synapses[spec.criterion], spec.dose, self.config.delta_v_mv,
                               self.config.s_half, inverted)

    def csr_of(self, variant: str) -> dict:
        if variant == "base":
            return self.csr
        if variant not in self.variant_csrs:
            from flybrain.psymech import variant_csr

            self.variant_csrs[variant] = variant_csr(self, variant)
        return self.variant_csrs[variant]

    def stimulated(self, group: str | None) -> np.ndarray:
        if group is None:
            return np.zeros(0, dtype=np.int64)
        return np.asarray(self.groups["stimuli"][group]["indices"], dtype=np.int64)

    def key_index(self, root_id: str) -> int:
        for k, key in enumerate(self.keys):
            if key.root_id == root_id:
                return k
        raise PsyError(f"нейрон {root_id} не среди ключевых ({', '.join(KEY_READOUTS)})")


def load_setup(config: PsyConfig | None = None, log=print) -> Setup:
    config = config or load_config()
    info_path = PROCESSED_DIR / processed.BUILD_INFO_FILE
    if not info_path.exists():
        raise PsyError("нет подготовленных данных. Сначала выполните:\n  uv run flybrain fetch-data\n"
                       "  uv run flybrain build-data")
    with open(info_path, encoding="utf-8") as handle:
        info = json.load(handle)
    neurons = processed.load_neurons()
    csr = processed.load_connectivity()
    groups = processed.load_groups()
    for c in config.conditions:
        if c.group not in groups["stimuli"]:
            raise PsyError(f"config/psy.yaml: нет группы-стимула «{c.group}» в groups.json")
    try:
        annotations = read_annotations(expected_sha256=info["inputs_sha256"][ANNOTATIONS_NAME])
        sets = serotonin_sets(neurons, annotations)
    except SerotoninError as exc:
        raise PsyError(str(exc)) from exc
    synapses = {name: serotonin_synapses(csr["indptr"], csr["indices"], csr["weight"], mask)
                for name, mask in sets.masks.items()}
    sha = {"connectivity": info["inputs_sha256"]["Connectivity_783.parquet"],
           "annotations": info["inputs_sha256"][ANNOTATIONS_NAME]}
    shares, dominant = _modality(neurons, csr, sha["connectivity"], log)
    keys = []
    for name in KEY_READOUTS:
        group = groups["readouts"][name]
        pairs = sorted(zip(group["indices"], group["root_ids"]), key=lambda p: str(neurons.at[p[0], "side"]))
        keys += [KeyNeuron(name, str(neurons.at[i, "side"]), int(i), str(r)) for i, r in pairs]
    setup = Setup(config, neurons, csr, groups, sets, synapses, shares, dominant, keys, sha)
    for c in config.conditions:
        setup.key_index(c.readout_root_id)  # «исполнитель» обязан быть среди ключевых нейронов
    return setup


def _modality(neurons: pd.DataFrame, csr: dict, connectivity_sha: str, log) -> tuple[np.ndarray, np.ndarray]:
    meta = json.dumps({"format": RESULT_FORMAT, "connectivity": connectivity_sha, "modalities": MODALITIES})
    if MODALITY_CACHE.exists():
        with np.load(MODALITY_CACHE) as arrays:
            if str(arrays["meta"]) == meta:
                return arrays["shares"], arrays["dominant"]
    log("Считаю, от каких органов чувств приходит вход каждого нейрона (один раз, около минуты)…")
    shares, dominant = input_shares(neurons, csr["indptr"], csr["indices"], csr["weight"])
    PSY_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODALITY_CACHE, shares=shares.astype(np.float32), dominant=dominant.astype(np.int8),
                        meta=np.array(meta))
    return shares.astype(np.float32), dominant.astype(np.int8)


# ---------- один расчёт ----------

@dataclass(frozen=True)
class RunSpec:
    """Один расчёт: стимул (или без стимула), доза, критерий серотониновых нейронов, шум, повторы, протокол."""
    part: str
    condition: str | None  # ключ условия из config/psy.yaml; None — без стимула
    hz: float
    dose: float
    criterion: str
    sigma: float
    trials: int
    warmup_ms: float
    stimulus_ms: float
    after_ms: float
    mode: str = "p1"  # P1b: "kc_inhibit" — у клеток Кеньона порог от дозы повышается (D-060)
    variant: str = "base"  # P1b: изменённые связи модели для поиска механизма (flybrain.psymech, D-058)
    quiet_after: bool = False  # P1b: после окна выключается шум (а не стимул)

    @property
    def key(self) -> str:
        what = f"{self.condition}@{self.hz:g}hz" if self.condition else "nostim"
        key = f"{self.part}__{self.criterion}__{what}__d{self.dose:g}__s{self.sigma:g}__t{self.trials}"
        # Добавки P1b — только когда они не по умолчанию: ключи (и зёрна) расчётов P1 не меняются.
        if self.mode != "p1":
            key += f"__m{self.mode}"
        if self.variant != "base":
            key += f"__v{self.variant}"
        if self.quiet_after:
            key += "__quiet"
        return key

    @property
    def seed(self) -> int:
        return zlib.crc32(f"psy:{self.key}".encode("utf-8"))

    def file(self) -> Path:
        return PSY_DIR / f"{self.key.replace('@', '_')}.npz"


@dataclass
class RunResult:
    totals: np.ndarray  # (N,) спайки за время стимула, сумма по повторам
    after: np.ndarray  # (N,) спайки после выключения стимула, сумма по повторам
    key_counts: np.ndarray  # (повторы, ключевые нейроны) за время стимула
    edges_ms: np.ndarray  # (E,) моменты записи суммарной активности, от начала окна стимула
    cumulative: np.ndarray  # (E, повторы) спайки не стимулируемых нейронов с начала окна стимула
    seconds: float


def plan(config: PsyConfig, parts: list[str]) -> list[RunSpec]:
    specs = []
    by_key = {c.key: c for c in config.conditions}
    main = lambda part, crit, c, hz, dose, trials, after: RunSpec(  # noqa: E731
        part, c.key, hz, dose, crit, 0.0, trials, 0.0, config.stimulus_ms, after)
    if "runaway" in parts:
        c = by_key[config.runaway_condition]
        for crit in (config.criterion, config.sensitivity_criterion):
            specs += [main("runaway", crit, c, c.hz, d, config.runaway_trials, config.after_ms)
                      for d in config.runaway_doses]
    if "main" in parts:
        specs += [main("main", config.criterion, c, c.hz, d, config.trials, config.after_ms)
                  for d in config.main_doses for c in config.conditions]
    if "thresholds" in parts:
        specs += [main("thresholds", config.criterion, c, f, d, config.threshold_trials, 0.0)
                  for d in config.doses for c in config.conditions for f in c.threshold_hz]
    if "spont" in parts:
        specs += [RunSpec("spont", None, 0.0, d, config.criterion, s, config.spont_trials, config.warmup_ms,
                          config.stimulus_ms, 0.0) for s in config.sigmas_mv for d in config.main_doses]
    if "broad" in parts:
        specs += [main("main", config.sensitivity_criterion, c, c.hz, d, config.trials, config.after_ms)
                  for d in config.doses if d > 0 for c in config.conditions]
    specs += plan_p1b(config, parts)
    unique = list(dict.fromkeys(specs))
    return unique


def kc_spec(spec: RunSpec) -> RunSpec:
    """Тот же расчёт в режиме «KC тормозятся». При дозе 0 режимы совпадают — берётся расчёт P1 (то же зерно)."""
    from dataclasses import replace

    return spec if spec.dose == 0 else replace(spec, part=f"kc_{spec.part}", mode="kc_inhibit")


def plan_p1b(config: PsyConfig, parts: list[str]) -> list[RunSpec]:
    b = config.p1b
    if b is None:
        return []
    specs = []
    if "mech" in parts:
        specs += [RunSpec("mech", None, 0.0, 0.0, config.criterion, s, b.mech_trials, 0.0, b.mech_noise_ms,
                          b.mech_off_ms, variant=v, quiet_after=True) for v in b.mech_variants for s in b.mech_sigmas_mv]
    by_key = {c.key: c for c in config.conditions}
    if "mech" in parts:
        c = by_key[config.runaway_condition]
        specs += [RunSpec("mech_runaway", c.key, c.hz, d, config.criterion, 0.0, config.runaway_trials, 0.0,
                          config.stimulus_ms, config.after_ms, variant=b.mech_dose_variant) for d in b.mech_dose_doses]
    main = lambda part, c, hz, dose, trials, after: RunSpec(  # noqa: E731
        part, c.key, hz, dose, config.criterion, 0.0, trials, 0.0, config.stimulus_ms, after)
    if "kc_runaway" in parts:
        c = by_key[config.runaway_condition]
        specs += [kc_spec(main("runaway", c, c.hz, d, config.runaway_trials, config.after_ms))
                  for d in b.kc_runaway_doses]
    if "kc_main" in parts:
        specs += [kc_spec(main("main", c, c.hz, d, config.trials, config.after_ms))
                  for d in b.kc_main_doses for c in config.conditions]
    if "kc_thresholds" in parts:
        specs += [kc_spec(main("thresholds", c, f, d, config.threshold_trials, 0.0))
                  for d in b.kc_doses for c in config.conditions for f in c.threshold_hz]
    if "kc_spont" in parts:
        specs += [kc_spec(RunSpec("spont", None, 0.0, d, config.criterion, b.kc_spont_sigma_mv, config.spont_trials,
                                  config.warmup_ms, config.stimulus_ms, 0.0)) for d in b.kc_main_doses]
    return specs


def run_meta(setup: Setup, spec: RunSpec) -> str:
    c = setup.config
    stim = setup.stimulated(setup.config.condition(spec.condition).group if spec.condition else None)
    meta = {
        "format": RESULT_FORMAT, "spec": spec.key, "seed": spec.seed, "engine": engine_version(),
        "delta_v_mv": c.delta_v_mv, "s_half": c.s_half, "bin_ms": c.bin_ms, "sha": setup.sha,
        "serotonin": [int(i) for i in setup.sets.indices(spec.criterion)], "stimulated": [int(i) for i in stim],
        "key_neurons": [k.index for k in setup.keys],
    }
    if spec.mode != "p1":  # добавки P1b — только не по умолчанию, чтобы расчёты P1 оставались годными
        meta["kc_count"] = int(setup.kc_mask.sum())
    if spec.variant != "base":
        from flybrain.psymech import variant_description

        meta["variant"] = variant_description(spec.variant)
    return json.dumps(meta, sort_keys=True)


def run_engine(setup: Setup, spec: RunSpec, device: str) -> RunResult:
    import torch

    from flybrain.engine import Engine

    started = time.monotonic()
    csr = setup.csr_of(spec.variant)
    engine = Engine(csr["indptr"], csr["indices"], csr["weight"], n_trials=spec.trials, device=device, seed=spec.seed)
    group = setup.config.condition(spec.condition).group if spec.condition else None
    stim = setup.stimulated(group)
    engine.set_threshold_shift(setup.shift(spec))
    engine.set_noise(spec.sigma)
    dt = engine.params.dt
    to_steps = lambda ms: int(round(ms / dt))  # noqa: E731
    warmup, window, after = to_steps(spec.warmup_ms), to_steps(spec.stimulus_ms), to_steps(spec.after_ms)
    chunk = to_steps(setup.config.bin_ms)
    stim_t = torch.as_tensor(stim, device=engine.device)
    keys_t = torch.as_tensor([k.index for k in setup.keys], device=engine.device)

    def others() -> np.ndarray:  # спайки не стимулируемых нейронов с начала расчёта, по повторам
        total = engine.counts.sum(0)
        if len(stim):
            total = total - engine.counts[stim_t].sum(0)
        return total.cpu().numpy().astype(np.int64)

    def advance(until: int, record: list) -> None:
        while engine.step < until:
            edge = min(until, (engine.step // chunk + 1) * chunk)
            engine.run_steps(edge - engine.step)
            record.append((engine.step, others()))

    marks: list = []
    advance(warmup, [])  # шум набирается; эти шаги не считаются
    start_counts = engine.counts.cpu().numpy().astype(np.int64)
    start_others = others()
    marks.append((engine.step, start_others))
    if len(stim):
        engine.set_stimulus("excite", stim, spec.hz)
    advance(warmup + window, marks)
    window_counts = engine.counts.cpu().numpy().astype(np.int64)
    if len(stim):
        engine.remove_stimulus("excite")
    if spec.quiet_after:
        engine.set_noise(0.0)
    advance(warmup + window + after, marks)
    end_counts = engine.counts.cpu().numpy().astype(np.int64)
    on_cuda = engine.device.type == "cuda"
    del engine
    if on_cuda:
        torch.cuda.empty_cache()
    steps = np.array([s for s, _ in marks], dtype=np.int64)
    cumulative = np.stack([v for _, v in marks]) - start_others[None, :]
    key_index = keys_t.cpu().numpy()
    return RunResult(
        totals=(window_counts - start_counts).sum(1), after=(end_counts - window_counts).sum(1),
        key_counts=(window_counts - start_counts)[key_index].T.copy(),
        edges_ms=(steps - warmup) * dt, cumulative=cumulative, seconds=time.monotonic() - started)


def load_saved(path: Path, meta: str) -> RunResult | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as a:
            if str(a["meta"]) != meta:
                return None
            return RunResult(a["totals"].astype(np.int64), a["after"].astype(np.int64),
                             a["key_counts"].astype(np.int64), a["edges_ms"].astype(np.float64),
                             a["cumulative"].astype(np.int64), float(a["seconds"]))
    except (OSError, ValueError, KeyError):
        return None


def save_run(path: Path, result: RunResult, meta: str, device: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as handle:
        np.savez_compressed(handle, totals=result.totals.astype(np.int32), after=result.after.astype(np.int32),
                            key_counts=result.key_counts.astype(np.int32), edges_ms=result.edges_ms,
                            cumulative=result.cumulative.astype(np.int64), seconds=np.array(result.seconds),
                            meta=np.array(meta), device=np.array(device))
    part.replace(path)


def saved_runs(setup: Setup, specs: list[RunSpec]) -> dict[RunSpec, RunResult]:
    """Уже посчитанные расчёты из `data/psy/` (с теми же настройками); чего нет — пропускается."""
    runs = {}
    for spec in specs:
        result = load_saved(spec.file(), run_meta(setup, spec))
        if result is not None:
            runs[spec] = result
    return runs


def collect(setup: Setup, parts: list[str], device: str = "auto", rerun: bool = False,
            log=print) -> dict[RunSpec, RunResult]:
    specs = plan(setup.config, parts)
    results: dict[RunSpec, RunResult] = {}
    todo = []
    for spec in specs:
        meta = run_meta(setup, spec)
        saved = None if rerun else load_saved(spec.file(), meta)
        if saved is None:
            todo.append((spec, meta))
        else:
            results[spec] = saved
    if todo:
        log(f"Расчётов: {len(specs)}, готовы {len(specs) - len(todo)}, считаю {len(todo)}.")
    for number, (spec, meta) in enumerate(todo, start=1):
        log(f"  [{number}/{len(todo)}] {spec.key} …", end="", flush=True)
        result = run_engine(setup, spec, device)
        save_run(spec.file(), result, meta, device)
        results[spec] = result
        log(f" {result.seconds:.0f} с")
    return {spec: results[spec] for spec in specs}


# ---------- метрики ----------

@dataclass
class Metrics:
    """Что считаем по одному расчёту (правила — D-056)."""
    responders_b2: int  # > 0 Гц за все повторы, вместе со стимулируемыми (правило Б2)
    responders: int  # то же без стимулируемых
    active: int  # ≥ 1 Гц, без стимулируемых (D-018)
    mean_rate: float  # средняя частота по всем не стимулируемым нейронам, Гц
    n_eff: float  # эффективное число нейронов exp(H) по спайкам не стимулируемых
    super_classes: dict[str, int]  # откликнувшиеся (> 0 Гц, без стимулируемых) по суперклассам
    modalities: dict[str, int]  # те же — по ведущей модальности входа
    modality_entropy: float  # энтропия распределения по ведущим модальностям, бит
    foreign_share: float | None  # доля откликнувшихся, у которых ведущая модальность не та, что у стимула
    visual_system: int  # откликнувшиеся в зрительной системе (optic, visual_projection, visual_centrifugal)
    spike_shares: np.ndarray  # средние доли модальностей входа, взвешенные по спайкам
    key_rates: list[tuple[float, float]]  # (среднее, погрешность), Гц, по ключевым нейронам
    late_rate: float  # спайков не стимулируемых в окне конца стимула, на повтор в секунду
    after_rate: float  # то же в окне после выключения стимула
    after_ratio: float | None  # after_rate / late_rate
    after_neurons: int  # нейронов, выстреливших после выключения стимула (за все повторы)
    runaway: bool | None  # разгон по правилу D-056 (None — после стимула не смотрели)
    sustained_trials: int | None  # в скольких повторах по отдельности выполнено то же правило
    after_classes: dict[str, int]  # кто стреляет в окне после стимула: cell_class (или суперкласс), нейронов


def _mean_se(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def _window(result: RunResult, start_ms: float, end_ms: float) -> tuple[np.ndarray, float] | None:
    """Спайки не стимулируемых нейронов в окне [start, end] (по записанным моментам) по повторам и длина окна, с."""
    edges = result.edges_ms
    lo = np.flatnonzero(np.isclose(edges, start_ms) | (edges >= start_ms))
    hi = np.flatnonzero(np.isclose(edges, end_ms) | (edges <= end_ms))
    if not len(lo) or not len(hi) or edges[hi[-1]] <= edges[lo[0]]:
        return None
    a, b = lo[0], hi[-1]
    return (result.cumulative[b] - result.cumulative[a]).astype(np.float64), (edges[b] - edges[a]) / 1000


def metrics(setup: Setup, spec: RunSpec, result: RunResult) -> Metrics:
    config = setup.config
    condition = config.condition(spec.condition) if spec.condition else None
    stim = setup.stimulated(condition.group if condition else None)
    others = np.ones(setup.n, dtype=bool)
    others[stim] = False
    seconds = spec.trials * spec.stimulus_ms / 1000
    rates = result.totals / seconds
    responders = (result.totals > 0) & others
    spikes = np.where(others, result.totals, 0).astype(np.float64)
    total = spikes.sum()
    p = spikes[spikes > 0] / total if total > 0 else np.zeros(0)
    n_eff = float(math.exp(-(p * np.log(p)).sum())) if total > 0 else 0.0
    super_class = setup.neurons["super_class"].fillna("без аннотации").to_numpy()
    sc_counts = pd.Series(super_class[responders]).value_counts().to_dict()
    dom = setup.dominant[responders]
    names = np.array([*MODALITIES, "none"])
    dom_names = names[np.where(dom >= 0, dom, len(MODALITIES))]
    mod_counts = {m: int((dom_names == m).sum()) for m in names}
    shares = np.array([mod_counts[m] for m in MODALITIES], dtype=np.float64)
    q = shares[shares > 0] / shares.sum() if shares.sum() > 0 else np.zeros(0)
    entropy = max(0.0, float(-(q * np.log2(q)).sum()))  # max — чтобы пустой набор давал 0, а не −0
    foreign = None
    if condition is not None and responders.sum() > 0:
        foreign = float((dom_names != condition.modality).mean())
    visual = int(np.isin(super_class[responders], VISUAL_SUPER_CLASSES).sum())
    spike_shares = (setup.shares * spikes[:, None]).sum(0) / total if total > 0 else np.zeros(len(MODALITIES))
    key_rates = [_mean_se(result.key_counts[:, k] / (spec.stimulus_ms / 1000)) for k in range(len(setup.keys))]
    late = _window(result, *config.late_window_ms) if spec.after_ms > 0 else None
    after = _window(result, *config.after_window_ms) if spec.after_ms > 0 else None
    late_rate = late[0].sum() / spec.trials / late[1] if late else float("nan")
    after_rate = after[0].sum() / spec.trials / after[1] if after else float("nan")
    ratio = runaway = sustained = None
    if late and after:
        ratio = after_rate / late_rate if late_rate > 0 else (math.inf if after_rate > 0 else 0.0)
        runaway = bool(ratio >= config.min_ratio)
        per_late, per_after = late[0] / late[1], after[0] / after[1]
        sustained = int(((per_after > 0) & (per_after >= config.min_ratio * per_late)).sum())
    after_mask = (result.after > 0) & others
    labels = setup.neurons["cell_class"].fillna(setup.neurons["super_class"]).fillna("без аннотации").to_numpy()
    after_classes = {str(k): int(v) for k, v in pd.Series(labels[after_mask]).value_counts().items()}
    return Metrics(
        responders_b2=int((result.totals > 0).sum()), responders=int(responders.sum()),
        active=int(((rates >= ACTIVE_HZ) & others).sum()), mean_rate=float(rates[others].mean()), n_eff=n_eff,
        super_classes={str(k): int(v) for k, v in sc_counts.items()}, modalities=mod_counts,
        modality_entropy=entropy, foreign_share=foreign, visual_system=visual, spike_shares=spike_shares,
        key_rates=key_rates, late_rate=late_rate, after_rate=after_rate, after_ratio=ratio,
        after_neurons=int(after_mask.sum()), runaway=runaway, sustained_trials=sustained, after_classes=after_classes)


def readout_threshold(points: list[tuple[float, float]], level: float = ACTIVE_HZ) -> tuple[str, float | None]:
    """Наименьшая частота стимула, при которой «исполнитель» активен: линейно между соседними точками сетки.

    Возвращает вид ответа («между», «не выше», «выше») и число (частоту) или None.
    """
    points = sorted(points)
    if not points:
        return "нет данных", None
    if points[0][1] >= level:
        return "не выше", points[0][0]
    for (f0, r0), (f1, r1) in zip(points, points[1:]):
        if r0 < level <= r1:
            return "между", f0 + (level - r0) * (f1 - f0) / (r1 - r0)
    return "выше", points[-1][0]


@dataclass
class Results:
    setup: Setup
    runs: dict[RunSpec, RunResult]
    metrics: dict[RunSpec, Metrics] = field(default_factory=dict)

    def find(self, **want) -> list[RunSpec]:
        return [s for s in self.runs if all(getattr(s, k) == v for k, v in want.items())]

    def one(self, **want) -> tuple[RunSpec, Metrics]:
        found = self.find(**want)
        if len(found) != 1:
            raise PsyError(f"ожидался один расчёт {want}, найдено {len(found)}")
        return found[0], self.metrics[found[0]]


def analyse(setup: Setup, runs: dict[RunSpec, RunResult]) -> Results:
    results = Results(setup, runs)
    for spec, result in runs.items():
        results.metrics[spec] = metrics(setup, spec, result)
    return results


def runaway_dose(results: Results, criterion: str) -> float | None:
    """Наименьшая доза поиска, при которой разгон по правилу D-056; None — разгона в сетке нет."""
    specs = sorted(results.find(part="runaway", criterion=criterion), key=lambda s: s.dose)
    return next((s.dose for s in specs if results.metrics[s].runaway), None)


# ---------- команда ----------

def run_psy(device: str = "auto", rerun: bool = False, parts: list[str] | None = None) -> int:
    from flybrain.psyreport import write_report

    print("FlyBrain: эксперимент P1 «психоделик» (вычислительная гипотеза, не модель ощущений мухи)\n")
    try:
        config = load_config()
        parts = parts or list(PARTS)
        unknown = [part for part in parts if part not in PARTS]
        if unknown:
            raise PsyError(f"нет такой части опыта: {', '.join(unknown)}. Есть: {', '.join(PARTS)}")
        if (config.runaway_dose is None or not config.doses) and set(parts) - {"runaway"}:
            print("Дозы основного опыта ещё не заданы (config/psy.yaml: doses, runaway_dose): считаю только поиск "
                  "дозы разгона.\n")
            parts = ["runaway"]
        setup = load_setup(config)
        started = time.monotonic()
        collect(setup, [p for p in parts if p != "hyst"], device=device, rerun=rerun)
        hysteresis = None
        if config.p1b is not None:
            from flybrain.psymech import run_hysteresis

            hysteresis = run_hysteresis(setup, device=device, rerun=rerun and "hyst" in parts,
                                        compute="hyst" in parts)
        # Отчёты — по всем готовым расчётам, а не только по посчитанным сейчас: `--only` не обрезает отчёт.
        runs = saved_runs(setup, plan(config, list(PARTS)))
        p1_specs = set(plan(config, list(P1_PARTS)))
        results = analyse(setup, {s: r for s, r in runs.items() if s in p1_specs})
        written = write_report(results)
        if config.p1b is not None:
            from flybrain.p1breport import write_p1b_report

            written += write_p1b_report(analyse(setup, runs), hysteresis)
    except (PsyError, OSError) as exc:
        print(f"[ОШИБКА] {exc}", file=sys.stderr)
        return 1
    for crit in (config.criterion, config.sensitivity_criterion):
        if results.find(part="runaway", criterion=crit):
            dose = runaway_dose(results, crit)
            print(f"Доза разгона ({crit}): {'не найдена в сетке поиска' if dose is None else f'{dose:g}'}")
    print(f"\nГотово за {time.monotonic() - started:.0f} с. Отчёт: "
          + ", ".join(path.relative_to(PROJECT_ROOT).as_posix() for path in written))
    return 0
