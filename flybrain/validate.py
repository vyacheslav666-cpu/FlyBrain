"""`flybrain validate`: движок против эталона Brian2 и выводы статьи → docs/VALIDATION.md (C6, D-031).

1. Берёт условия эталона (`config/reference.yaml`) и готовые повторы Brian2 из `data/reference/` (C4).
2. Считает каждое условие на движке дважды: основной расчёт и «шумовой предел» — тот же расчёт
   с другим зерном случайных чисел. Результаты сохраняются в `data/validation/` и при следующем запуске
   берутся оттуда, если не изменились код движка, связи и условия.
3. Проверяет пункты А (движок против эталона, пороги D-018), Б1–Б7 (выводы статьи; одни и те же правила
   для эталона и для движка), В (тень, без порога), Д1 и Д2 (добавлены после эталона). Правила — D-031, D-032.
4. Пишет `docs/VALIDATION.md` и картинки в `docs/validation/` (модуль `flybrain.report`).
"""

import hashlib
import json
import math
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain import processed
from flybrain.fetch import sha256_of
from flybrain.groups import load_group_config
from flybrain.paths import PROCESSED_DIR, PROJECT_ROOT
from flybrain.reference import ACTIVE_HZ, INFO_FILE, REFERENCE_DIR, SHUFFLED_FILE, Condition, \
    load_reference_config, load_trial

VALIDATION_DIR = PROJECT_ROOT / "data" / "validation"
ENGINE_DIR = PROJECT_ROOT / "flybrain" / "engine"
# Меняется, когда меняется формат сохранённых результатов движка; старые тогда пересчитываются.
RESULT_FORMAT = 1
RUNS = ("main", "repeat")  # основной расчёт движка и «шумовой предел» (другое зерно)

# А — пороги D-018, не меняются.
CORR_MIN = 0.95
OVERLAP_MIN = 0.80
KEY_ABS_HZ = 3.0
KEY_REL = 0.15
# Б и Д — правила D-031.
BOUNDARY_SE = 2.0  # «на границе», если до порога меньше двух погрешностей среднего
B3_MAX_SHARE = 0.5  # Б3: с горечью MN9 не больше половины ответа на сахар
D1_MIN_RATIO = 2.0  # Д1: вместе MN9 как минимум вдвое сильнее, чем от каждого стимула по отдельности
B2_CONDITIONS = {10: "sugar_10hz", 50: "sugar_50hz", 100: "sugar_100hz", 200: "sugar_200hz"}
PAPER_B2 = {10: 45, 200: 455}  # Shiu et al., версия 630 (раздел Results)
PAPER_READOUTS = ("MN9", "aDN1", "aDN2", "aBN1")  # у них в figures.ipynb записан свой номер нейрона

PASS, FAIL, BOUNDARY, INFO = "pass", "fail", "boundary", "info"


class ValidateError(Exception):
    """Проверку нельзя запустить: нет эталона, данных или они не сходятся."""


# ---------- ключевые нейроны и расчёты ----------

@dataclass(frozen=True)
class KeyNeuron:
    readout: str  # MN9, aDN1, …
    side: str  # сторона мухи по таблице аннотаций (D-011)
    index: int
    root_id: str
    paper: bool  # этот номер записан в figures.ipynb статьи

    @property
    def label(self) -> str:
        return f"{self.readout} {'Л' if self.side == 'left' else 'П'}"


@dataclass
class Run:
    """Один расчёт условия: спайки каждого нейрона за все повторы и спайки ключевых нейронов по повторам."""
    totals: np.ndarray  # (нейроны,)
    key_counts: np.ndarray  # (повторы, ключевые нейроны)
    duration_s: float
    seconds: float = float("nan")  # сколько длился расчёт

    @property
    def n_trials(self) -> int:
        return int(self.key_counts.shape[0])

    @property
    def rates(self) -> np.ndarray:
        """Средняя частота каждого нейрона за все повторы, Гц."""
        return self.totals / (self.n_trials * self.duration_s)

    def trial_rates(self, column: int) -> np.ndarray:
        return self.key_counts[:, column] / self.duration_s


def mean_se(values: np.ndarray) -> tuple[float, float]:
    """Среднее и его погрешность: стандартное отклонение по повторам / √(число повторов)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1) / math.sqrt(len(values)))


def load_key_neurons(groups: dict, neurons: pd.DataFrame, readouts: list[str]) -> list[KeyNeuron]:
    """Ключевые нейроны по обе стороны; у MN9, aDN1, aDN2, aBN1 отмечен номер из статьи."""
    group_config = load_group_config()
    keys = []
    for readout in readouts:
        group = groups["readouts"][readout]
        entries = group_config["readouts"][readout].get("paper_ids") or []
        paper_ids = {str(entry["id"]) for entry in entries if entry["status"] == "in_783"}
        pairs = sorted(zip(group["indices"], group["root_ids"]), key=lambda pair: str(neurons.at[pair[0], "side"]))
        for index, root_id in pairs:
            keys.append(KeyNeuron(readout, str(neurons.at[index, "side"]), int(index), str(root_id),
                                  str(root_id) in paper_ids))
    for readout in PAPER_READOUTS:
        if sum(key.paper for key in keys if key.readout == readout) != 1:
            raise ValidateError(f"у «{readout}» должен быть ровно один номер из статьи (config/neuron_groups.yaml)")
    return keys


def stimuli_of(condition: Condition, groups: dict) -> dict:
    """Стимулы условия так, как их записывает эталон C4 в каждом повторе (`meta`)."""
    exc2 = groups["stimuli"][condition.excite2_group]["indices"] if condition.excite2_group else []
    return {"exc": list(groups["stimuli"][condition.excite_group]["indices"]), "exc2": list(exc2),
            "r_poi_hz": condition.excite_hz, "r_poi2_hz": condition.excite2_hz}


def stimulated_of(condition: Condition, groups: dict) -> np.ndarray:
    """Все нейроны, на которых стоит стимул (в том числе с частотой 0 Гц): в А1, А2 и Б2 не считаются."""
    stimuli = stimuli_of(condition, groups)
    return np.array(sorted(set(stimuli["exc"]) | set(stimuli["exc2"])), dtype=np.int64)


def load_reference_run(condition: Condition, groups: dict, n_trials: int, duration_ms: float,
                       keys: list[KeyNeuron], n_neurons: int, connectivity_sha: str) -> Run:
    """Повторы Brian2 одного условия (C4); каждый сверяется с нынешними настройками."""
    trial_dir = REFERENCE_DIR / "trials" / condition.key
    files = sorted(trial_dir.glob("trial_*.npz"))[:n_trials]
    if len(files) < n_trials:
        raise ValidateError(f"эталон «{condition.name_ru}»: готово повторов {len(files)} из {n_trials}. "
                            "Выполните: uv run flybrain reference")
    want = stimuli_of(condition, groups) | {"duration_ms": duration_ms, "connectivity_sha256": connectivity_sha}
    totals = np.zeros(n_neurons, dtype=np.int64)
    key_counts = np.zeros((n_trials, len(keys)), dtype=np.int64)
    columns = [key.index for key in keys]
    for trial_number, path in enumerate(files):
        trial = load_trial(path)
        meta = json.loads(str(trial["meta"]))
        if any(meta.get(name) != value for name, value in want.items()):
            raise ValidateError(f"повтор {path.relative_to(PROJECT_ROOT).as_posix()} посчитан с другими настройками. "
                                "Выполните: uv run flybrain reference")
        counts = np.bincount(trial["neuron"], minlength=n_neurons)
        totals += counts
        key_counts[trial_number] = counts[columns]
    return Run(totals, key_counts, duration_ms / 1000)


# ---------- движок ----------

def engine_version() -> str:
    """Контрольная сумма кода движка: изменился код — сохранённые результаты не подходят."""
    digest = hashlib.sha256()
    for path in sorted(ENGINE_DIR.glob("*.py")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:12]


def engine_seed(key: str, run: str) -> int:
    """Зерно движка: своё у каждого условия и расчёта и одно и то же при каждом запуске."""
    return zlib.crc32(f"engine-{run}:{key}".encode("utf-8"))


def csr_summing_duplicates(post: np.ndarray, pre: np.ndarray, weight: np.ndarray, n_neurons: int) -> dict:
    """CSR, где повторяющиеся пары «отправитель → получатель» сложены.

    После перемешивания (D-029) одна пара может встретиться дважды. В Brian2 это две связи, и обе
    в один шаг делают g += w; сумма весов в одной связи даёт ровно то же.
    """
    order = np.lexsort((pre, post))
    post, pre, weight = post[order], pre[order], np.asarray(weight, dtype=np.int64)[order]
    first = np.ones(len(post), dtype=bool)
    first[1:] = (post[1:] != post[:-1]) | (pre[1:] != pre[:-1])
    starts = np.flatnonzero(first)
    indptr = np.zeros(n_neurons + 1, dtype=np.int64)
    np.cumsum(np.bincount(post[starts], minlength=n_neurons), out=indptr[1:])
    return {"indptr": indptr, "indices": pre[starts].astype(np.int32),
            "weight": np.add.reduceat(weight, starts).astype(np.int32)}


def load_shuffled_csr(n_neurons: int) -> dict:
    """Связи контроля Б7 — тот же файл, на котором считал Brian2 (C4)."""
    table = pd.read_parquet(REFERENCE_DIR / SHUFFLED_FILE,
                            columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    return csr_summing_duplicates(table["Postsynaptic_Index"].to_numpy(np.int64),
                                  table["Presynaptic_Index"].to_numpy(np.int64),
                                  table["Excitatory x Connectivity"].to_numpy(np.int64), n_neurons)


def run_engine(csr: dict, stimuli: dict, seed: int, n_trials: int, duration_ms: float, keys: list[KeyNeuron],
               device: str, progress=None) -> Run:
    import torch

    from flybrain.engine import Engine

    started = time.monotonic()
    engine = Engine(csr["indptr"], csr["indices"], csr["weight"], n_trials=n_trials, device=device, seed=seed)
    engine.set_stimulus("excite", stimuli["exc"], stimuli["r_poi_hz"])
    if stimuli["exc2"]:
        engine.set_stimulus("excite2", stimuli["exc2"], stimuli["r_poi2_hz"])
    total = int(round(duration_ms / engine.params.dt))
    chunk = 56 * engine.block_steps  # примерно 100 мс жизни мухи
    done = 0
    while done < total:
        steps = min(chunk, total - done)
        engine.run_steps(steps)
        engine.synchronize()
        done += steps
        if progress:
            progress(done / total)
    counts = engine.counts.cpu().numpy().astype(np.int64)  # (нейроны, повторы)
    on_cuda = engine.device.type == "cuda"
    del engine
    if on_cuda:
        torch.cuda.empty_cache()
    return Run(counts.sum(axis=1), counts[[key.index for key in keys]].T.copy(), duration_ms / 1000,
               time.monotonic() - started)


def result_meta(condition: Condition, run: str, stimuli: dict, n_trials: int, duration_ms: float,
                connectivity_sha: str, keys: list[KeyNeuron], version: str) -> str:
    return json.dumps({
        "format": RESULT_FORMAT, "condition": condition.key, "run": run, "seed": engine_seed(condition.key, run),
        "trials": n_trials, "duration_ms": duration_ms, **stimuli, "connectivity_sha256": connectivity_sha,
        "key_neurons": [key.index for key in keys], "engine": version,
    }, sort_keys=True)


def load_saved(path: Path, meta: str) -> Run | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as arrays:
            if str(arrays["meta"]) != meta:
                return None
            return Run(arrays["totals"].astype(np.int64), arrays["key_counts"].astype(np.int64),
                       float(arrays["duration_s"]), float(arrays["seconds"]))
    except (OSError, ValueError, KeyError):
        return None


def save_run(path: Path, run: Run, meta: str, device: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as handle:
        np.savez_compressed(handle, totals=run.totals.astype(np.int32), key_counts=run.key_counts.astype(np.int32),
                            duration_s=np.array(run.duration_s), seconds=np.array(run.seconds),
                            meta=np.array(meta), device=np.array(device))
    part.replace(path)


# ---------- пункт А ----------

@dataclass
class KeyDiff:
    key: KeyNeuron
    base: float
    test: float
    allowed: float

    @property
    def ok(self) -> bool:
        return abs(self.test - self.base) <= self.allowed


@dataclass
class Agreement:
    """Пункт А для одного условия: насколько расчёт `test` совпадает с расчётом `base`."""
    corr: float  # nan — нельзя посчитать (меньше трёх нейронов или все частоты одинаковы)
    overlap: float
    n_base: int
    n_test: int
    n_common: int
    n_union: int
    keys: list[KeyDiff]

    @property
    def corr_ok(self) -> bool:
        return not math.isnan(self.corr) and self.corr >= CORR_MIN

    @property
    def overlap_ok(self) -> bool:
        return self.overlap >= OVERLAP_MIN

    @property
    def keys_ok(self) -> bool:
        return all(diff.ok for diff in self.keys)

    @property
    def ok(self) -> bool:
        return self.corr_ok and self.overlap_ok and self.keys_ok

    def worst_key(self) -> KeyDiff | None:
        """Ключевой нейрон, ближе всех подошедший к допуску (среди тех, кто хоть где-то стреляет)."""
        firing = [diff for diff in self.keys if diff.base > 0 or diff.test > 0]
        return max(firing, key=lambda diff: abs(diff.test - diff.base) / diff.allowed) if firing else None


def agreement(base: Run, test: Run, stimulated: np.ndarray, keys: list[KeyNeuron]) -> Agreement:
    rate_base, rate_test = base.rates, test.rates
    counted = np.ones(len(rate_base), dtype=bool)
    counted[stimulated] = False
    active_base = (rate_base >= ACTIVE_HZ) & counted
    active_test = (rate_test >= ACTIVE_HZ) & counted
    union = active_base | active_test
    n_union = int(union.sum())
    corr = float("nan")
    if n_union >= 3 and rate_base[union].std() > 0 and rate_test[union].std() > 0:
        corr = float(np.corrcoef(rate_base[union], rate_test[union])[0, 1])
    common = int((active_base & active_test).sum())
    diffs = [KeyDiff(key, float(rate_base[key.index]), float(rate_test[key.index]),
                     max(KEY_ABS_HZ, KEY_REL * float(rate_base[key.index]))) for key in keys]
    return Agreement(corr, common / n_union if n_union else 1.0, int(active_base.sum()), int(active_test.sum()),
                     common, n_union, diffs)


# ---------- пункты Б, В, Д ----------

@dataclass
class Check:
    """Одна проверка внутри пункта: «запас» до порога (больше нуля — выполнено) и его погрешность."""
    text: str
    margin: float
    se: float = 0.0
    strict: bool = False  # нужно строго больше нуля (например, «не активен» — это < 1 Гц)

    @property
    def state(self) -> str:
        if self.se > 0 and abs(self.margin) < BOUNDARY_SE * self.se:
            return BOUNDARY
        return PASS if self.margin > 0 or (self.margin == 0 and not self.strict) else FAIL


@dataclass
class Verdict:
    tests: list[Check]
    numbers: list[str]  # сырые числа для отчёта
    values: dict = field(default_factory=dict)  # числа для картинок: подпись → (среднее, погрешность)

    @property
    def state(self) -> str:
        states = {test.state for test in self.tests}
        if not states:
            return INFO
        return FAIL if FAIL in states else BOUNDARY if BOUNDARY in states else PASS


def number(value: float, digits: int = 1) -> str:
    """Число по-русски: десятичная запятая, тысячи через пробел."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def hz(mean: float, se: float) -> str:
    digits = 2 if 0 < abs(mean) < 1 else 1  # иначе редкие спайки выглядели бы как «0,0 ± 0,0»
    return f"{number(mean, digits)} ± {number(se, digits)} Гц" if se > 0 else f"{number(mean, digits)} Гц"


class Facts:
    """Числа одного набора расчётов (эталон или движок), из которых решаются пункты Б, В, Д."""

    def __init__(self, runs: dict[str, Run], keys: list[KeyNeuron], stimulated: dict[str, np.ndarray]):
        self.runs, self.keys, self.stimulated = runs, keys, stimulated

    def key(self, readout: str, side: str | None = None) -> KeyNeuron:
        """Нейрон по стороне; без стороны — тот, чей номер записан в статье."""
        for key in self.keys:
            if key.readout == readout and (key.side == side if side else key.paper):
                return key
        raise ValidateError(f"нет ключевого нейрона {readout} {side or '(из статьи)'}")

    def column(self, key: KeyNeuron) -> int:
        return self.keys.index(key)

    def rate(self, condition: str, key: KeyNeuron) -> tuple[float, float]:
        return mean_se(self.runs[condition].trial_rates(self.column(key)))

    def trial_rates(self, condition: str, key: KeyNeuron) -> np.ndarray:
        return self.runs[condition].trial_rates(self.column(key))

    def spikes(self, condition: str, key: KeyNeuron) -> int:
        return int(self.runs[condition].key_counts[:, self.column(key)].sum())

    def responders(self, condition: str, above_zero: bool = True, with_stimulated: bool = False) -> int:
        """Сколько нейронов откликнулось: частота > 0 Гц (правило статьи) или ≥ 1 Гц (D-018)."""
        rates = self.runs[condition].rates
        responded = rates > 0 if above_zero else rates >= ACTIVE_HZ
        if not with_stimulated:
            responded = responded.copy()
            responded[self.stimulated[condition]] = False
        return int(responded.sum())


def side_ru(key: KeyNeuron) -> str:
    return "левый" if key.side == "left" else "правый"


def active_test(text: str, facts: Facts, condition: str, key: KeyNeuron) -> Check:
    mean, se = facts.rate(condition, key)
    return Check(text, mean - ACTIVE_HZ, se)


def silent_test(text: str, facts: Facts, condition: str, key: KeyNeuron) -> Check:
    mean, se = facts.rate(condition, key)
    return Check(text, ACTIVE_HZ - mean, se, strict=True)


def check_b1(facts: Facts) -> Verdict:
    condition = "sugar_100hz"
    left, right = facts.key("MN9", "left"), facts.key("MN9", "right")
    (l_mean, l_se), (r_mean, r_se) = facts.rate(condition, left), facts.rate(condition, right)
    diff, diff_se = mean_se(facts.trial_rates(condition, right) - facts.trial_rates(condition, left))
    return Verdict(
        [active_test("левый MN9 активен (≥ 1 Гц)", facts, condition, left),
         active_test("правый MN9 активен (≥ 1 Гц)", facts, condition, right),
         Check("правый (сторона, противоположная стимулу) сильнее левого", diff, diff_se, strict=True)],
        [f"MN9 левый — {hz(l_mean, l_se)}, правый — {hz(r_mean, r_se)}; разница {hz(diff, diff_se)}"],
        {"MN9 Л": (l_mean, l_se), "MN9 П": (r_mean, r_se)})


def check_b2(facts: Facts) -> Verdict:
    # Основной счёт — вместе со стимулируемыми: так считает get_rate у Shiu (D-032); без них — для справки.
    counts = {f: facts.responders(key, with_stimulated=True) for f, key in B2_CONDITIONS.items()}
    without = {f: facts.responders(key) for f, key in B2_CONDITIONS.items()}
    active = {f: facts.responders(key, above_zero=False) for f, key in B2_CONDITIONS.items()}
    freqs = sorted(counts)
    growth = min(counts[b] - counts[a] for a, b in zip(freqs, freqs[1:]))

    def row(values: dict) -> str:
        return ", ".join(f"{f} Гц — {values[f]}" for f in freqs)

    return Verdict(
        [Check("при 10 Гц — десятки (10–99)", min(counts[10] - 10, 99 - counts[10])),
         Check("при 200 Гц — сотни (100–999)", min(counts[200] - 100, 999 - counts[200])),
         Check("число растёт с каждой частотой", growth, strict=True)],
        [f"откликнулось (> 0 Гц, вместе со стимулируемыми): {row(counts)}",
         f"для справки, без стимулируемых: {row(without)}",
         f"для справки, по правилу D-018 (≥ 1 Гц, без стимулируемых): {row(active)}"],
        {"counts": counts, "without_stimulated": without, "active": active})


def check_b3(facts: Facts) -> Verdict:
    mn9 = facts.key("MN9")
    sugar, sugar_se = facts.rate("sugar_100hz", mn9)
    mixed, mixed_se = facts.rate("sugar_100hz_bitter_200hz", mn9)
    share = f" (это {number(100 * mixed / sugar, 0)} % от ответа на сахар)" if sugar > 0 else ""
    return Verdict(
        [active_test("без горечи MN9 отвечает на сахар (≥ 1 Гц)", facts, "sugar_100hz", mn9),
         Check("с горечью — не больше половины ответа на сахар", B3_MAX_SHARE * sugar - mixed,
              math.hypot(B3_MAX_SHARE * sugar_se, mixed_se))],
        [f"MN9 из статьи ({side_ru(mn9)}): сахар 100 Гц — {hz(sugar, sugar_se)}; "
         f"сахар 100 + горечь 200 Гц — {hz(mixed, mixed_se)}{share}"],
        {"сахар": (sugar, sugar_se), "сахар + горечь": (mixed, mixed_se)})


def check_b4(facts: Facts) -> Verdict:
    condition = "water_100hz"
    left, right = facts.key("MN9", "left"), facts.key("MN9", "right")
    rates = {key: facts.rate(condition, key) for key in (left, right)}
    best = max(rates, key=lambda key: rates[key][0])
    return Verdict(
        [Check("хотя бы один MN9 активен (≥ 1 Гц)", rates[best][0] - ACTIVE_HZ, rates[best][1])],
        [f"MN9 левый — {hz(*rates[left])}, правый — {hz(*rates[right])}"],
        {"MN9 Л": rates[left], "MN9 П": rates[right]})


def check_b5(facts: Facts) -> Verdict:
    condition = "antenna_100hz"
    tests, parts, values, others = [], [], {}, []
    for readout in ("aDN1", "aDN2", "aBN1"):
        key = facts.key(readout)
        mean, se = facts.rate(condition, key)
        tests.append(active_test(f"{readout} из статьи ({side_ru(key)}) активен", facts, condition, key))
        parts.append(f"{readout} ({side_ru(key)}) — {hz(mean, se)}")
        values[readout] = (mean, se)
        other = facts.key(readout, "left" if key.side == "right" else "right")
        others.append(f"{readout} {side_ru(other)} — {hz(*facts.rate(condition, other))}")
    return Verdict(tests, ["нейроны из статьи: " + "; ".join(parts),
                           "для справки, их пары на другой стороне (в статье их нет): " + "; ".join(others)], values)


def check_b6(facts: Facts) -> Verdict:
    abn1 = facts.key("aBN1")
    ce, ce_se = facts.rate("jon_ce_100hz", abn1)
    f, f_se = facts.rate("jon_f_100hz", abn1)
    n = facts.runs["jon_f_100hz"].n_trials
    ratio = f", в {number(ce / f, 0)} раз меньше" if f > 0 else ""
    return Verdict(
        [active_test("от JO-CE aBN1 активен (≥ 1 Гц)", facts, "jon_ce_100hz", abn1),
         silent_test("от JO-F aBN1 не активен (< 1 Гц)", facts, "jon_f_100hz", abn1)],
        [f"aBN1 из статьи ({side_ru(abn1)}): JO-CE — {hz(ce, ce_se)}, JO-F — {hz(f, f_se)}{ratio}",
         f"от JO-F: {facts.spikes('jon_f_100hz', abn1)} спайков за {n} повторов по 1 с "
         f"(порог «активного» 1 Гц — это {n} спайков)"],
        {"JO-CE": (ce, ce_se), "JO-F": (f, f_se)})


def check_b7(facts: Facts) -> Verdict:
    condition = "sugar_100hz_shuffled"
    left, right = facts.key("MN9", "left"), facts.key("MN9", "right")
    return Verdict(
        [silent_test("левый MN9 не активен (< 1 Гц)", facts, condition, left),
         silent_test("правый MN9 не активен (< 1 Гц)", facts, condition, right)],
        [f"MN9 левый — {hz(*facts.rate(condition, left))}, правый — {hz(*facts.rate(condition, right))}"],
        {"MN9 Л": facts.rate(condition, left), "MN9 П": facts.rate(condition, right)})


def check_v(facts: Facts) -> Verdict:
    condition = "shadow_left_100hz"
    left, right = facts.key("GF", "left"), facts.key("GF", "right")
    (l_mean, l_se), (r_mean, r_se) = facts.rate(condition, left), facts.rate(condition, right)
    return Verdict([], [f"гигантское волокно (GF): левое — {hz(l_mean, l_se)}, правое — {hz(r_mean, r_se)}",
                        f"всего откликнулось нейронов (≥ 1 Гц, без стимулируемых): "
                        f"{facts.responders(condition, above_zero=False)}"],
                   {"GF Л": (l_mean, l_se), "GF П": (r_mean, r_se)})


def check_d1(facts: Facts) -> Verdict:
    mn9 = facts.key("MN9")
    parts = {"сахар 40": "sugar_40hz_water_0hz", "вода 40": "sugar_0hz_water_40hz", "вместе": "sugar_40hz_water_40hz"}
    values = {label: facts.rate(condition, mn9) for label, condition in parts.items()}
    single = max(("сахар 40", "вода 40"), key=lambda label: values[label][0])
    both, both_se = values["вместе"]
    other = facts.key("MN9", "left" if mn9.side == "right" else "right")
    return Verdict(
        [active_test("вместе MN9 активен (≥ 1 Гц)", facts, parts["вместе"], mn9),
         Check("вместе как минимум вдвое сильнее, чем от каждого по отдельности",
              both - D1_MIN_RATIO * values[single][0], math.hypot(both_se, D1_MIN_RATIO * values[single][1]))],
        [f"MN9 из статьи ({side_ru(mn9)}): сахар 40 Гц отдельно — {hz(*values['сахар 40'])}; "
         f"вода 40 Гц отдельно — {hz(*values['вода 40'])}; вместе — {hz(both, both_se)}",
         f"для справки, MN9 {side_ru(other)}: " + "; ".join(
             f"{label} — {hz(*facts.rate(condition, other))}" for label, condition in parts.items())],
        values)


def check_d2(facts: Facts) -> Verdict:
    condition = "water_200hz"
    mn9 = facts.key("MN9")
    other = facts.key("MN9", "left" if mn9.side == "right" else "right")
    mean, se = facts.rate(condition, mn9)
    return Verdict(
        [active_test("MN9 из статьи активен (≥ 1 Гц)", facts, condition, mn9)],
        [f"MN9 из статьи ({side_ru(mn9)}) — {hz(mean, se)}; для справки, MN9 {side_ru(other)} — "
         f"{hz(*facts.rate(condition, other))}"],
        {"MN9 Л": facts.rate(condition, facts.key("MN9", "left")),
         "MN9 П": facts.rate(condition, facts.key("MN9", "right"))})


@dataclass(frozen=True)
class Item:
    key: str
    title: str
    conditions: tuple[str, ...]
    check: object  # функция Facts → Verdict


ITEMS = (
    Item("Б1", "Сахар (одна сторона) → MN9 с обеих сторон, на противоположной сильнее", ("sugar_100hz",), check_b1),
    Item("Б2", "Сахар 10 → 200 Гц: откликнувшихся нейронов от десятков до сотен", tuple(B2_CONDITIONS.values()),
         check_b2),
    Item("Б3", "Сахар + сильная горечь: ответ MN9 на сахар подавлен", ("sugar_100hz", "sugar_100hz_bitter_200hz"),
         check_b3),
    Item("Б4", "Вода 100 Гц → MN9 активен", ("water_100hz",), check_b4),
    Item("Б5", "Касание усика (все JON) → aDN1, aDN2, aBN1 активны", ("antenna_100hz",), check_b5),
    Item("Б6", "JO-CE включает aBN1, JO-F — почти нет", ("jon_ce_100hz", "jon_f_100hz"), check_b6),
    Item("Б7", "Связи перемешаны → MN9 молчит", ("sugar_100hz_shuffled",), check_b7),
    Item("В", "Тень (LPLC2 + LC4, левая сторона) → гигантское волокно", ("shadow_left_100hz",), check_v),
    Item("Д1", "Сахар 40 и вода 40 Гц по отдельности MN9 почти не включают, а вместе включают",
         ("sugar_40hz_water_0hz", "sugar_0hz_water_40hz", "sugar_40hz_water_40hz"), check_d1),
    Item("Д2", "Вода 200 Гц → MN9 активен", ("water_200hz",), check_d2),
)


# ---------- команда ----------

@dataclass
class Results:
    """Всё, что нужно отчёту."""
    conditions: list[Condition]
    keys: list[KeyNeuron]
    reference: dict[str, Run]
    engine: dict[str, Run]
    repeat: dict[str, Run]
    agreements: dict[str, Agreement]
    noise: dict[str, Agreement]
    verdicts: dict[str, tuple[Verdict, Verdict]]  # пункт → (эталон, движок)
    stimulated: dict[str, np.ndarray]
    info: dict


def _connectivity_sha(conditions: list[Condition]) -> dict[str, str]:
    """Суммы файлов связей, на которых считал эталон; движок обязан считать на тех же."""
    with open(PROCESSED_DIR / processed.BUILD_INFO_FILE, encoding="utf-8") as handle:
        build_info = json.load(handle)
    shas = {"original": build_info["inputs_sha256"]["Connectivity_783.parquet"]}
    if any(condition.connectivity == "shuffled" for condition in conditions):
        if not (REFERENCE_DIR / SHUFFLED_FILE).exists():
            raise ValidateError(f"нет data/reference/{SHUFFLED_FILE}. Выполните: uv run flybrain reference")
        shas["shuffled"] = sha256_of(REFERENCE_DIR / SHUFFLED_FILE)
    return shas


def collect(device: str = "auto", rerun: bool = False, log=print) -> Results:
    """Эталон, расчёты движка (готовые или новые) и все пункты проверки."""
    if not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists():
        raise ValidateError("нет подготовленных данных. Выполните: uv run flybrain fetch-data, "
                            "затем uv run flybrain build-data")
    if not (REFERENCE_DIR / INFO_FILE).exists():
        raise ValidateError("нет эталона. Выполните: uv run flybrain reference (около 25 минут)")
    config = load_reference_config()
    conditions, n_trials, duration_ms = config["conditions"], int(config["trials"]), float(config["duration_ms"])
    needed = {condition for item in ITEMS for condition in item.conditions}
    missing = sorted(needed - {condition.key for condition in conditions})
    if missing:
        raise ValidateError(f"в config/reference.yaml нет условий: {', '.join(missing)}")
    groups = processed.load_groups()
    neurons = processed.load_neurons()
    keys = load_key_neurons(groups, neurons, config["summary_readouts"])
    n_neurons = len(neurons)
    shas = _connectivity_sha(conditions)
    with open(REFERENCE_DIR / INFO_FILE, encoding="utf-8") as handle:
        reference_info = json.load(handle)

    log("Эталон Brian2 (C4): читаю повторы…")
    reference, stimulated = {}, {}
    for condition in conditions:
        reference[condition.key] = load_reference_run(condition, groups, n_trials, duration_ms, keys, n_neurons,
                                                      shas[condition.connectivity])
        stimulated[condition.key] = stimulated_of(condition, groups)
    log(f"[OK] эталон: {len(conditions)} условий по {n_trials} повторов, настройки совпадают с нынешними")

    version = engine_version()
    engine_runs: dict[str, dict[str, Run]] = {run: {} for run in RUNS}
    csr_cache: dict[str, dict] = {}
    todo = []
    for condition in conditions:
        stimuli = stimuli_of(condition, groups)
        for run in RUNS:
            meta = result_meta(condition, run, stimuli, n_trials, duration_ms, shas[condition.connectivity], keys,
                               version)
            path = VALIDATION_DIR / f"{condition.key}.{run}.npz"
            saved = None if rerun else load_saved(path, meta)
            if saved is None:
                todo.append((condition, run, stimuli, meta, path))
            else:
                engine_runs[run][condition.key] = saved
    log(f"Движок: расчётов {2 * len(conditions)} (по 2 на условие: основной и «шумовой предел»), "
        f"готовых {2 * len(conditions) - len(todo)}, считать {len(todo)}")
    device_used = None
    started = time.monotonic()
    for number_done, (condition, run, stimuli, meta, path) in enumerate(todo, start=1):
        if condition.connectivity not in csr_cache:
            if condition.connectivity == "shuffled":
                log("  готовлю перемешанные связи (тот же файл, что у эталона)…")
                csr_cache["shuffled"] = load_shuffled_csr(n_neurons)
            else:
                csr_cache["original"] = processed.load_connectivity()
        title = f"  [{number_done}/{len(todo)}] {condition.name_ru}, {'основной' if run == 'main' else 'шумовой предел'}"

        def progress(share: float, title=title) -> None:
            print(f"\r{title}: {100 * share:3.0f} %", end="", flush=True)

        result = run_engine(csr_cache[condition.connectivity], stimuli, engine_seed(condition.key, run), n_trials,
                            duration_ms, keys, device, progress)
        device_used = device_used or _device_label(device)
        print(f"\r{title}: готово за {number(result.seconds, 0)} с", flush=True)
        save_run(path, result, meta, device_used)
        engine_runs[run][condition.key] = result
    if todo:
        log(f"[OK] движок посчитан за {number((time.monotonic() - started) / 60)} мин")

    agreements, noise = {}, {}
    for condition in conditions:
        key = condition.key
        agreements[key] = agreement(reference[key], engine_runs["main"][key], stimulated[key], keys)
        noise[key] = agreement(engine_runs["main"][key], engine_runs["repeat"][key], stimulated[key], keys)
    reference_facts = Facts(reference, keys, stimulated)
    engine_facts = Facts(engine_runs["main"], keys, stimulated)
    verdicts = {item.key: (item.check(reference_facts), item.check(engine_facts)) for item in ITEMS}
    info = {"n_trials": n_trials, "duration_ms": duration_ms, "engine_version": version,
            "device": device_used or _saved_device(conditions), "reference": reference_info,
            "connectivity_sha256": shas}
    return Results(conditions, keys, reference, engine_runs["main"], engine_runs["repeat"], agreements, noise,
                   verdicts, stimulated, info)


def _device_label(device: str) -> str:
    import torch

    from flybrain.engine import pick_device

    chosen = pick_device(device)
    return f"видеокарта {torch.cuda.get_device_name(chosen)}" if chosen.type == "cuda" else "процессор"


def _saved_device(conditions: list[Condition]) -> str:
    """На чём были посчитаны сохранённые результаты (если в этот раз ничего не пересчитывалось)."""
    labels = set()
    for condition in conditions:
        for run in RUNS:
            path = VALIDATION_DIR / f"{condition.key}.{run}.npz"
            with np.load(path) as arrays:
                labels.add(str(arrays["device"]))
    return ", ".join(sorted(labels))


MARKS = {PASS: "✅", FAIL: "❌", BOUNDARY: "⚠️", INFO: "ℹ️"}
MARK_WORDS = {PASS: "да", FAIL: "нет", BOUNDARY: "на границе", INFO: "описание"}


def run_validate(device: str = "auto", rerun: bool = False) -> int:
    from flybrain.report import REPORT_FILE, write_report

    started = time.monotonic()
    print("FlyBrain: проверка движка против эталона Brian2 и выводов статьи (C6)\n")
    try:
        results = collect(device=device, rerun=rerun)
    except ValidateError as exc:
        print(f"\n[ОШИБКА] {exc}", file=sys.stderr)
        return 1
    write_report(results)

    passed = sum(agreement.ok for agreement in results.agreements.values())
    total = len(results.agreements)
    print(f"\nА. Движок против эталона: условий без расхождений {passed} из {total} "
          f"(корреляция ≥ {number(CORR_MIN, 2)}, совпадение ≥ {number(100 * OVERLAP_MIN, 0)} %, ключевые ±15 % или ±3 Гц)")
    for condition in results.conditions:
        agreement = results.agreements[condition.key]
        corr = "—" if math.isnan(agreement.corr) else number(agreement.corr, 3)
        print(f"  {MARKS[PASS if agreement.ok else FAIL]} {condition.name_ru}: корреляция {corr}, "
              f"совпадение {number(100 * agreement.overlap, 0)} %, ключевые нейроны "
              f"{'в допуске' if agreement.keys_ok else 'ВНЕ допуска'}")
    print("\nБ, В, Д. Выводы статьи (эталон Brian2 / движок):")
    for item in ITEMS:
        reference, engine = results.verdicts[item.key]
        print(f"  {item.key:<3} {MARKS[reference.state]} {MARK_WORDS[reference.state]:<11} / "
              f"{MARKS[engine.state]} {MARK_WORDS[engine.state]:<11} {item.title}")
    print(f"\nОтчёт: {REPORT_FILE.relative_to(PROJECT_ROOT).as_posix()} (картинки — docs/validation/). "
          f"Готово за {number((time.monotonic() - started) / 60)} мин.")
    return 0
