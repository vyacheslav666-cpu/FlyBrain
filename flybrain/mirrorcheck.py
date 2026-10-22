"""Проверка правой стороны для «Горечи» и «Касания усика» перед тем, как включать её в интерфейс (D-047).

1. Наборы нейронов: левый список статьи против правых двойников (`flybrain.mirror`), рядом — то же для сахара,
   у которого в статье есть списки для обеих сторон (D-034). Для сведения, без порога.
2. Опыт Б-серии зеркально на движке: горечь — зеркало Б3, мерило — сахар, зеркало Б1. Условия правой стороны —
   `config/mirror_check.yaml`; левые берутся из основной проверки (`config/reference.yaml`).
3. Решение по правилу D-047: двойники есть, на правой стороне тот же вывод, что на левой, и расхождение
   не больше, чем у сахара слева/справа.

Модуль вызывает `flybrain validate`; отчёт — раздел «Е» в docs/VALIDATION.md (`flybrain.report`).
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.paths import CONFIG_DIR
from flybrain.reference import ACTIVE_HZ, Condition, load_reference_config
from flybrain.validate import B3_MAX_SHARE, MARK_WORDS, PASS, Check, Facts, KeyNeuron, Run, ValidateError, \
    Verdict, active_test, hz, number, side_ru

MIRROR_CONFIG = CONFIG_DIR / "mirror_check.yaml"
SUGAR_LEFT, SUGAR_RIGHT = "sugar", "sugar_right"
# Стимулы, у которых в статье список только для левой стороны: левая группа, правая (двойники), название.
STIMULI = (("bitter", "bitter_right", "Горечь"), ("antenna", "antenna_right", "Касание усика"))
KEY_READOUTS = ("MN9", "aDN1", "aDN2", "aBN1", "GF")
# Условия опытов. Левые — из основной проверки, правые — из config/mirror_check.yaml.
LEFT_SUGAR, LEFT_MIXED = "sugar_100hz", "sugar_100hz_bitter_200hz"
RIGHT_SUGAR, RIGHT_MIXED = "sugar_right_100hz", "sugar_right_100hz_bitter_right_200hz"
RIGHT_SUGAR_200, RIGHT_MIXED_200 = "sugar_right_200hz", "sugar_right_200hz_bitter_right_200hz"  # для сведения
# Для объяснения, добавлено после результата (D-047): сахар слева + только те левые нейроны горечи, у которых есть
# двойники (группа PAIRED_LEFT — весь левый тип LB1a).
PAIRED_LEFT, PAIRED_MIXED = "bitter_paired_left", "sugar_100hz_bitter_paired_left_200hz"
MIRROR_CONDITIONS = (RIGHT_SUGAR, RIGHT_MIXED, RIGHT_SUGAR_200, RIGHT_MIXED_200, PAIRED_MIXED)


def load_mirror_conditions(path: Path = MIRROR_CONFIG) -> list[Condition]:
    conditions = load_reference_config(path)["conditions"]
    missing = sorted(set(MIRROR_CONDITIONS) - {condition.key for condition in conditions})
    if missing:
        raise ValidateError(f"в {path.name} нет условий: {', '.join(missing)}")
    return conditions


# ---------- наборы нейронов ----------

class Wiring:
    """Связи модели (CSR из data/processed): сколько синапсов набор даёт каждому нейрону и что приходит к цели."""

    def __init__(self, csr: dict):
        self.indptr = csr["indptr"]
        self.senders = csr["indices"]
        self.weight = csr["weight"].astype(np.int64)
        self.n = len(self.indptr) - 1
        self.receivers = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.indptr))

    def outputs(self, members: np.ndarray) -> np.ndarray:
        """Синапсы от набора к каждому нейрону-получателю, со знаком модели (D-025)."""
        chosen = np.zeros(self.n, dtype=bool)
        chosen[members] = True
        take = chosen[self.senders]
        return np.bincount(self.receivers[take], weights=self.weight[take], minlength=self.n)

    def into(self, target: int, values: np.ndarray) -> float:
        """Σ по отправителям j: синапсы j → цель × values[j]."""
        start, stop = self.indptr[target], self.indptr[target + 1]
        return float((self.weight[start:stop] * values[self.senders[start:stop]]).sum())


@dataclass
class SetStats:
    """Набор нейронов стимула и его связи (для сведения, D-047, пункт 1)."""
    label: str
    side: str  # сторона стимула: «своя» сторона для исполнителей
    count: int
    types: dict[str, int]
    output: int  # всего выходных синапсов набора
    to_motor: int  # прямые синапсы на все моторные нейроны
    to_descending: int  # на все нисходящие
    to_keys: dict[str, int]  # прямые синапсы на ключевых исполнителей: «MN9 своя», «aBN1 другая», …
    mn9_two_step: dict[str, float]  # путь к MN9 через один промежуточный нейрон: «своя», «другая»


def set_stats(label: str, side: str, members: np.ndarray, wiring: Wiring, neurons: pd.DataFrame,
              groups: dict, keys: list[KeyNeuron]) -> SetStats:
    members = np.asarray(members, dtype=np.int64)
    flow = wiring.outputs(members)
    readouts = groups["readouts"]
    types = neurons["cell_type"].iloc[members].fillna("без типа").value_counts()
    to_keys, two_step = {}, {}
    for key in keys:
        if key.readout not in KEY_READOUTS:
            continue
        where = "своя" if key.side == side else "другая"
        to_keys[f"{key.readout} {where}"] = int(flow[key.index])
        if key.readout == "MN9":
            two_step[where] = wiring.into(key.index, flow)
    return SetStats(label, side, len(members), {str(t): int(c) for t, c in sorted(types.items())},
                    int(flow.sum()), int(flow[readouts["motor_all"]["indices"]].sum()),
                    int(flow[readouts["descending_all"]["indices"]].sum()), to_keys, two_step)


# ---------- опыт ----------

@dataclass
class Target:
    """Одна цель зеркального опыта: величина при левом стимуле и при правом (среднее, погрешность)."""
    label: str
    left: tuple[float, float] | None  # None — величину нельзя посчитать (например, подавлять нечего)
    right: tuple[float, float] | None
    left_text: str
    right_text: str
    floor: float = 0.0  # если обе величины ниже — разницы нет (у частот это порог «активного», 1 Гц)

    @property
    def discrepancy(self) -> float:
        """1 − меньшее / большее: 0 — одинаково, 1 — одна из сторон ничего не делает (D-047, пункт 3)."""
        if self.left is None or self.right is None:
            return 1.0
        a, b = max(self.left[0], 0.0), max(self.right[0], 0.0)
        if max(a, b) <= 0 or max(a, b) < self.floor:
            return 0.0
        return 1.0 - min(a, b) / max(a, b)


def worst(targets: list[Target]) -> float:
    return max(target.discrepancy for target in targets)


def suppression(facts: Facts, sugar: str, mixed: str, key: KeyNeuron) -> tuple[float, float] | None:
    """Какую долю ответа MN9 на сахар убирает горечь: 1 − MN9(сахар + горечь) / MN9(сахар), от 0 до 1."""
    base, base_se = facts.rate(sugar, key)
    if base < ACTIVE_HZ:
        return None  # MN9 на сахар не отвечает — подавлять нечего
    rest, rest_se = facts.rate(mixed, key)
    share = min(max(1.0 - rest / base, 0.0), 1.0)
    return share, math.hypot(rest_se / base, rest * base_se / base**2)


def percent(value: tuple[float, float] | None) -> str:
    if value is None:
        return "нельзя посчитать: MN9 на сахар не отвечает"
    mean, se = value
    return f"{number(100 * mean, 0)} ± {number(100 * se, 0)} %"


def check_b3_side(facts: Facts, sugar: str, mixed: str, key: KeyNeuron, where: str) -> Verdict:
    """Правило Б3 (D-031) для стимулов одной стороны: MN9 на противоположной стороне `key`."""
    base, base_se = facts.rate(sugar, key)
    rest, rest_se = facts.rate(mixed, key)
    share = f" (это {number(100 * rest / base, 0)} % от ответа на сахар)" if base > 0 else ""
    return Verdict(
        [active_test("без горечи MN9 отвечает на сахар (≥ 1 Гц)", facts, sugar, key),
         Check("с горечью — не больше половины ответа на сахар", B3_MAX_SHARE * base - rest,
               math.hypot(B3_MAX_SHARE * base_se, rest_se))],
        [f"MN9 {side_ru(key)} (противоположная сторона): сахар {where} 100 Гц — {hz(base, base_se)}; "
         f"сахар {where} 100 + горечь {where} 200 Гц — {hz(rest, rest_se)}{share}"],
        {"сахар": (base, base_se), "сахар + горечь": (rest, rest_se)})


@dataclass
class MirrorResult:
    key: str
    name_ru: str
    left_group: str
    right_group: str
    matches: list[dict]  # mirror_types из groups.json: тип, сколько в списке, сколько слева и справа, двойники
    stats: list[SetStats] = field(default_factory=list)
    left: Verdict | None = None  # опыт Б-серии с левым стимулом (как в основной проверке)
    right: Verdict | None = None  # то же зеркально
    targets: list[Target] = field(default_factory=list)
    info: list[str] = field(default_factory=list)  # для сведения, на решение не влияет
    explain: list[str] = field(default_factory=list)  # для объяснения, добавлено после результата
    include: bool = False
    reasons: list[str] = field(default_factory=list)  # почему кнопка не включается

    @property
    def twins(self) -> int:
        return sum(len(match["twins"]) for match in self.matches)

    @property
    def paired(self) -> int:
        return sum(len(match["listed"]) for match in self.matches if match["twins"])

    @property
    def listed(self) -> int:
        return sum(len(match["listed"]) for match in self.matches)

    @property
    def discrepancy(self) -> float | None:
        return worst(self.targets) if self.targets else None


@dataclass
class MirrorCheck:
    conditions: list[Condition]
    runs: dict[str, Run]
    sugar_stats: list[SetStats]
    sugar_targets: list[Target]
    results: list[MirrorResult]

    @property
    def sugar_discrepancy(self) -> float:
        return worst(self.sugar_targets)


def sugar_targets(facts: Facts) -> list[Target]:
    """Мерило: сахар слева (список статьи) и справа (другой список статьи, D-034), 100 Гц — зеркало Б1."""
    targets = []
    for label, left_side in (("MN9 на противоположной стороне", "right"), ("MN9 на той же стороне", "left")):
        left_key = facts.key("MN9", left_side)
        right_key = facts.key("MN9", "left" if left_side == "right" else "right")
        left, right = facts.rate(LEFT_SUGAR, left_key), facts.rate(RIGHT_SUGAR, right_key)
        targets.append(Target(label, left, right, f"MN9 {side_ru(left_key)} — {hz(*left)}",
                              f"MN9 {side_ru(right_key)} — {hz(*right)}", floor=ACTIVE_HZ))
    return targets


def bitter_experiment(result: MirrorResult, facts: Facts) -> None:
    """Горечь: зеркало Б3 (D-047, пункт 2) и расхождение по доле подавления (пункт 3)."""
    result.left = check_b3_side(facts, LEFT_SUGAR, LEFT_MIXED, facts.key("MN9", "right"), "слева")
    result.right = check_b3_side(facts, RIGHT_SUGAR, RIGHT_MIXED, facts.key("MN9", "left"), "справа")
    for label, left_side in (("MN9 на противоположной стороне", "right"), ("MN9 на той же стороне", "left")):
        left_key = facts.key("MN9", left_side)
        right_key = facts.key("MN9", "left" if left_side == "right" else "right")
        left = suppression(facts, LEFT_SUGAR, LEFT_MIXED, left_key)
        right = suppression(facts, RIGHT_SUGAR, RIGHT_MIXED, right_key)
        result.targets.append(Target(f"{label}: доля ответа на сахар, которую убирает горечь", left, right,
                                     f"MN9 {side_ru(left_key)} — {percent(left)}",
                                     f"MN9 {side_ru(right_key)} — {percent(right)}"))
    for key in (facts.key("MN9", "left"), facts.key("MN9", "right")):
        base, rest = facts.rate(RIGHT_SUGAR_200, key), facts.rate(RIGHT_MIXED_200, key)
        removed = percent(suppression(facts, RIGHT_SUGAR_200, RIGHT_MIXED_200, key))
        result.info.append(f"MN9 {side_ru(key)}: сахар справа 200 Гц — {hz(*base)}; вместе с горечью справа "
                           f"200 Гц — {hz(*rest)} (убрано {removed})")
    for key in (facts.key("MN9", "right"), facts.key("MN9", "left")):
        base, rest = facts.rate(LEFT_SUGAR, key), facts.rate(PAIRED_MIXED, key)
        removed = percent(suppression(facts, LEFT_SUGAR, PAIRED_MIXED, key))
        result.explain.append(f"MN9 {side_ru(key)}: сахар слева 100 Гц — {hz(*base)}; вместе с горечью слева, "
                              f"только LB1a, 200 Гц — {hz(*rest)} (убрано {removed})")


def decide(result: MirrorResult, sugar_discrepancy: float) -> None:
    """Правило D-047, пункт 4: двойники есть, тот же вывод справа, расхождение не больше, чем у сахара."""
    reasons = []
    if not result.twins:
        reasons.append("правых двойников по правилу нет ни у одного типа клеток")
    elif result.right is None:
        reasons.append("для этого стимула зеркальный опыт не настроен")
    else:
        if result.left.state != PASS or result.right.state != PASS:
            reasons.append(f"вывод опыта слева — «{MARK_WORDS[result.left.state]}», справа — "
                           f"«{MARK_WORDS[result.right.state]}», а нужно «да» с обеих сторон")
        if result.discrepancy > sugar_discrepancy:
            reasons.append(f"расхождение {number(result.discrepancy, 2)} больше, чем у сахара "
                           f"({number(sugar_discrepancy, 2)})")
    result.reasons = reasons
    result.include = not reasons


def collect(groups: dict, neurons: pd.DataFrame, keys: list[KeyNeuron], runs: dict[str, Run],
            stimulated: dict[str, np.ndarray], wiring: Wiring, conditions: list[Condition]) -> MirrorCheck:
    """Всё для раздела «Е» отчёта. `runs` — основные расчёты движка вместе с условиями правой стороны."""
    stimuli = groups["stimuli"]
    facts = Facts(runs, keys, stimulated)

    def stats(label, side, members):
        return set_stats(label, side, members, wiring, neurons, groups, keys)

    sugar = [stats("сахар слева — список статьи", "left", stimuli[SUGAR_LEFT]["indices"]),
             stats("сахар справа — список статьи", "right", stimuli[SUGAR_RIGHT]["indices"])]
    check = MirrorCheck(conditions, runs, sugar, sugar_targets(facts), [])
    for key, right_key, name in STIMULI:
        right = stimuli[right_key]
        if right.get("mirror_of") != key:
            raise ValidateError(f"группа «{right_key}» должна быть двойниками «{key}» (mirror_of, D-047)")
        result = MirrorResult(key, name, key, right_key, right["mirror_types"])
        result.stats.append(stats(f"{name.lower()} слева — список статьи", "left", stimuli[key]["indices"]))
        result.stats.append(stats(f"{name.lower()} справа — двойники", "right", right["indices"]))
        if result.twins:
            paired = sorted(i for match in result.matches if match["twins"] for i in match["listed"])
            if key == "bitter" and stimuli[PAIRED_LEFT]["indices"] != paired:
                raise ValidateError(f"группа «{PAIRED_LEFT}» должна совпадать с левыми нейронами горечи, у которых "
                                    "нашлись двойники (config/neuron_groups.yaml, D-047)")
            result.stats.append(stats("только пары: слева", "left", np.array(paired, dtype=np.int64)))
            result.stats.append(stats("только пары: справа", "right", right["indices"]))
        if key == "bitter" and result.twins:
            bitter_experiment(result, facts)
        decide(result, check.sugar_discrepancy)
        check.results.append(result)
    return check
