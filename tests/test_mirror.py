"""Тесты правых двойников и проверки правой стороны (D-047): правило «тип клетки + сторона» и расхождение."""

import numpy as np
import pandas as pd
import pytest

from flybrain.mirror import match_twins, twins_of
from flybrain.mirrorcheck import MirrorResult, Target, Wiring, decide, worst
from flybrain.validate import PASS, Check, Verdict


def _neurons() -> pd.DataFrame:
    rows = [
        ("A", "left"), ("A", "left"), ("A", "right"), ("A", "right"),  # 0–3: весь тип в списке, 2 и 2
        ("B", "left"), ("B", "left"), ("B", "right"), ("B", "right"), ("B", "right"),  # 4–8: 2 слева, 3 справа
        ("C", "left"), ("C", "left"), ("C", "left"), ("C", "right"), ("C", "right"), ("C", "right"),  # 9–14
        (None, "left"),  # 15: без типа
        ("E", "left"), ("E", "right"),  # 16–17: один к одному
        ("F", "right"),  # 18: правый нейрон в «левом» списке
    ]
    return pd.DataFrame(rows, columns=["cell_type", "side"])


def test_twins_only_when_the_whole_left_type_is_listed_and_counts_match():
    matches = {match.cell_type: match for match in match_twins([0, 1, 4, 5, 9, 10, 15, 16, 18], _neurons())}
    assert matches["A"].twins == [2, 3] and matches["A"].reason == ""
    assert matches["E"].twins == [17]  # nL = nR = 1 — обычный двойник «один к одному»
    assert matches["B"].twins == [] and matches["B"].reason == "count_differs"  # весь тип, но справа 3, слева 2
    assert (matches["B"].left_total, matches["B"].right_total) == (2, 3)
    assert matches["C"].twins == [] and matches["C"].reason == "part_of_type"  # в списке 2 из 3 левых
    assert matches[None].reason == "no_type" and matches["F"].reason == "not_left"
    assert len(matches["C"].listed) == 2 and matches["C"].left_total == 3


def test_all_twins_are_collected_and_nothing_is_guessed():
    matches = match_twins([0, 1, 4, 5, 9, 10, 16], _neurons())
    assert twins_of(matches).tolist() == [2, 3, 17]
    assert twins_of(match_twins([9], _neurons())).tolist() == []  # двойников нет — группа пуста


def test_discrepancy_is_one_minus_smaller_over_larger():
    assert Target("MN9", (60.0, 1.0), (1.8, 0.2), "", "").discrepancy == pytest.approx(0.97)
    assert Target("MN9", (0.3, 0.1), (0.0, 0.0), "", "", floor=1.0).discrepancy == 0.0  # обе ниже 1 Гц
    assert Target("доля", (1.0, 0.0), None, "", "").discrepancy == 1.0  # подавлять было нечего
    assert Target("доля", (0.0, 0.0), (0.0, 0.0), "", "").discrepancy == 0.0
    assert worst([Target("a", (1.0, 0), (0.9, 0), "", ""), Target("b", (1.0, 0), (0.5, 0), "", "")]) == 0.5


def test_wiring_counts_synapses_from_a_set_and_paths_through_one_neuron():
    # Получатель 2 ← 0 (3 синапса) и 1 (−2); получатель 3 ← 2 (4).
    csr = {"indptr": np.array([0, 0, 0, 2, 3]), "indices": np.array([0, 1, 2]), "weight": np.array([3, -2, 4])}
    wiring = Wiring(csr)
    flow = wiring.outputs(np.array([0, 1]))
    assert flow.tolist() == [0, 0, 1, 0]
    assert wiring.into(3, flow) == 4.0  # 0,1 → 2 → 3: (3 − 2) × 4


def _verdict(state_margin: float) -> Verdict:
    return Verdict([Check("т", state_margin)], [])


def _result(twins: bool) -> MirrorResult:
    match = {"cell_type": "LB1a", "listed": [1], "left_total": 1, "right_total": 1, "twins": [2] if twins else [],
             "reason": "" if twins else "part_of_type"}
    return MirrorResult("bitter", "Горечь", "bitter", "bitter_right", [match])


def test_button_needs_twins_same_verdict_and_no_bigger_gap_than_sugar():
    empty = _result(twins=False)
    decide(empty, 0.97)
    assert not empty.include and "двойников" in empty.reasons[0]

    good = _result(twins=True)
    good.left, good.right = _verdict(1.0), _verdict(1.0)
    good.targets = [Target("MN9", (1.0, 0), (0.9, 0), "", "")]
    decide(good, 0.5)
    assert good.left.state == PASS and good.include and not good.reasons

    worse = _result(twins=True)
    worse.left, worse.right = _verdict(1.0), _verdict(-1.0)
    worse.targets = [Target("MN9", (1.0, 0), (0.2, 0), "", "")]
    decide(worse, 0.5)
    assert not worse.include and len(worse.reasons) == 2  # и вывод другой, и расхождение больше, чем у сахара
