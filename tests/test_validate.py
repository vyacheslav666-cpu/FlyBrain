"""Проверка C6 без расчёта мозга: правила пунктов А, Б, Д на искусственных числах (D-031)."""

import numpy as np
import pytest

from flybrain import processed
from flybrain.paths import PROCESSED_DIR
from flybrain.reference import load_reference_config
from flybrain.validate import BOUNDARY, FAIL, INFO, ITEMS, PASS, Facts, KeyNeuron, Run, Check, Verdict, agreement, \
    check_b2, check_b6, check_d2, csr_summing_duplicates, engine_seed, mean_se

N_TRIALS = 30


def _run(totals, key_counts=None) -> Run:
    totals = np.asarray(totals, dtype=np.int64)
    key_counts = np.zeros((N_TRIALS, 0), dtype=np.int64) if key_counts is None else np.asarray(key_counts)
    return Run(totals, key_counts, duration_s=1.0)


def test_csr_sums_duplicate_pairs():
    rng = np.random.default_rng(1)
    n = 7
    pre, post = rng.integers(0, n, 300), rng.integers(0, n, 300)  # много повторов и петель «сам на себя»
    weight = rng.integers(-30, 30, 300)
    csr = csr_summing_duplicates(post, pre, weight, n)
    dense = np.zeros((n, n), dtype=np.int64)
    np.add.at(dense, (post, pre), weight)  # Brian2: каждая связь добавляет свой вес
    rebuilt = np.zeros((n, n), dtype=np.int64)
    for row in range(n):
        span = slice(csr["indptr"][row], csr["indptr"][row + 1])
        assert len(set(csr["indices"][span])) == len(csr["indices"][span])  # пары не повторяются
        rebuilt[row, csr["indices"][span]] = csr["weight"][span]
    np.testing.assert_array_equal(rebuilt, dense)


def test_agreement_thresholds():
    keys = [KeyNeuron("MN9", "left", 0, "1", False), KeyNeuron("MN9", "right", 1, "2", True)]
    base = _run([30 * 60, 30 * 40, 30 * 5, 30 * 2, 0, 30 * 99])
    same = agreement(base, base, stimulated=np.array([5]), keys=keys)
    assert same.corr == pytest.approx(1.0) and same.overlap == 1.0 and same.ok
    assert same.n_union == 4  # стимулируемый нейрон 5 не считается
    # MN9 Л: 60 → 68 Гц — в допуске (15 % от 60 = 9 Гц); MN9 П: 40 → 47 — больше и 3 Гц, и 15 % (6 Гц).
    shifted = _run([30 * 68, 30 * 47, 30 * 5, 30 * 2, 0, 30 * 99])
    result = agreement(base, shifted, np.array([5]), keys)
    assert [diff.ok for diff in result.keys] == [True, False] and not result.keys_ok
    # Набор активных: нейрон 3 исчез (2 → 0 Гц), появился 4 (0 → 1 Гц): общих 3 из 5.
    other = _run([30 * 60, 30 * 40, 30 * 5, 0, 30, 30 * 99])
    assert agreement(base, other, np.array([5]), keys).overlap == pytest.approx(3 / 5)
    # Все молчат в обоих расчётах — наборы совпадают, корреляцию посчитать нельзя.
    silent = agreement(_run([0] * 6), _run([0] * 6), np.array([5]), keys)
    assert silent.overlap == 1.0 and np.isnan(silent.corr) and not silent.ok


def test_check_states():
    assert Check("≥ порога", 0.0).state == PASS
    assert Check("< порога", 0.0, strict=True).state == FAIL
    assert Check("далеко выше", 1.0, se=0.4).state == PASS
    assert Check("далеко ниже", -1.0, se=0.4).state == FAIL
    assert Check("рядом", 0.5, se=0.3).state == BOUNDARY  # 0,5 < 2 × 0,3
    assert Check("ровно на пороге", 0.0, se=0.2, strict=True).state == BOUNDARY
    assert Verdict([Check("", 1), Check("", 0.1, 0.1)], []).state == BOUNDARY
    assert Verdict([Check("", -1), Check("", 0.1, 0.1)], []).state == FAIL
    assert Verdict([], []).state == INFO
    mean, se = mean_se(np.array([0, 1, 2, 1, 1]))
    assert mean == 1.0 and se == pytest.approx(np.std([0, 1, 2, 1, 1], ddof=1) / np.sqrt(5))


def _abn1_facts(ce_counts, f_counts) -> Facts:
    keys = [KeyNeuron("aBN1", "left", 0, "1", True), KeyNeuron("aBN1", "right", 1, "2", False)]
    runs = {}
    for condition, counts in (("jon_ce_100hz", ce_counts), ("jon_f_100hz", f_counts)):
        key_counts = np.stack([counts, np.zeros(N_TRIALS, dtype=np.int64)], axis=1)
        runs[condition] = _run(key_counts.sum(axis=0), key_counts)
    return Facts(runs, keys, {condition: np.zeros(0, dtype=np.int64) for condition in runs})


def test_b6_on_the_threshold_is_boundary():
    ce = np.full(N_TRIALS, 26)
    exactly_one = np.tile([0, 1, 2], N_TRIALS // 3)  # 30 спайков за 30 повторов = ровно 1 Гц
    verdict = check_b6(_abn1_facts(ce, exactly_one))
    assert [test.state for test in verdict.tests] == [PASS, BOUNDARY] and verdict.state == BOUNDARY
    assert "30 спайков за 30 повторов" in verdict.numbers[1]
    assert check_b6(_abn1_facts(ce, np.zeros(N_TRIALS, dtype=np.int64))).state == PASS
    assert check_b6(_abn1_facts(ce, np.full(N_TRIALS, 5))).state == FAIL


def test_b2_counts_above_zero_without_stimulated():
    def facts_for(counts: dict[int, int]) -> Facts:
        runs, stimulated = {}, {}
        for freq, condition in (("10", "sugar_10hz"), ("50", "sugar_50hz"), ("100", "sugar_100hz"),
                                ("200", "sugar_200hz")):
            totals = np.zeros(1000, dtype=np.int64)
            totals[:20] = 30 * 99  # стимулируемые
            totals[20:20 + counts[int(freq)]] = 3  # 0,1 Гц: «> 0 Гц» по статье, но не «≥ 1 Гц»
            runs[condition] = _run(totals)
            stimulated[condition] = np.arange(20)
        return Facts(runs, [], stimulated)

    # Основной счёт — вместе с 20 стимулируемыми, как get_rate у Shiu (D-032); без них — для справки.
    verdict = check_b2(facts_for({10: 22, 50: 346, 100: 392, 200: 433}))
    assert verdict.state == PASS
    assert verdict.values["counts"] == {10: 42, 50: 366, 100: 412, 200: 453}
    assert verdict.values["without_stimulated"][10] == 22 and verdict.values["active"][200] == 0
    assert check_b2(facts_for({10: 22, 50: 30, 100: 40, 200: 50})).state == FAIL  # 200 Гц: 70 — не «сотни»
    assert check_b2(facts_for({10: 22, 50: 400, 100: 392, 200: 433})).state == FAIL  # не растёт
    # Без стимулируемых при 10 Гц было бы 4 (не «десятки»), но основной счёт — 24: пункт выполнен.
    assert check_b2(facts_for({10: 4, 50: 346, 100: 392, 200: 433})).state == PASS


def test_d2_uses_paper_mn9():
    keys = [KeyNeuron("MN9", "left", 0, "1", False), KeyNeuron("MN9", "right", 1, "720575940660219265", True)]

    def facts_for(left: int, right: int) -> Facts:
        key_counts = np.stack([np.full(N_TRIALS, left), np.full(N_TRIALS, right)], axis=1)
        return Facts({"water_200hz": _run(key_counts.sum(axis=0), key_counts)}, keys,
                     {"water_200hz": np.zeros(0, dtype=np.int64)})

    assert check_d2(facts_for(0, 5)).state == PASS
    assert check_d2(facts_for(5, 0)).state == FAIL  # активен только левый, а правило — про MN9 из статьи


def test_engine_seeds_distinct():
    config = load_reference_config()
    seeds = {engine_seed(condition.key, run) for condition in config["conditions"] for run in ("main", "repeat")}
    assert len(seeds) == 2 * len(config["conditions"])


def test_items_use_known_conditions():
    keys = {condition.key for condition in load_reference_config()["conditions"]}
    for item in ITEMS:
        assert set(item.conditions) <= keys, item.key


@pytest.mark.skipif(not (PROCESSED_DIR / processed.GROUPS_FILE).exists(), reason="нет data/processed")
def test_paper_neurons_are_marked():
    from flybrain.validate import load_key_neurons

    keys = load_key_neurons(processed.load_groups(), processed.load_neurons(), ["MN9", "aDN1", "aDN2", "aBN1", "GF"])
    paper = {key.readout: (key.root_id, key.side) for key in keys if key.paper}
    # Номера из figures.ipynb Shiu (id_mn9, id_DN1_1, id_DN2_l, id_aBN1); стороны — по нынешней таблице (D-011).
    assert paper == {"MN9": ("720575940660219265", "right"), "aDN1": ("720575940616185531", "right"),
                     "aDN2": ("720575940629806974", "right"), "aBN1": ("720575940630907434", "left")}
    assert len(keys) == 10


def test_summary_for_about_window_uses_report_marks():
    from types import SimpleNamespace

    from flybrain.report import summary_data

    ok, bad = Verdict([Check("ok", 1.0)], ["1"]), Verdict([Check("bad", -1.0)], ["2"])
    edge = Verdict([Check("edge", 0.1, se=0.2)], ["3"])
    shadow = Verdict([], ["4"], {"GF Л": (144.0, 0.5), "GF П": (0.2, 0.1)})
    verdicts = {item.key: (ok, ok) for item in ITEMS}
    verdicts.update({"Б4": (bad, bad), "Б6": (edge, ok), "В": (shadow, shadow)})
    results = SimpleNamespace(verdicts=verdicts, conditions=[SimpleNamespace(key="a"), SimpleNamespace(key="b")],
                              agreements={"a": SimpleNamespace(ok=True), "b": SimpleNamespace(ok=False)}, mirror=None)
    data = summary_data(results)
    assert data["a"] == {"passed": 1, "total": 2} and data["mirror"] == []
    assert [item["key"] for item in data["items"]] == [item.key for item in ITEMS]
    items = {item["key"]: item for item in data["items"]}
    assert items["Б1"]["words"] == "Повторился и в эталоне Brian2, и в движке."
    assert (items["Б4"]["reference"], items["Б4"]["engine"]) == (FAIL, FAIL)
    assert "дело не в движке" in items["Б4"]["words"]
    assert items["Б6"]["reference"] == BOUNDARY and items["Б6"]["words"].startswith("Эталон — на границе, движок — да.")
    assert items["В"]["reference"] == INFO and "только левое" in items["В"]["words"]
    assert items["Д1"]["added_after_reference"] and not items["Б1"]["added_after_reference"]
    assert data["marks"][BOUNDARY] == {"mark": "⚠️", "word": "на границе"}
