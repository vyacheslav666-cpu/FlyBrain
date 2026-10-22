"""Эксперимент P1 «психоделик»: серотониновые нейроны, форма эффекта, модальности и метрики (D-053…D-056)."""

import numpy as np
import pandas as pd
import pytest

from flybrain import processed
from flybrain.modality import MODALITIES, input_shares
from flybrain.paths import PROCESSED_DIR, RAW_DIR
from flybrain.psy import KeyNeuron, RunResult, RunSpec, Setup, load_config, metrics, plan, readout_threshold
from flybrain.serotonin import SerotoninSets, mentions, read_annotations, saturation, serotonin_methods, \
    serotonin_sets, serotonin_synapses, threshold_shift

HAVE_DATA = (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists() and \
    (RAW_DIR / "flywire" / "Supplemental_file1_neuron_annotations.tsv").exists()


def test_serotonin_methods_follow_semicolon_parts():
    # k-я часть медиаторов — из k-го источника; способ — в скобках в конце источника.
    assert serotonin_methods("serotonin", "Sitaraman et al., 2008 (immuno)") == ["immuno"]
    assert serotonin_methods("natalisin; serotonin, MIP; acetylcholine-negative",
                             "Jiang 2013 (immuno); Scheunemann 2018 (immuno); Croset 2018 (scRNA-seq)") == ["immuno"]
    assert serotonin_methods("glutamate, serotonin, proctolin; glutamate",
                             "Wolff et al., 2024 (EASI-FISH); Turner-Evans 2020 (scRNA-seq)") == ["easi-fish"]
    assert serotonin_methods("serotonin; amnesiac; gaba; gaba, serotonin",
                             "Lee 2011 (immuno); Waddell 2000 (immuno); Haynes 2015 (immuno); Aso 2019 (scRNA-seq)") \
        == ["immuno", "scrna-seq"]
    # «serotonin-negative» — не серотонин; источника нет — способ не указан.
    assert serotonin_methods("acetylcholine, serotonin-negative", "Davis 2020 (TAPIN)") == []
    assert serotonin_methods("allatostatin-c; serotonin", "Cheong 2023 (MCFO)") == ["не указан"]
    assert serotonin_methods(None, None) == [] and serotonin_methods("", "") == []
    assert mentions("histamine; acetylcholine, histamine", "histamine")
    assert not mentions("acetylcholine-negative", "acetylcholine")


def test_saturation_and_threshold_shift():
    s = np.array([0, 1, 5, 20, 1000])
    assert saturation(s, 5) == pytest.approx([0, 1 / 6, 0.5, 0.8, 1000 / 1005])
    assert threshold_shift(s, 2.0, 1.0, 5) == pytest.approx(2 * saturation(s, 5))
    zero = threshold_shift(s, 0.0, 1.0, 5)
    assert not zero.any()  # при d = 0 — ровно нули: движок идёт прежним путём
    with pytest.raises(ValueError):
        threshold_shift(s, -1.0, 1.0, 5)


def test_serotonin_synapses_counts_inputs_from_sources():
    # 0 → 2 (3 синапса, возбуждает), 1 → 2 (−4, тормозит), 0 → 1 (2): источники — нейрон 0 и 1.
    indptr = np.array([0, 0, 1, 3])
    indices = np.array([0, 0, 1], dtype=np.int32)
    weight = np.array([2, 3, -4], dtype=np.int16)
    assert list(serotonin_synapses(indptr, indices, weight, np.array([True, False, False]))) == [0, 2, 3]
    assert list(serotonin_synapses(indptr, indices, weight, np.array([True, True, False]))) == [0, 2, 7]


def test_readout_threshold_interpolates_between_grid_points():
    assert readout_threshold([(10, 0.0), (20, 0.5), (30, 1.5)]) == ("между", pytest.approx(25.0))
    assert readout_threshold([(10, 2.0), (20, 5.0)]) == ("не выше", 10)
    assert readout_threshold([(10, 0.0), (20, 0.2)]) == ("выше", 20)


def _tiny_neurons(super_class, cell_class) -> pd.DataFrame:
    return pd.DataFrame({"super_class": super_class, "cell_class": cell_class, "nt": "acetylcholine",
                         "cell_type": None, "root_id": np.arange(len(super_class)) + 10**17})


def test_input_shares_absorb_at_sensory_neurons():
    # 0 — фоторецептор, 1 — механорецептор; 2 получает 3 синапса от 0 и 1 синапс (тормозящий) от 1;
    # 3 получает только от 2 — у него те же доли через шаг; 4 ни от кого не получает.
    neurons = _tiny_neurons(["sensory", "sensory", "central", "central", "central"],
                            ["visual", "mechanosensory", None, None, None])
    indptr = np.array([0, 0, 0, 2, 3, 3])
    indices = np.array([0, 1, 2], dtype=np.int32)
    weight = np.array([3, -1, 5], dtype=np.int16)
    shares, dominant = input_shares(neurons, indptr, indices, weight)
    visual, mechano = MODALITIES.index("visual"), MODALITIES.index("mechano")
    assert shares[2, visual] == pytest.approx(0.75) and shares[2, mechano] == pytest.approx(0.25)
    assert shares[3] == pytest.approx(shares[2])
    assert list(dominant) == [visual, mechano, visual, visual, -1]


def _fake_setup(config) -> Setup:
    """Сеть из 6 нейронов: 0 — стимул (сахар), 1–5 — остальные; «исполнители» — нейроны 1–5."""
    n = 6
    neurons = _tiny_neurons(["sensory", "central", "optic", "central", "descending", "motor"],
                            ["gustatory", None, None, None, None, None])
    neurons["side"] = "left"
    shares = np.zeros((n, len(MODALITIES)), dtype=np.float32)
    shares[[0, 1, 4, 5], MODALITIES.index("gustatory")] = 1
    shares[[2, 3], MODALITIES.index("visual")] = 1
    dominant = shares.argmax(1)
    roots = {c.readout_root_id for c in config.conditions}
    keys = [KeyNeuron(name, "left", 1 + k % 5, root) for k, (name, root) in enumerate(
        [("MN9", r) for r in sorted(roots)] + [("x", "0")])]
    sets = SerotoninSets({c: np.zeros(n, dtype=bool) for c in ("strict", "broad", "predicted")}, [[]] * n,
                         np.zeros(n), np.array([None] * n, dtype=object))
    groups = {"stimuli": {c.group: {"indices": [0]} for c in config.conditions}}
    return Setup(config, neurons, {}, groups, sets, {}, shares, dominant, keys, {})


def test_metrics_count_responders_diversity_and_runaway():
    config = load_config()
    setup = _fake_setup(config)
    condition = config.conditions[0]
    spec = RunSpec("main", condition.key, condition.hz, 1.0, config.criterion, 0.0, 2, 0.0, 1000.0, 500.0)
    # Окно стимула: нейрон 0 (стимул) — 80 спайков, 1 — 4, 2 — 4, 3 — 0, 4 — 1 (за 2 повтора по 1 с).
    totals = np.array([80, 4, 4, 0, 1, 0])
    edges = np.array([0.0, 800.0, 1000.0, 1300.0, 1500.0])
    late = np.array([[0, 0], [5, 7], [10, 10], [10, 10], [10, 10]])  # после стимула — тишина
    result = RunResult(totals, np.zeros(6, dtype=np.int64), np.zeros((2, len(setup.keys)), dtype=np.int64),
                       edges, late, 1.0)
    m = metrics(setup, spec, result)
    assert m.responders_b2 == 4 and m.responders == 3 and m.active == 2  # 4/2 с = 2 Гц ≥ 1, 1/2 с — нет
    p = np.array([4, 4, 1]) / 9
    assert m.n_eff == pytest.approx(np.exp(-(p * np.log(p)).sum()))
    assert m.visual_system == 1 and m.foreign_share == pytest.approx(1 / 3)  # «чужой» — зрительный нейрон 2
    assert m.late_rate == pytest.approx(8 / 2 / 0.2) and m.after_rate == 0 and m.runaway is False
    assert m.sustained_trials == 0 and m.after_classes == {}
    # Та же активность, но после стимула не гаснет: 300–500 мс после — столько же, сколько в конце стимула.
    still = np.array([[0, 0], [5, 7], [10, 10], [12, 14], [17, 17]])
    m2 = metrics(setup, spec, RunResult(totals, np.zeros(6, dtype=np.int64), result.key_counts, edges, still, 1.0))
    assert m2.after_ratio == pytest.approx(1.0) and m2.runaway is True and m2.sustained_trials == 2


def test_config_and_plan():
    config = load_config()
    assert config.criterion == "strict" and config.sensitivity_criterion == "broad"
    assert 0.0 in config.runaway_doses
    specs = plan(config, ["runaway", "main", "thresholds", "spont", "broad"])
    assert len(specs) == len(set(specs)) and len({s.key for s in specs}) == len(specs)
    assert all(s.sigma > 0 for s in specs if s.part == "spont") and all(s.sigma == 0 for s in specs
                                                                        if s.part != "spont")


@pytest.mark.skipif(not HAVE_DATA, reason="нужны data/processed и таблица аннотаций")
def test_serotonin_sets_on_flywire_783():
    neurons = processed.load_neurons()
    sets = serotonin_sets(neurons, read_annotations())
    strict, broad, predicted = (sets.masks[c] for c in ("strict", "broad", "predicted"))
    assert (strict.sum(), broad.sum(), predicted.sum()) == (62, 197, 2201)
    assert not (strict & ~broad).any()  # строгий набор — часть широкого
    types = set(neurons["cell_type"][strict])
    assert {"CSD", "DPM", "5-HTPMPD01", "5-HTPMPV01", "5-HTPMPV03"} <= types
    assert not {"Delta7", "FC3", "hDeltaC"} & types  # серотонин у них — только по РНК (EASI-FISH)
