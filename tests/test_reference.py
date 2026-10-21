"""Проверки эталонных прогонов (C4) без запуска Brian2: условия, перемешивание связей, зёрна, частоты."""

import numpy as np
import pandas as pd
import pytest

from flybrain import processed
from flybrain.paths import PROCESSED_DIR
from flybrain.reference import condition_rates, load_reference_config, shuffle_connectivity, trial_seed

# Условия из docs/PLAN_M1.md, коммит C4.
PLAN_CONDITIONS = {
    "sugar_10hz", "sugar_50hz", "sugar_100hz", "sugar_200hz", "sugar_100hz_bitter_200hz", "water_100hz",
    "antenna_100hz", "jon_ce_100hz", "jon_f_100hz", "shadow_left_100hz", "sugar_100hz_shuffled",
}
# Добавлены в C6 после того, как увидели эталон (пункт Д1, D-031).
ADDED_CONDITIONS = {"sugar_40hz_water_0hz", "sugar_0hz_water_40hz", "sugar_40hz_water_40hz"}


def test_config_has_plan_conditions():
    config = load_reference_config()
    assert config["trials"] == 30
    assert config["duration_ms"] == 1000
    assert {condition.key for condition in config["conditions"]} == PLAN_CONDITIONS | ADDED_CONDITIONS
    by_key = {condition.key: condition for condition in config["conditions"]}
    for key in ADDED_CONDITIONS:
        assert "после того, как увидели эталон" in by_key[key].honesty_ru
    # Как сетка Shiu Sugar_Water_*: сахар — первая группа, вода — вторая; «отдельно» — другая на 0 Гц.
    assert [(by_key[key].excite_group, by_key[key].excite_hz, by_key[key].excite2_group, by_key[key].excite2_hz)
            for key in ("sugar_40hz_water_0hz", "sugar_0hz_water_40hz", "sugar_40hz_water_40hz")] == [
        ("sugar", 40, "water", 0), ("sugar", 0, "water", 40), ("sugar", 40, "water", 40)]
    mixed = by_key["sugar_100hz_bitter_200hz"]
    assert (mixed.excite_group, mixed.excite_hz, mixed.excite2_group, mixed.excite2_hz) == ("sugar", 100, "bitter", 200)
    assert by_key["sugar_100hz_shuffled"].connectivity == "shuffled"
    assert sum(condition.connectivity == "shuffled" for condition in config["conditions"]) == 1
    # Всё, что задано нашим выбором, а не статьёй, подписано (CLAUDE.md, правило честности).
    assert by_key["sugar_100hz_shuffled"].honesty_ru and by_key["shadow_left_100hz"].honesty_ru


@pytest.mark.skipif(not (PROCESSED_DIR / processed.GROUPS_FILE).exists(), reason="нет data/processed")
def test_config_groups_exist():
    config = load_reference_config()
    groups = processed.load_groups()
    for condition in config["conditions"]:
        assert condition.excite_group in groups["stimuli"]
        assert condition.excite2_group is None or condition.excite2_group in groups["stimuli"]
    assert all(key in groups["readouts"] for key in config["summary_readouts"])


def test_shuffle_keeps_degrees_and_weights():
    rng = np.random.default_rng(0)
    connectivity = pd.DataFrame({
        "Presynaptic_Index": rng.integers(0, 50, 2000),
        "Postsynaptic_Index": rng.integers(0, 50, 2000),
        "Excitatory x Connectivity": rng.integers(-20, 20, 2000),
    })
    shuffled = shuffle_connectivity(connectivity, seed=783)
    # Отправители и веса на месте: у каждого нейрона те же выходы с теми же весами.
    pd.testing.assert_series_equal(shuffled["Presynaptic_Index"], connectivity["Presynaptic_Index"])
    pd.testing.assert_series_equal(shuffled["Excitatory x Connectivity"], connectivity["Excitatory x Connectivity"])
    # Число входов у каждого нейрона то же, а сами получатели другие.
    assert (np.bincount(shuffled["Postsynaptic_Index"], minlength=50)
            == np.bincount(connectivity["Postsynaptic_Index"], minlength=50)).all()
    assert (shuffled["Postsynaptic_Index"] != connectivity["Postsynaptic_Index"]).mean() > 0.9
    # Одно и то же зерно — один и тот же контроль; исходная таблица не тронута.
    pd.testing.assert_frame_equal(shuffle_connectivity(connectivity, seed=783), shuffled)
    assert not shuffle_connectivity(connectivity, seed=1).equals(shuffled)


def test_trial_seeds_stable_and_distinct():
    keys = sorted(PLAN_CONDITIONS)
    seeds = {trial_seed(key, trial) for key in keys for trial in range(30)}
    assert len(seeds) == len(keys) * 30
    assert trial_seed("sugar_100hz", 0) == trial_seed("sugar_100hz", 0)


def test_condition_rates_mean_over_trials():
    trials = [{"neuron": np.array([0, 0, 2])}, {"neuron": np.array([0, 2, 2, 2])}]
    rates = condition_rates(trials, n_neurons=4, duration_s=0.5)
    # Нейрон 0: 3 спайка за 2 повтора по 0,5 с = 3 Гц; нейрон 2: 4 спайка = 4 Гц.
    np.testing.assert_allclose(rates, [3.0, 0.0, 4.0, 0.0])
