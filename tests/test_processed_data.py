"""Проверки настоящих данных в data/processed/. Пропускаются, если `flybrain build-data` ещё не запускали."""

import numpy as np
import pytest
import yaml

from flybrain import processed
from flybrain.paths import CONFIG_DIR, PROCESSED_DIR

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists(), reason="нет data/processed — запустите build-data"
)


@pytest.fixture(scope="module")
def neurons():
    return processed.load_neurons()


def test_counts_match_shiu_and_flywire(neurons):
    csr = processed.load_connectivity()
    assert len(neurons) == 138_639
    assert neurons["super_class"].notna().sum() == 138_625
    assert len(csr["indices"]) == 15_091_983
    assert np.abs(csr["weight"].astype(np.int64)).sum() == 54_492_922
    assert (csr["weight"] > 0).sum() == 9_059_302
    assert csr["indptr"][-1] == len(csr["indices"])


def test_mirrored_x_puts_fly_left_on_larger_x(neurons):
    # D-011: после отражения X «левые» по таблице нейроны лежат при больших x_um, чем «правые».
    left = neurons.loc[neurons["side"] == "left", "x_um"].mean()
    right = neurons.loc[neurons["side"] == "right", "x_um"].mean()
    assert left > right


def test_groups_as_in_plan(neurons):
    groups = processed.load_groups()
    stimuli, readouts = groups["stimuli"], groups["readouts"]
    assert stimuli["sugar"]["count"] == 20 and stimuli["bitter"]["count"] == 20
    assert stimuli["antenna"]["count"] == 145
    assert stimuli["shadow_left"]["count"] == 162 and stimuli["shadow_right"]["count"] == 152
    assert set(neurons["side"].iloc[stimuli["sugar"]["indices"]]) == {"left"}
    assert sorted(readouts["MN9"]["root_ids"]) == ["720575940618238523", "720575940660219265"]
    assert readouts["motor_all"]["count"] == 110 and readouts["descending_all"]["count"] == 1299


def test_names_cover_all_classes(neurons):
    with open(CONFIG_DIR / "names_ru.yaml", encoding="utf-8") as handle:
        names = yaml.safe_load(handle)
    assert set(neurons["super_class"].dropna()) <= set(names["super_classes"])
    assert set(neurons["side"].dropna()) <= set(names["sides"])
    assert set(neurons["nt"].dropna()) <= set(names["neurotransmitters"])
