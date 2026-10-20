import json
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

from flybrain.groups import SECTIONS, STATUSES, GroupError, load_group_config, read_data_pickle, \
    read_notebook_lists, resolve_groups
from flybrain.paths import CONFIG_DIR

A, B, C, D = 720575940600000001, 720575940600000002, 720575940600000003, 720575940600000004


def _neurons() -> pd.DataFrame:
    return pd.DataFrame({
        "root_id": [A, B, C],
        "cell_type": ["LB3c", "LB3c", "CB0701"],
        "side": ["left", "left", "right"],
        "super_class": ["sensory", "sensory", "motor"],
        "cell_class": [None, None, None],
        "cell_sub_class": [None, None, None],
    })


def _paper(**lists) -> dict:
    return {"figures.ipynb": {name: [ids] for name, ids in lists.items()}}


def _config(entries, **extra) -> dict:
    group = {"name_ru": "Сахар", "about_ru": "тест", "paper": {"file": "figures.ipynb", "name": "neu_sugar"},
             "paper_ids": entries, **extra}
    return {"stimuli": {"sugar": group}}


def test_paper_group_uses_only_ids_found_in_783():
    config = _config([{"id": A, "status": "in_783"}, {"id": D, "status": "not_in_783"},
                      {"id": B, "status": "in_783"}])
    groups = resolve_groups(config, _neurons(), _paper(neu_sugar=[A, D, B]))
    sugar = groups["stimuli"]["sugar"]
    assert sugar.indices.tolist() == [0, 1]
    assert sugar.paper_total == 3 and sugar.paper_missing == [D]


def test_wrong_status_is_an_error():
    config = _config([{"id": A, "status": "in_783"}, {"id": D, "status": "in_783"}])
    with pytest.raises(GroupError, match="не найден в 783"):
        resolve_groups(config, _neurons(), _paper(neu_sugar=[A, D]))


def test_list_must_match_paper_exactly():
    config = _config([{"id": A, "status": "in_783"}])
    with pytest.raises(GroupError, match="не совпадает со статьёй"):
        resolve_groups(config, _neurons(), _paper(neu_sugar=[A, B]))


def test_select_must_contain_paper_ids_and_union_merges():
    config = {"readouts": {
        "MN9": {"name_ru": "MN9", "about_ru": "т", "select": {"cell_type": "CB0701"},
                "paper": {"file": "figures.ipynb", "name": "ids_mn9"},
                "paper_ids": [{"id": C, "status": "in_783"}, {"id": D, "status": "not_in_783"}]},
        "wrong": {"name_ru": "т", "about_ru": "т", "select": {"cell_type": "LB3c"},
                  "paper": {"file": "figures.ipynb", "name": "ids_mn9"},
                  "paper_ids": [{"id": C, "status": "in_783"}, {"id": D, "status": "not_in_783"}]},
    }}
    with pytest.raises(GroupError, match="не попали в отбор"):
        resolve_groups(config, _neurons(), _paper(ids_mn9=[C, D]))
    del config["readouts"]["wrong"]
    config["readouts"]["left_sensory"] = {"name_ru": "т", "about_ru": "т", "honesty_ru": "наш выбор",
                                         "select": {"super_class": "sensory", "side": ["left"]}}
    config["readouts"]["both"] = {"name_ru": "т", "about_ru": "т", "honesty_ru": "наш выбор",
                                 "union": ["MN9", "left_sensory"]}
    groups = resolve_groups(config, _neurons(), _paper(ids_mn9=[C, D]))["readouts"]
    assert groups["MN9"].indices.tolist() == [2]
    assert groups["left_sensory"].indices.tolist() == [0, 1]
    assert groups["both"].indices.tolist() == [0, 1, 2]


def test_group_not_from_paper_needs_honesty_note():
    config = {"stimuli": {"shadow": {"name_ru": "т", "about_ru": "т", "select": {"cell_type": "LB3c"}}}}
    with pytest.raises(GroupError, match="honesty_ru"):
        resolve_groups(config, _neurons(), {})


def test_notebook_lists_are_read_without_running_code(tmp_path):
    cells = [
        {"cell_type": "code", "source": ["import os\n", f"neu_sugar = [\n    {A}, {B},\n]\n", "freqs = [10, 20]\n"]},
        {"cell_type": "markdown", "source": ["# рисунок"]},
        {"cell_type": "code", "source": [f"id_mn9 = {C}\n", "os.remove('x')\n"]},
    ]
    path = tmp_path / "figures.ipynb"
    path.write_text(json.dumps({"cells": cells}), encoding="utf-8")
    assert read_notebook_lists(path) == {"neu_sugar": [[A, B]], "id_mn9": [[C]]}


def test_data_pickle_refuses_code(tmp_path):
    good, bad = tmp_path / "good.pickle", tmp_path / "bad.pickle"
    good.write_bytes(pickle.dumps({"roundup": [A, B]}))
    bad.write_bytes(pickle.dumps({"x": np.int64(1)}))  # ссылается на код numpy
    assert read_data_pickle(good) == {"roundup": [A, B]}
    with pytest.raises(pickle.UnpicklingError):
        read_data_pickle(bad)


def test_group_config_matches_decisions():
    config = load_group_config()
    assert set(config) == set(SECTIONS)
    # Длины списков статьи — из D-012 и D-014 (номера версии 630).
    expected = {"sugar": 21, "bitter": 21, "water": 18, "ir94e": 18, "jon_ce": 70, "jon_f": 60, "jon_dm": 16}
    for key, total in expected.items():
        entries = config["stimuli"][key]["paper_ids"]
        assert len(entries) == total, key
    for section in SECTIONS:
        for key, spec in config[section].items():
            assert spec["name_ru"] and spec["about_ru"], key
            assert "paper" in spec or spec.get("honesty_ru"), f"{key}: нет пометки [ЧЕСТНОСТЬ]"
            for entry in spec.get("paper_ids", []):
                assert entry["status"] in STATUSES


def test_names_cover_readout_types():
    with open(CONFIG_DIR / "names_ru.yaml", encoding="utf-8") as handle:
        names = yaml.safe_load(handle)
    for key, spec in load_group_config()["readouts"].items():
        cell_type = spec.get("select", {}).get("cell_type")
        if cell_type:
            assert cell_type in names["cell_types"], key
    assert {"left", "right", "center", "na", "missing"} <= set(names["sides"])
    assert set(names["signs"]) == {1, -1, 0}
