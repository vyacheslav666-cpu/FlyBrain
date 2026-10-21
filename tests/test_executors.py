"""Тесты панели «исполнителей» (C11, D-035): список нейронов, ожидания по статье, проверка описания."""

import copy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from flybrain import executors, processed, stimuli
from flybrain.groups import load_group_config
from flybrain.paths import PROCESSED_DIR
from flybrain.server import _paper_root_ids, create_app

NEURONS = pd.DataFrame({
    "cell_type": ["CB0701", "CB0701", "DNp01", "DNp01", "DNg62", "XYZ", None],
    "side": ["left", "right", "left", "right", "right", "center", None],
})
NAMES = {"CB0701": ("MN9", "мотонейрон хоботка"), "DNp01": ("GF", "гигантское волокно"), "DNg62": ("aDN1", "груминг")}
SIDES = {"left": "левая", "right": "правая", "center": "по центру", "missing": "без аннотации"}


def label(cell_type):
    if cell_type is None:
        return "без типа", None
    return NAMES.get(cell_type, (cell_type, None))


def _readout(indices):
    return {"count": len(indices), "indices": indices, "root_ids": [str(1000 + i) for i in indices],
            "honesty_ru": "выбран нами"}


GROUPS = {"readouts": {
    "motor_all": _readout([0, 1]),
    "descending_all": _readout([2, 3, 4, 5, 6]),
    "MN9": _readout([0, 1]),
    "GF": _readout([2, 3]),
    "aDN1": _readout([4]),
}}
PAPER = {"aDN1": {"1004"}, "MN9": {"1001"}}


def _config(**entry):
    base = {"needs": ["sugar"], "readout": "MN9", "which": "all", "source": "paper", "test": "active", "why_ru": "почему"}
    return {"expected": [{**base, **entry}], "caveats": []}


def _expected(config):
    return executors.build_expected(GROUPS, NEURONS, label, PAPER, config, {"sugar", "bitter", "shadow"}, SIDES)


def test_list_has_motor_first_then_descending_with_names_and_sides():
    rows = executors.build_list(GROUPS, NEURONS, label)
    assert [row["i"] for row in rows] == [0, 1, 4, 2, 3, 5, 6]  # моторные, потом нисходящие по имени
    assert [row["class"] for row in rows] == ["motor"] * 2 + ["descending"] * 5
    mn9 = rows[0]
    assert (mn9["name"], mn9["type"], mn9["about"], mn9["side"]) == ("MN9", "CB0701", "мотонейрон хоботка", "left")
    assert "type" not in rows[5] and rows[5]["name"] == "XYZ"  # тип совпадает с именем — не повторяем
    assert rows[6]["name"] == "без типа" and rows[6]["side"] == "unknown"


def test_which_all_takes_both_sides_and_which_paper_only_the_paper_neuron():
    both = _expected(_config(readout="MN9", which="all"))["entries"][0]
    assert [(n["i"], n["side"]) for n in both["neurons"]] == [(0, "left"), (1, "right")]
    assert both["neurons"][0]["side_name"] == "левый"
    paper = _expected(_config(readout="MN9", which="paper"))["entries"][0]
    assert [n["i"] for n in paper["neurons"]] == [1]


def test_own_choice_is_marked_and_carries_the_honesty_text():
    own = _expected(_config(readout="GF", source="own", needs=["shadow"]))["entries"][0]
    assert own["source"] == "own" and own["honesty"] == "выбран нами" and own["name"] == "GF"
    paper = _expected(_config())["entries"][0]
    assert paper["honesty"] == ""  # у статьи пометка [ЧЕСТНОСТЬ] не нужна


def test_bad_description_is_an_error_not_a_silent_gap():
    for bad in ({"readout": "нет"}, {"needs": ["нет"]}, {"needs": []}, {"which": "x"}, {"test": "x"}, {"source": "x"},
                {"why_ru": ""}):
        with pytest.raises(executors.ExecutorsError):
            _expected(_config(**bad))
    with pytest.raises(executors.ExecutorsError):  # нейрона из статьи в данных нет
        executors.build_expected(GROUPS, NEURONS, label, {"MN9": set()}, _config(which="paper"), {"sugar"}, SIDES)
    config = _config()
    config["caveats"] = [{"stimulus": "нет", "text_ru": "x"}]
    with pytest.raises(executors.ExecutorsError):
        _expected(config)
    broken = copy.deepcopy(GROUPS)
    del broken["readouts"]["motor_all"]
    with pytest.raises(executors.ExecutorsError):
        executors.build_list(broken, NEURONS, label)


needs_data = pytest.mark.skipif(
    not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists(), reason="нет data/processed — запустите build-data"
)


@needs_data
def test_real_config_covers_the_paper_experiments_and_marks_the_giant_fiber_as_ours():
    groups, neurons = processed.load_groups(), processed.load_neurons()
    keys = {button["key"] for button in stimuli.load_buttons()}
    from flybrain import webdata
    names = webdata.load_yaml(webdata.NAMES_FILE)
    short = lambda cell_type: (names["cell_types"][cell_type]["short"], None) if cell_type in names["cell_types"] \
        else (str(cell_type), None)  # noqa: E731
    panel = executors.build_panel(groups, neurons, short, _paper_root_ids(load_group_config()),
                                  executors.load_config(), keys, names["sides"])
    assert [item["count"] for item in panel["classes"]] == [110, 1299]
    assert len(panel["neurons"]) == 1409 and len({row["i"] for row in panel["neurons"]}) == 1409
    assert panel["active_hz"] == 1.0  # порог D-018
    by_name = {}
    for entry in panel["expected"]["entries"]:
        by_name.setdefault((tuple(entry["needs"]), entry["name"]), entry)
    sugar_mn9 = by_name[(("sugar",), "MN9")]
    assert [n["side"] for n in sugar_mn9["neurons"]] == ["left", "right"] and sugar_mn9["test"] == "active"
    # Б5: только нейроны, номера которых записаны у Shiu — по одному
    for name in ("aDN1", "aDN2", "aBN1"):
        assert len(by_name[(("antenna",), name)]["neurons"]) == 1
    assert by_name[(("sugar", "bitter"), "MN9")]["test"] == "weaker"  # порог Б3 нужен ответ на один сахар
    gf = by_name[(("shadow",), "GF")]
    assert gf["source"] == "own" and "не из статьи" in gf["honesty"].lower()
    listed = {row["i"] for row in panel["neurons"]}
    assert {n["i"] for n in sugar_mn9["neurons"]} <= listed  # MN9 есть и в списке моторных


@needs_data
def test_server_gives_the_executor_panel():
    with TestClient(create_app()) as client:
        answer = client.get("/api/executors")
        assert answer.status_code == 200 and answer.headers["cache-control"] == "no-cache"
        panel = answer.json()
        assert len(panel["neurons"]) == 1409 and panel["expected"]["entries"]
        assert client.get("/web/executors.js").status_code == 200
