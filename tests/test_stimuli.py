"""Тесты панели стимулов (C10, D-034): каталог кнопок, стороны, честные описания."""

import json

import numpy as np
import pytest

from flybrain import processed, stimuli
from flybrain.docpage import DOCS_DIR, VALIDATION_SUMMARY
from flybrain.paths import PROCESSED_DIR


def _group(indices, paper="figures.ipynb: neu_x", total=3, missing=("1",), honesty=None):
    return {"name_ru": "Сахар (левая сторона)", "about_ru": "тест (список Shiu)", "honesty_ru": honesty,
            "paper": paper, "paper_total": total, "paper_missing_root_ids": list(missing) if paper else [],
            "count": len(indices), "indices": list(indices)}


def _groups():
    return {"stimuli": {"a_left": _group([1, 2]), "a_right": _group([5]), "own": _group([7], paper=None, total=0)}}


def test_two_sides_are_a_choice_and_one_side_is_locked():
    buttons = [
        {"key": "a", "sides": {"left": "a_left", "right": "a_right"}, "side_note_ru": "списки разной длины"},
        {"key": "b", "sides": {"left": "a_left"}, "side_lock_ru": "в статье список только для одной стороны"},
    ]
    catalog, indices = stimuli.build_catalog(_groups(), buttons)
    a, b = catalog["buttons"]
    assert not a["side_locked"] and [s["side"] for s in a["sides"]] == ["left", "right"]
    assert b["side_locked"] and [s["side"] for s in b["sides"]] == ["left"]
    assert b["side_note_ru"] == "в статье список только для одной стороны"
    assert indices["a"]["right"].tolist() == [5] and indices["b"]["left"].tolist() == [1, 2]
    assert a["default_hz"] == 100.0 and a["max_hz"] == 200.0 and a["default_side"] == "left"


def test_info_is_honest_about_source_and_counts():
    catalog, _ = stimuli.build_catalog(
        _groups(), [{"key": "a", "sides": {"left": "a_left"}, "side_lock_ru": "x"},
                    {"key": "own", "experiment": True, "sides": {"left": "own"}, "side_lock_ru": "y"}])
    paper_info = " ".join(catalog["buttons"][0]["sides"][0]["info"])
    assert "neu_x" in paper_info and "3 нейронов" in paper_info and "найдено 2" in paper_info
    assert "не найдены и не заменены" in paper_info and "прямое включение нейронов" in paper_info
    own = catalog["buttons"][1]
    assert own["experiment"] and "вне статьи" in " ".join(own["sides"][0]["info"])


def _mirror_group(twins):
    found = {"cell_type": "LB1a", "listed": [1, 2], "left_total": 2, "right_total": 2, "twins": list(twins),
             "reason": "" if twins else "part_of_type"}
    lost = {"cell_type": "LB1d", "listed": [3], "left_total": 3, "right_total": 3, "twins": [],
            "reason": "part_of_type"}
    group = _group(twins, paper=None, total=0, honesty="Не список статьи, а наш перенос (D-047).")
    return group | {"mirror_of": "a_left", "mirror_types": [found, lost]}


def test_mirror_side_says_where_twins_came_from_and_who_has_none():
    groups = _groups()
    groups["stimuli"]["a_mirror"] = _mirror_group([8, 9])
    catalog, indices = stimuli.build_catalog(
        groups, [{"key": "a", "sides": {"left": "a_left", "right": "a_mirror"}, "side_note_ru": "двойники"}])
    info = " ".join(catalog["buttons"][0]["sides"][1]["info"])
    assert "не список статьи" in info and "D-047" in info and "для 2 из 3 нейронов" in info
    assert "LB1a (2 слева — 2 справа)" in info and "LB1d — 1" in info and "[ЧЕСТНОСТЬ]" in info
    assert indices["a"]["right"].tolist() == [8, 9]


def test_an_empty_side_is_an_error():
    groups = _groups()
    groups["stimuli"]["a_mirror"] = _mirror_group([])  # двойников не нашлось — кнопку на эту сторону не ставим
    with pytest.raises(stimuli.StimulusError, match="ни одного нейрона"):
        stimuli.build_catalog(groups, [{"key": "a", "sides": {"left": "a_left", "right": "a_mirror"}}])


def test_a_second_side_without_a_group_is_an_error():
    with pytest.raises(stimuli.StimulusError):
        stimuli.build_catalog(_groups(), [{"key": "a", "sides": {"left": "a_left", "right": "nope"}}])
    with pytest.raises(stimuli.StimulusError):  # одна сторона без пояснения, почему выбор отключён
        stimuli.build_catalog(_groups(), [{"key": "a", "sides": {"left": "a_left"}}])
    with pytest.raises(stimuli.StimulusError):  # сторона, которой нет
        stimuli.build_catalog(_groups(), [{"key": "a", "sides": {"left": "a_left", "up": "a_right"}}])


needs_data = pytest.mark.skipif(
    not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists(), reason="нет data/processed — запустите build-data"
)


@needs_data
def test_real_panel_follows_the_paper_lists_for_sides():
    """Сторона выбирается только там, где в статье есть списки для обеих; остальное отключено (D-034)."""
    neurons = processed.load_neurons()
    catalog, indices = stimuli.build_catalog(processed.load_groups(), stimuli.load_buttons())
    by_key = {button["key"]: button for button in catalog["buttons"]}
    assert list(by_key) == ["sugar", "bitter", "antenna", "shadow"]
    assert {key: not button["side_locked"] for key, button in by_key.items()} == {
        "sugar": True, "bitter": False, "antenna": False, "shadow": True}
    assert by_key["shadow"]["experiment"] and not by_key["sugar"]["experiment"]
    for key, button in by_key.items():
        for side in button["sides"]:  # сторона кнопки совпадает со стороной нейронов по таблице аннотаций (D-011)
            sides_of_neurons = set(neurons["side"].iloc[indices[key][side["side"]]])
            assert sides_of_neurons == {side["side"]}, (key, side["side"])
    assert [len(indices["sugar"][side]) for side in ("left", "right")] == [20, 9]
    # Горечь и усик — только левая: списка для правой в статье нет, а правые двойники проверку не прошли (D-047).
    assert set(indices["bitter"]) == {"left"} and set(indices["antenna"]) == {"left"}
    assert "D-047" in by_key["bitter"]["side_note_ru"] and "D-047" in by_key["antenna"]["side_note_ru"]
    assert by_key["bitter"]["side_lock_short_ru"] and by_key["antenna"]["side_lock_short_ru"]
    assert set(np.concatenate(list(indices["sugar"].values()))).isdisjoint(indices["bitter"]["left"])


@needs_data
def test_twins_go_on_a_button_only_after_the_check_passed():
    """Правые двойники (mirror_of) — на кнопке, только если их пропустил раздел Е проверки (D-047)."""
    groups = processed.load_groups()
    with open(DOCS_DIR / VALIDATION_SUMMARY, encoding="utf-8") as handle:
        checked = {item["key"]: item for item in json.load(handle)["mirror"]}
    assert set(checked) == {"bitter", "antenna"}
    for button in stimuli.load_buttons():
        for side, key in button["sides"].items():
            mirror_of = groups["stimuli"][key].get("mirror_of")
            if mirror_of:
                assert checked[mirror_of]["include"], f"{button['key']}: двойники «{key}» не прошли проверку"
    assert checked["antenna"]["twins"] == 0 and not checked["antenna"]["include"]
