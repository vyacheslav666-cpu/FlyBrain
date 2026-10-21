"""Тесты панели стимулов (C10, D-034): каталог кнопок, стороны, честные описания."""

import numpy as np
import pytest

from flybrain import processed, stimuli
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
    # Горечь и усик — только левая: списка для правой в статье нет, мы его не придумываем.
    assert set(indices["bitter"]) == {"left"} and set(indices["antenna"]) == {"left"}
    assert set(np.concatenate(list(indices["sugar"].values()))).isdisjoint(indices["bitter"]["left"])
