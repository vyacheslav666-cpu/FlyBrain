"""Кнопки-стимулы (C10, D-034): какие группы нейронов включает каждая кнопка и на какой стороне мухи.

Описание кнопок — `config/stimulus_panel.yaml`, группы нейронов — `data/processed/groups.json` (D-012…D-014, D-025).
"""

import numpy as np
import yaml

from flybrain.paths import CONFIG_DIR

PANEL_CONFIG = CONFIG_DIR / "stimulus_panel.yaml"
SIDE_NAMES = {"left": "левая сторона мухи", "right": "правая сторона мухи"}
DEFAULT_HZ = 100.0
MAX_HZ = 200.0
DEFAULT_SIDE = "left"


class StimulusError(Exception):
    """Описание кнопок не сходится с группами нейронов."""


def load_buttons(path=PANEL_CONFIG) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)["buttons"]


def _mirror_origin(group: dict) -> str:
    """Откуда правые двойники (D-047): для скольких нейронов левого списка нашлись и почему не для остальных."""
    found = [match for match in group["mirror_types"] if match["twins"]]
    lost = [match for match in group["mirror_types"] if not match["twins"]]
    paired = sum(len(match["listed"]) for match in found)
    listed = sum(len(match["listed"]) for match in group["mirror_types"])
    types = ", ".join(f"{match['cell_type']} ({len(match['listed'])} слева — {len(match['twins'])} справа)"
                      for match in found)
    text = (f"Это не список статьи [ЧЕСТНОСТЬ]: правые двойники нейронов левого списка статьи, найденные по таблице "
            f"аннотаций FlyWire 783 (тип клетки + сторона, правило D-047). Двойники нашлись для {paired} из {listed} "
            f"нейронов левого списка: {types}.")
    if lost:
        text += " Без двойника: " + "; ".join(
            f"{match['cell_type']} — {len(match['listed'])} (слева в списке {len(match['listed'])} из "
            f"{match['left_total']}, справа {match['right_total']})" for match in lost) + "."
    return text


def _side_info(group: dict) -> list[str]:
    """Честное описание группы для кнопки «ⓘ»: что включается, сколько, откуда список."""
    missing = len(group["paper_missing_root_ids"])
    lost = f"; остальные ({missing}) в 783 не найдены и не заменены «похожими»" if missing else ""
    if group.get("mirror_of"):
        origin = _mirror_origin(group)
    elif group["paper"]:
        file, name = group["paper"].split(": ")
        origin = (f"Список взят из статьи Shiu et al. 2024: {file}, список «{name}» — {group['paper_total']} "
                  f"нейронов, в версии данных 783 найдено {group['count']}{lost}.")
    elif group["paper_total"]:
        origin = (f"Список — объединение списков статьи Shiu et al. 2024: всего {group['paper_total']} нейронов, "
                  f"в версии данных 783 найдено {group['count']}{lost}.")
    else:
        origin = (f"Это опыт вне статьи: {group['count']} нейронов выбрали мы по таблице аннотаций FlyWire "
                  "[ЧЕСТНОСТЬ], а не авторы статьи.")
    about = group["about_ru"].strip()
    lines = [
        f"{about[0].upper()}{about[1:]}.",
        f"Сколько: {group['count']} нейронов. {origin}",
        "Как включается: каждый из этих нейронов получает случайные импульсы с выбранной частотой, как в статье. "
        "Это прямое включение нейронов, а не физический стимул (D-012…D-014).",
    ]
    if group["honesty_ru"]:
        lines.append(f"[ЧЕСТНОСТЬ] {group['honesty_ru'].strip()}")
    return lines


def build_catalog(groups: dict, buttons: list[dict]) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Описание панели для страницы (JSON) и номера нейронов: кнопка → сторона → номера."""
    stimuli = groups["stimuli"]
    catalog, indices = [], {}
    for button in buttons:
        sides = button["sides"]
        if not sides or DEFAULT_SIDE not in sides or set(sides) - set(SIDE_NAMES):
            raise StimulusError(f"кнопка «{button['key']}»: стороны должны быть left и/или right, left обязателен")
        missing = [key for key in sides.values() if key not in stimuli]
        if missing:
            raise StimulusError(f"кнопка «{button['key']}»: нет групп {missing} в groups.json")
        empty = [key for key in sides.values() if not stimuli[key]["count"]]
        if empty:  # например, правые двойники, которых не нашлось (D-047)
            raise StimulusError(f"кнопка «{button['key']}»: в группах {empty} нет ни одного нейрона")
        indices[button["key"]] = {side: np.asarray(stimuli[key]["indices"], dtype=np.int64)
                                  for side, key in sides.items()}
        locked = len(sides) == 1
        if locked != bool(button.get("side_lock_ru")):
            raise StimulusError(f"кнопка «{button['key']}»: одна сторона ⇔ задано side_lock_ru")
        entry = {
            "key": button["key"],
            "name_ru": stimuli[sides[DEFAULT_SIDE]]["name_ru"].split(" (")[0],
            "experiment": bool(button.get("experiment")),
            "default_hz": DEFAULT_HZ,
            "max_hz": MAX_HZ,
            "default_side": DEFAULT_SIDE,
            "side_locked": locked,
            "side_note_ru": (button.get("side_lock_ru") or button.get("side_note_ru") or "").strip(),
            "side_lock_short_ru": (button.get("side_lock_short_ru") or "").strip(),
            "sides": [
                {"side": side, "label": SIDE_NAMES[side], "group": key, "count": stimuli[key]["count"],
                 "info": _side_info(stimuli[key])}
                for side, key in sides.items()
            ],
        }
        catalog.append(entry)
    return {"buttons": catalog}, indices
