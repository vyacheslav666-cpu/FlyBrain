"""Панель «исполнителей» (C11, D-035): моторные и нисходящие нейроны и «ожидается по статье» для включённых стимулов.

Описание ожиданий — `config/executors.yaml`, группы нейронов — `data/processed/groups.json` (D-025).
"""

import pandas as pd
import yaml

from flybrain.paths import CONFIG_DIR
from flybrain.reference import ACTIVE_HZ

EXECUTORS_CONFIG = CONFIG_DIR / "executors.yaml"
LIST_GROUPS = {"motor": "motor_all", "descending": "descending_all"}
CLASS_NAMES = {"motor": "моторный", "descending": "нисходящий"}
SIDE_NAMES = {"left": "левый", "right": "правый", "center": "по центру"}
WHICH = ("all", "paper")
TESTS = ("active", "weaker")
SOURCES = ("paper", "own")


class ExecutorsError(Exception):
    """Описание панели не сходится с группами нейронов или с кнопками стимулов."""


def load_config(path=EXECUTORS_CONFIG) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _side(neurons: pd.DataFrame, index: int) -> str:
    side = neurons["side"].iat[index]
    return "unknown" if pd.isna(side) else str(side)


def _text(value) -> str | None:
    return None if pd.isna(value) else str(value)


def build_list(groups: dict, neurons: pd.DataFrame, label) -> list[dict]:
    """Все моторные и нисходящие нейроны: номер, имя, класс, сторона и, если есть, пояснение и тип FlyWire.

    `label(cell_type)` — как `NeuronLookup.label`: (подпись, пояснение). Порядок — сначала моторные, внутри по имени.
    """
    rows = []
    for class_key, group_key in LIST_GROUPS.items():
        if group_key not in groups["readouts"]:
            raise ExecutorsError(f"в groups.json нет группы «{group_key}»")
        for index in groups["readouts"][group_key]["indices"]:
            index = int(index)
            cell_type = _text(neurons["cell_type"].iat[index])
            name, about = label(cell_type)
            side = _side(neurons, index)
            row = {"i": index, "name": name, "class": class_key, "side": side}
            if about:
                row["about"] = about
            if cell_type and cell_type != name:  # у известных нейронов имя короткое («MN9»), тип FlyWire — «CB0701»
                row["type"] = cell_type
            rows.append(row)
    rows.sort(key=lambda row: (row["class"] != "motor", row["name"].lower(), row["side"], row["i"]))
    return rows


def _check_entry(entry: dict, stimulus_keys: set[str], readouts: dict) -> None:
    where = f"expected «{entry.get('readout')}»"
    if entry.get("readout") not in readouts:
        raise ExecutorsError(f"{where}: нет такой группы в разделе readouts")
    if not entry.get("needs") or not set(entry["needs"]) <= stimulus_keys:
        raise ExecutorsError(f"{where}: needs должен быть списком ключей кнопок стимулов, а не {entry.get('needs')}")
    for field, allowed in (("which", WHICH), ("test", TESTS), ("source", SOURCES)):
        if entry.get(field) not in allowed:
            raise ExecutorsError(f"{where}: {field} должен быть одним из {allowed}")
    if not entry.get("why_ru"):
        raise ExecutorsError(f"{where}: нужно why_ru — зачем этот нейрон ждём")


def build_expected(groups: dict, neurons: pd.DataFrame, label, paper_root_ids: dict[str, set[str]],
                   config: dict, stimulus_keys: set[str], side_names: dict) -> dict:
    """Ожидания для страницы: список записей (когда показывать, какие нейроны, как проверять) и пояснения."""
    entries = []
    for entry in config["expected"]:
        _check_entry(entry, stimulus_keys, groups["readouts"])
        group = groups["readouts"][entry["readout"]]
        pairs = list(zip(group["indices"], group["root_ids"]))
        if entry["which"] == "paper":
            pairs = [(index, root_id) for index, root_id in pairs
                     if str(root_id) in paper_root_ids.get(entry["readout"], set())]
            if not pairs:
                raise ExecutorsError(f"expected «{entry['readout']}»: в версии 783 нет нейрона из статьи")
        pairs.sort(key=lambda pair: _side(neurons, int(pair[0])))
        name, about = label(_text(neurons["cell_type"].iat[int(pairs[0][0])]))
        entries.append({
            "needs": list(entry["needs"]),
            "name": name,
            "about": about,
            "source": entry["source"],
            "test": entry["test"],
            "why": entry["why_ru"],
            "honesty": (group.get("honesty_ru") or "").strip() if entry["source"] == "own" else "",
            "neurons": [{"i": int(index), "side": side, "side_name": SIDE_NAMES.get(side, side_names.get(side, side))}
                        for index, side in ((int(index), _side(neurons, int(index))) for index, _ in pairs)],
        })
    caveats = []
    for caveat in config.get("caveats", []):
        if caveat["stimulus"] not in stimulus_keys:
            raise ExecutorsError(f"caveats: нет кнопки «{caveat['stimulus']}»")
        caveats.append({"stimulus": caveat["stimulus"], "side": caveat.get("side"), "alone": bool(caveat.get("alone")),
                        "text": caveat["text_ru"].strip()})
    return {"entries": entries, "caveats": caveats}


def build_panel(groups: dict, neurons: pd.DataFrame, label, paper_root_ids: dict[str, set[str]],
                config: dict, stimulus_keys: set[str], side_names: dict) -> dict:
    """Всё для панели «исполнителей»: список нейронов и ожидания (отдаётся страницей как JSON)."""
    return {
        "active_hz": ACTIVE_HZ,
        "classes": [{"key": key, "name": CLASS_NAMES[key], "count": groups["readouts"][group]["count"]}
                    for key, group in LIST_GROUPS.items()],
        "neurons": build_list(groups, neurons, label),
        "expected": build_expected(groups, neurons, label, paper_root_ids, config, stimulus_keys, side_names),
    }
