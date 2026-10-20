"""Данные для браузера: облако нейронов в двоичном виде и описание классов для легенды (коммит C7)."""

import gzip
import struct

import numpy as np
import pandas as pd
import yaml

from flybrain.paths import CONFIG_DIR

LEGEND_FILE = CONFIG_DIR / "legend.yaml"
NAMES_FILE = CONFIG_DIR / "names_ru.yaml"

CLOUD_FORMAT_VERSION = 1
NO_POSITION = 255  # код класса у нейрона без координат: браузер его не рисует
MAX_CLASSES = 16  # столько цветов помещается в шейдер страницы (web/main.js)


def load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_legend(path=LEGEND_FILE) -> list[dict]:
    classes = load_yaml(path)["classes"]
    if len(classes) > MAX_CLASSES:
        raise ValueError(f"в легенде {len(classes)} классов, а страница умеет не больше {MAX_CLASSES}")
    return classes


def class_codes(neurons: pd.DataFrame, legend: list[dict]) -> np.ndarray:
    """Номер класса в легенде для каждого нейрона; без координат — `NO_POSITION`."""
    code_of = {entry["key"]: code for code, entry in enumerate(legend)}
    has_position = neurons[["x_um", "y_um", "z_um"]].notna().all(axis=1).to_numpy()
    codes = np.full(len(neurons), NO_POSITION, dtype=np.uint8)
    unknown = set()
    for row, (super_class, shown) in enumerate(zip(neurons["super_class"], has_position)):
        if not shown:
            continue
        if super_class not in code_of:
            unknown.add(super_class)
            continue
        codes[row] = code_of[super_class]
    if unknown:
        raise ValueError(f"в config/legend.yaml нет классов: {sorted(map(str, unknown))}")
    return codes


def build_cloud(neurons: pd.DataFrame, legend: list[dict]) -> bytes:
    """Двоичное облако: заголовок (uint32 n, uint32 версия), float32 x,y,z на нейрон, uint8 класс на нейрон.

    Порядок нейронов — внутренний номер, поэтому номер точки в браузере равен номеру нейрона.
    Положения — в микрометрах, как в таблице нейронов (D-025); у нейронов без координат — нули.
    """
    codes = class_codes(neurons, legend)
    xyz = neurons[["x_um", "y_um", "z_um"]].to_numpy(dtype="<f4", na_value=0.0)
    return struct.pack("<II", len(neurons), CLOUD_FORMAT_VERSION) + xyz.tobytes() + codes.tobytes()


def build_cloud_gzip(neurons: pd.DataFrame, legend: list[dict]) -> bytes:
    return gzip.compress(build_cloud(neurons, legend), compresslevel=6)


def build_meta(neurons: pd.DataFrame, legend: list[dict], names: dict) -> dict:
    """Описание для страницы: классы (цвет, русское имя, сколько точек), размеры облака, пометки."""
    codes = class_codes(neurons, legend)
    shown = codes != NO_POSITION
    points = neurons.loc[shown, ["x_um", "y_um", "z_um"]].to_numpy()
    low, high = points.min(axis=0), points.max(axis=0)
    classes = [
        {
            "code": code,
            "key": entry["key"],
            "name": names["super_classes"][entry["key"]],
            "color": entry["color"],
            "count": int((codes == code).sum()),
        }
        for code, entry in enumerate(legend)
    ]
    sources = neurons.loc[shown, "pos_source"].value_counts()
    return {
        "format": CLOUD_FORMAT_VERSION,
        "total": len(neurons),
        "shown": int(shown.sum()),
        "hidden": int((~shown).sum()),
        "soma": int(sources.get("soma", 0)),
        "anchor": int(sources.get("anchor", 0)),
        "center": [float(v) for v in (low + high) / 2],
        "radius": float(np.linalg.norm(high - low) / 2),
        "classes": classes,
    }
