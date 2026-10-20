"""Подготовленные данные в data/processed/: имена файлов и чтение (docs/DATA.md, раздел 5)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.paths import PROCESSED_DIR

# Меняется, когда меняется формат файлов; тогда нужно заново выполнить `flybrain build-data`.
FORMAT_VERSION = 1

NEURONS_FILE = "neurons.parquet"
CONNECTIVITY_FILE = "connectivity.npz"
GROUPS_FILE = "groups.json"
PREVIEW_FILE = "preview.png"
BUILD_INFO_FILE = "build_info.json"


def load_neurons(directory: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Таблица нейронов; номер строки равен внутреннему номеру нейрона (столбец `index`)."""
    return pd.read_parquet(directory / NEURONS_FILE)


def load_connectivity(directory: Path = PROCESSED_DIR) -> dict[str, np.ndarray]:
    """Матрица связей CSR: строка — получатель, `indices` — отправители, `weight` — синапсы со знаком."""
    with np.load(directory / CONNECTIVITY_FILE) as arrays:
        return {name: arrays[name] for name in arrays.files}


def load_groups(directory: Path = PROCESSED_DIR) -> dict:
    with open(directory / GROUPS_FILE, encoding="utf-8") as handle:
        return json.load(handle)
