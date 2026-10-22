"""Серотониновые нейроны и их мишени — для эксперимента P1 «психоделик» (D-053, D-054, D-055).

Три способа назвать нейрон серотониновым (сравнение — docs/EXPERIMENT_PSY.md, раздел 2):
- **строгий** (основной, D-054): в столбце `known_nt` таблицы аннотаций FlyWire серотонин указан со ссылкой на
  окраску антителами (`immuno`) или на одиночные клоны из линий-драйверов серотониновых нейронов (`MCFO`);
- **широкий** (проверка устойчивости): серотонин указан в `known_nt` с любой ссылкой, в том числе только
  по РНК (EASI-FISH, scRNA-seq);
- **по предсказанию** (только для сравнения, в опыте не используется): `top_nt` = serotonin — медиатор,
  предсказанный нейросетью по снимкам (Eckstein et al. 2024).

`known_nt` и `known_nt_source` разбиты точкой с запятой на части, и части идут в одном порядке: k-я часть
медиаторов взята из k-го источника. Способ — текст в скобках в конце источника. Отрицания («serotonin-negative»)
серотонином не считаются.

Мишень — нейрон, получающий синапсы от серотониновых нейронов; s_i — их число (по модулю, как в файле связей
Shiu). Действие дозы d: порог ниже на d · ΔV · s_i / (s_i + s_half) (D-055).
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.fetch import sha256_of
from flybrain.paths import RAW_DIR

ANNOTATIONS_FILE = RAW_DIR / "flywire" / "Supplemental_file1_neuron_annotations.tsv"
ANNOTATIONS_NAME = "Supplemental_file1_neuron_annotations.tsv"
SEROTONIN = "serotonin"
STRICT_METHODS = ("immuno", "mcfo")  # окраска антителами; клоны из линий-драйверов серотониновых нейронов
CRITERIA = ("strict", "broad", "predicted")
CRITERIA_RU = {
    "strict": "строгий: окраска или клоны из серотониновых линий",
    "broad": "широкий: любая ссылка в литературе",
    "predicted": "по предсказанию медиатора",
}


class SerotoninError(Exception):
    """Таблица аннотаций не та или не сходится с таблицей нейронов."""


def serotonin_methods(known_nt: str | None, source: str | None) -> list[str]:
    """Способы (в нижнем регистре), которыми подтверждён серотонин; пусто — серотонин не указан.

    >>> serotonin_methods("natalisin; serotonin, MIP", "Jiang 2013 (immuno); Scheunemann 2018 (immuno)")
    ['immuno']
    """
    if not isinstance(known_nt, str) or not known_nt:
        return []
    sources = source.split(";") if isinstance(source, str) else []
    methods = []
    for k, part in enumerate(known_nt.split(";")):
        if SEROTONIN not in (token.strip() for token in part.split(",")):
            continue
        cited = sources[k].strip() if k < len(sources) else ""
        found = re.search(r"\(([^()]*)\)\s*$", cited)
        methods.append(found.group(1).strip().lower() if found else "не указан")
    return methods


def read_annotations(path: Path = ANNOTATIONS_FILE, expected_sha256: str | None = None) -> pd.DataFrame:
    """Столбцы медиаторов из таблицы аннотаций FlyWire (пустая ячейка — пропуск, как в `build-data`)."""
    if not path.exists():
        raise SerotoninError("нет таблицы аннотаций. Выполните: uv run flybrain fetch-data")
    if expected_sha256 and sha256_of(path) != expected_sha256:
        raise SerotoninError("таблица аннотаций не та, из которой собраны data/processed. "
                             "Выполните: uv run flybrain fetch-data и uv run flybrain build-data")
    return pd.read_csv(path, sep="\t", usecols=["root_id", "top_nt", "top_nt_conf", "known_nt", "known_nt_source"],
                       dtype={"root_id": "int64", "top_nt": "str", "top_nt_conf": "float64", "known_nt": "str",
                              "known_nt_source": "str"},
                       keep_default_na=False, na_values=[""])


@dataclass
class SerotoninSets:
    """Серотониновые нейроны модели по трём критериям: булевы массивы длиной N (номер — внутренний)."""
    masks: dict[str, np.ndarray]
    methods: list[list[str]]  # способы подтверждения серотонина у каждого нейрона (пусто — нет в known_nt)
    top_nt_conf: np.ndarray  # уверенность предсказания медиатора
    known_nt: np.ndarray  # медиатор из литературы как в таблице (строка или None)

    def indices(self, criterion: str) -> np.ndarray:
        return np.flatnonzero(self.masks[criterion])


def serotonin_sets(neurons: pd.DataFrame, annotations: pd.DataFrame) -> SerotoninSets:
    table = annotations.drop_duplicates("root_id").set_index("root_id").reindex(neurons["root_id"].to_numpy())
    methods = [serotonin_methods(nt, source) for nt, source in zip(table["known_nt"], table["known_nt_source"])]
    broad = np.array([bool(found) for found in methods])
    strict = np.array([any(method in STRICT_METHODS for method in found) for found in methods])
    predicted = (neurons["nt"] == SEROTONIN).to_numpy()
    if not np.array_equal(table["top_nt"].fillna("").to_numpy(), neurons["nt"].fillna("").to_numpy()):
        raise SerotoninError("медиатор в таблице аннотаций и в data/processed не совпадает. "
                             "Выполните: uv run flybrain build-data")
    known = np.array([value if isinstance(value, str) else None for value in table["known_nt"]], dtype=object)
    return SerotoninSets({"strict": strict, "broad": broad, "predicted": predicted}, methods,
                         table["top_nt_conf"].to_numpy(dtype=np.float64), known)


def mentions(known_nt: str | None, transmitter: str) -> bool:
    """Упомянут ли медиатор в `known_nt` как положительный (не «…-negative»)."""
    if not isinstance(known_nt, str):
        return False
    return any(token.strip() == transmitter for token in re.split(r"[;,]", known_nt))


def serotonin_synapses(indptr: np.ndarray, indices: np.ndarray, weight: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """s_i — сколько синапсов нейрон i получает от нейронов `sources` (булев массив). CSR: строка — получатель."""
    n = len(indptr) - 1
    receiver = np.repeat(np.arange(n), np.diff(indptr))
    from_source = sources[indices]
    return np.bincount(receiver[from_source], weights=np.abs(weight[from_source].astype(np.int64)),
                       minlength=n).astype(np.int64)


def saturation(s: np.ndarray, s_half: float) -> np.ndarray:
    """h(s) = s / (s + s_half): 0 без серотониновых входов, 1/2 при s = s_half, стремится к 1."""
    s = np.asarray(s, dtype=np.float64)
    return s / (s + float(s_half))


def threshold_shift(s: np.ndarray, dose: float, delta_v_mv: float, s_half: float) -> np.ndarray:
    """Насколько снижен порог каждого нейрона, мВ: d · ΔV · h(s_i). При d = 0 — все нули."""
    if dose < 0:
        raise ValueError("доза не может быть отрицательной")
    return float(dose) * float(delta_v_mv) * saturation(s, s_half)
