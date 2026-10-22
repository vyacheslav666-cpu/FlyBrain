"""Группы нейронов: стимулы и «исполнители» (config/neuron_groups.yaml, D-012…D-014, D-025)."""

import ast
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flybrain.mirror import TypeMatch, match_twins, twins_of
from flybrain.paths import CONFIG_DIR

GROUPS_CONFIG = CONFIG_DIR / "neuron_groups.yaml"
SECTIONS = ("stimuli", "readouts")
STATUSES = {
    "in_783": "из статьи, номер не менялся",
    "not_in_783": "не найден в 783",
}
# По каким столбцам таблицы нейронов разрешён отбор `select`.
SELECT_COLUMNS = ("cell_type", "side", "super_class", "cell_class", "cell_sub_class")
# Номера FlyWire — 18-значные; так их отличаем от прочих чисел в тетрадке Shiu.
_MIN_ROOT_ID = 10**17


class GroupError(Exception):
    """Описание группы не сходится со статьёй или с данными."""


@dataclass
class Group:
    key: str
    section: str
    name_ru: str
    about_ru: str
    honesty_ru: str | None
    source: str  # "paper", "select", "union" или "mirror"
    paper: str | None  # где список записан у Shiu, например «figures.ipynb: neu_sugar»
    indices: np.ndarray  # внутренние номера нейронов группы, по возрастанию
    paper_total: int = 0
    paper_missing: list[int] = field(default_factory=list)
    mirror_of: str | None = None  # для source == "mirror": чьи это правые двойники (D-047)
    mirror: list[TypeMatch] = field(default_factory=list)


def load_group_config(path: Path = GROUPS_CONFIG) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_notebook_lists(path: Path) -> dict[str, list[list[int]]]:
    """Все присваивания номеров FlyWire в тетрадке: имя → список вариантов (код не выполняется)."""
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    found: dict[str, list[list[int]]] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                continue
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(value, int) and not isinstance(value, bool):
                value = [value]
            if (isinstance(value, list) and value
                    and all(isinstance(x, int) and x >= _MIN_ROOT_ID for x in value)):
                found.setdefault(node.targets[0].id, []).append(value)
    return found


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"в файле есть ссылка на код ({module}.{name}) — такой файл не читаем")


def read_data_pickle(path: Path):
    """Читает pickle, в котором только словари, списки, строки и числа; код из файла не выполняется."""
    with open(path, "rb") as handle:
        return _DataOnlyUnpickler(handle).load()


def load_paper_lists(shiu_dir: Path) -> dict[str, dict[str, list[list[int]]]]:
    """Исходные списки статьи Shiu: файл → имя → варианты списка."""
    sez = read_data_pickle(shiu_dir / "sez_neurons.pickle")
    return {
        "figures.ipynb": read_notebook_lists(shiu_dir / "figures.ipynb"),
        "sez_neurons.pickle": {name: [list(ids)] for name, ids in sez.items()},
    }


def _paper_list(spec: dict, paper_lists: dict, where: str) -> list[int]:
    file, name = spec["file"], spec["name"]
    variants = paper_lists.get(file, {}).get(name)
    if not variants:
        raise GroupError(f"{where}: в {file} нет списка «{name}»")
    if any(variant != variants[0] for variant in variants[1:]):
        raise GroupError(f"{where}: в {file} список «{name}» задан несколько раз по-разному")
    return variants[0]


def _check_paper_ids(entries: list[dict], original: list[int], known_ids: set[int], where: str) -> list[int]:
    """Сверяет номера со статьёй и статусы с данными; возвращает номера, найденные в 783."""
    ids = [int(entry["id"]) for entry in entries]
    if len(ids) != len(set(ids)):
        raise GroupError(f"{where}: номер повторяется")
    if sorted(ids) != sorted(original):
        extra = sorted(set(ids) - set(original))
        lost = sorted(set(original) - set(ids))
        raise GroupError(f"{where}: список не совпадает со статьёй (лишние {extra}, пропущены {lost})")
    found = []
    for entry in entries:
        root_id, status = int(entry["id"]), entry["status"]
        if status not in STATUSES:
            raise GroupError(f"{where}: неизвестный статус «{status}» у {root_id}")
        actual = "in_783" if root_id in known_ids else "not_in_783"
        if status != actual:
            raise GroupError(
                f"{where}: у номера {root_id} записано «{STATUSES[status]}», а по данным — «{STATUSES[actual]}»"
            )
        if actual == "in_783":
            found.append(root_id)
    return found


def _select(neurons: pd.DataFrame, criteria: dict, where: str) -> np.ndarray:
    mask = np.ones(len(neurons), dtype=bool)
    for column, wanted in criteria.items():
        if column not in SELECT_COLUMNS:
            raise GroupError(f"{where}: отбор по «{column}» не поддерживается")
        values = wanted if isinstance(wanted, list) else [wanted]
        mask &= neurons[column].isin(values).to_numpy()
    return np.flatnonzero(mask)


def resolve_groups(config: dict, neurons: pd.DataFrame, paper_lists: dict) -> dict[str, dict[str, Group]]:
    """Превращает описание групп во внутренние номера нейронов, проверяя всё по дороге."""
    root_ids = neurons["root_id"].to_numpy()
    index_of = {int(root_id): i for i, root_id in enumerate(root_ids)}
    known_ids = set(index_of)
    resolved: dict[str, dict[str, Group]] = {}
    for section in SECTIONS:
        resolved[section] = {}
        specs = config.get(section) or {}
        unions, mirrors = [], []
        for key, spec in specs.items():
            where = f"{section}.{key}"
            ways = [way for way in ("paper_ids", "select", "union", "mirror_of") if way in spec]
            if not ways or ("union" in ways or "mirror_of" in ways) and len(ways) > 1:
                raise GroupError(f"{where}: состав задаётся одним из paper_ids, select, union или mirror_of")
            if "paper" not in spec and not spec.get("honesty_ru"):
                raise GroupError(f"{where}: группа не из статьи должна иметь пояснение honesty_ru")
            group = Group(
                key=key, section=section, name_ru=spec["name_ru"], about_ru=spec["about_ru"],
                honesty_ru=spec.get("honesty_ru"), source="", paper=None, indices=np.array([], dtype=np.int64),
            )
            if "union" in spec or "mirror_of" in spec:
                group.source = "union" if "union" in spec else "mirror"
                (unions if "union" in spec else mirrors).append((group, spec.get("union") or spec["mirror_of"]))
                resolved[section][key] = group
                continue
            found: list[int] = []
            if "paper_ids" in spec:
                if "paper" not in spec:
                    raise GroupError(f"{where}: у paper_ids нет ссылки paper на файл статьи")
                original = _paper_list(spec["paper"], paper_lists, where)
                found = _check_paper_ids(spec["paper_ids"], original, known_ids, where)
                group.paper = f"{spec['paper']['file']}: {spec['paper']['name']}"
                group.paper_total = len(original)
                group.paper_missing = [i for i in original if i not in known_ids]
            if "select" in spec:
                group.source = "select"
                group.indices = _select(neurons, spec["select"], where)
                outside = sorted(set(found) - set(root_ids[group.indices].tolist()))
                if outside:
                    raise GroupError(f"{where}: номера из статьи {outside} не попали в отбор {spec['select']}")
            else:
                group.source = "paper"
                group.indices = np.array(sorted(index_of[i] for i in found), dtype=np.int64)
            if len(group.indices) == 0:
                raise GroupError(f"{where}: в группе не оказалось ни одного нейрона")
            resolved[section][key] = group
        for group, parts in unions:
            where = f"{section}.{group.key}"
            members: set[int] = set()
            for part in parts:
                other = resolved[section].get(part)
                if other is None or other.source == "union":
                    raise GroupError(f"{where}: нельзя объединить «{part}»")
                members.update(other.indices.tolist())
                group.paper_total += other.paper_total
                group.paper_missing += other.paper_missing
            group.indices = np.array(sorted(members), dtype=np.int64)
        for group, of in mirrors:
            # Двойники бывают только у списка статьи (или объединения списков) — правило D-047.
            where = f"{section}.{group.key}"
            left = resolved[section].get(of)
            if left is None or left.source not in ("paper", "union") or not left.paper_total:
                raise GroupError(f"{where}: двойников ищем только для списка статьи, а «{of}» не такой")
            group.mirror_of = of
            group.mirror = match_twins(left.indices, neurons)
            group.indices = twins_of(group.mirror)  # может быть пусто: тогда двойников нет (D-047)
    return resolved


def groups_to_json(groups: dict[str, dict[str, Group]], neurons: pd.DataFrame) -> dict:
    """Вид для data/processed/groups.json. Номера FlyWire — строками: в браузере большие числа теряют точность."""
    root_ids = neurons["root_id"].to_numpy()
    out: dict = {"statuses": STATUSES}
    for section, by_key in groups.items():
        out[section] = {
            key: {
                "name_ru": group.name_ru,
                "about_ru": group.about_ru,
                "honesty_ru": group.honesty_ru,
                "source": group.source,
                "paper": group.paper,
                "count": int(len(group.indices)),
                "paper_total": group.paper_total,
                "paper_missing_root_ids": [str(i) for i in group.paper_missing],
                **({"mirror_of": group.mirror_of, "mirror_types": [match.to_json() for match in group.mirror]}
                   if group.source == "mirror" else {}),
                "indices": group.indices.tolist(),
                "root_ids": [str(i) for i in root_ids[group.indices]],
            }
            for key, group in by_key.items()
        }
    return out
