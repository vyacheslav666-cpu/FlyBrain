"""`flybrain build-data`: таблица нейронов, матрица связей, группы и картинка в data/processed/ (C3).

Подробно о файлах и столбцах — docs/DATA.md, раздел 5; решения — D-025 в docs/DECISIONS.md.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flybrain import processed
from flybrain.fetch import Source, load_sources, sha256_of
from flybrain.groups import GROUPS_CONFIG, Group, GroupError, groups_to_json, load_group_config, \
    load_paper_lists, resolve_groups
from flybrain.paths import CONFIG_DIR, PROCESSED_DIR, PROJECT_ROOT
from flybrain.preview import Highlight, draw_preview

NAMES_CONFIG = CONFIG_DIR / "names_ru.yaml"
REQUIRED_INPUTS = (
    "Completeness_783.csv",
    "Connectivity_783.parquet",
    "figures.ipynb",
    "sez_neurons.pickle",
    "Supplemental_file1_neuron_annotations.tsv",
)
# Размер вокселя FlyWire в микрометрах: 4 × 4 × 40 нм (D-010).
VOXEL_UM = {"x": 0.004, "y": 0.004, "z": 0.04}
# Текстовые столбцы аннотаций, которые переносим в таблицу нейронов как есть.
TEXT_COLUMNS = ("super_class", "cell_class", "cell_sub_class", "cell_type", "hemibrain_type",
                "synonyms", "side", "flow")
# Правило знака из D-005: ГАМК и глутамат тормозят, остальные медиаторы возбуждают.
INHIBITORY_NT = ("gaba", "glutamate")


class BuildError(Exception):
    """Исходные данные неполны или не сходятся между собой."""


def check_inputs() -> dict[str, Source]:
    """Проверяет, что исходные файлы скачаны и совпадают с закреплёнными (D-009)."""
    sources = {source.name: source for source in load_sources()}
    checked = {}
    for name in REQUIRED_INPUTS:
        source = sources[name]
        where = source.dest.relative_to(PROJECT_ROOT).as_posix()
        if not source.dest.exists():
            raise BuildError(f"нет файла {where}. Сначала выполните: uv run flybrain fetch-data")
        if sha256_of(source.dest) != source.sha256:
            raise BuildError(f"файл {where} не совпадает с закреплённым. Выполните: uv run flybrain fetch-data")
        checked[name] = source
    return checked


def read_annotations(path: Path) -> pd.DataFrame:
    """Таблица аннотаций FlyWire. Пропуском считается только пустая ячейка (строка «na» — это значение)."""
    numeric = [f"{kind}_{axis}" for kind in ("pos", "soma") for axis in "xyz"]
    return pd.read_csv(
        path, sep="\t", usecols=["root_id", *numeric, *TEXT_COLUMNS, "top_nt"],
        dtype={"root_id": "int64", **{column: "float64" for column in numeric},
               **{column: "str" for column in (*TEXT_COLUMNS, "top_nt")}},
        keep_default_na=False, na_values=[""],
    )


def check_connectivity(connectivity: pd.DataFrame, root_ids: np.ndarray) -> None:
    """Файл связей Shiu должен ссылаться на те же нейроны и в том же порядке, что и список нейронов."""
    for role in ("Presynaptic", "Postsynaptic"):
        index = connectivity[f"{role}_Index"].to_numpy()
        if index.min() < 0 or index.max() >= len(root_ids):
            raise BuildError(f"в файле связей номер {role}_Index вне списка нейронов")
        if not np.array_equal(root_ids[index], connectivity[f"{role}_ID"].to_numpy()):
            raise BuildError(f"в файле связей {role}_ID не совпадает с порядком строк Completeness_783.csv")
    excitatory = connectivity["Excitatory"].to_numpy()
    count = connectivity["Connectivity"].to_numpy()
    if not np.isin(excitatory, (-1, 1)).all() or (count < 1).any():
        raise BuildError("в файле связей неожиданные значения Excitatory или Connectivity")
    if not np.array_equal(excitatory * count, connectivity["Excitatory x Connectivity"].to_numpy()):
        raise BuildError("в файле связей «Excitatory x Connectivity» не равно произведению столбцов")


def model_signs(pre: np.ndarray, excitatory: np.ndarray, n_neurons: int) -> np.ndarray:
    """Знак нейрона в модели Shiu: +1, −1 или 0, если выходных связей нет (D-025)."""
    sign = np.zeros(n_neurons, dtype=np.int8)
    sign[pre] = excitatory
    if not np.array_equal(sign[pre], excitatory):
        raise BuildError("в файле связей у одного нейрона связи разного знака")
    return sign


def build_neuron_table(root_ids: np.ndarray, annotations: pd.DataFrame, sign: np.ndarray) -> pd.DataFrame:
    """Строка = внутренний номер нейрона (порядок Completeness_783.csv)."""
    ann = annotations.set_index("root_id").reindex(root_ids)
    has_soma = ann["soma_x"].notna().to_numpy()
    has_anchor = ann["pos_x"].notna().to_numpy()
    table = pd.DataFrame({
        "index": np.arange(len(root_ids), dtype=np.int32),
        "root_id": root_ids.astype(np.int64),
    })
    # Точка нейрона — тело клетки, если известно, иначе опорная точка (D-016).
    for axis in "xyz":
        voxels = np.where(has_soma, ann[f"soma_{axis}"].to_numpy(), ann[f"pos_{axis}"].to_numpy())
        um = voxels * VOXEL_UM[axis]
        if axis == "x":
            um = -um  # снимки FlyWire зеркальные — отражаем X (D-011)
        table[f"{axis}_um"] = um.astype(np.float32)
    table["pos_source"] = np.where(has_soma, "soma", np.where(has_anchor, "anchor", "none"))
    for column in TEXT_COLUMNS:
        table[column] = ann[column].to_numpy()
    table["nt"] = ann["top_nt"].to_numpy()
    table["sign"] = sign
    return table


def build_connectivity_csr(post: np.ndarray, pre: np.ndarray, weight: np.ndarray, n_neurons: int) -> dict:
    """Матрица связей в виде CSR (хранятся только существующие связи).

    Строка — получатель, `indices` — отправители, `weight` — число синапсов со знаком.
    Вход нейрона i = сумма weight · (спайки отправителей) по его строке, × 0,275 мВ (D-005).
    """
    order = np.lexsort((pre, post))
    post_sorted, pre_sorted = post[order], pre[order]
    same = (post_sorted[1:] == post_sorted[:-1]) & (pre_sorted[1:] == pre_sorted[:-1])
    if same.any():
        raise BuildError("в файле связей есть повторяющиеся пары нейронов")
    if np.abs(weight).max() > np.iinfo(np.int16).max:
        raise BuildError("число синапсов в паре не помещается в int16")
    indptr = np.zeros(n_neurons + 1, dtype=np.int64)
    np.cumsum(np.bincount(post, minlength=n_neurons), out=indptr[1:])
    return {
        "indptr": indptr,
        "indices": pre_sorted.astype(np.int32),
        "weight": weight[order].astype(np.int16),
        "n_neurons": np.int64(n_neurons),
    }


def sign_nt_mismatches(neurons: pd.DataFrame) -> int:
    """Сколько нейронов имеют знак в модели, отличный от правила D-005 по медиатору из аннотаций."""
    known = neurons["nt"].notna() & (neurons["sign"] != 0)
    expected = np.where(neurons["nt"].isin(INHIBITORY_NT), -1, 1)
    return int((known & (neurons["sign"] != expected)).sum())


def _fmt(number: int) -> str:
    return f"{number:,}".replace(",", " ")


def _count(number: int, one: str, few: str, many: str) -> str:
    """«1 нейрон», «2 нейрона», «5 нейронов» — с правильным окончанием."""
    last_two, last = number % 100, number % 10
    word = many if 11 <= last_two <= 14 else one if last == 1 else few if 2 <= last <= 4 else many
    return f"{_fmt(number)} {word}"


def _sides(neurons: pd.DataFrame, group: Group, names: dict) -> str:
    counts = neurons["side"].iloc[group.indices].fillna("missing").value_counts()
    return ", ".join(f"{names['sides'].get(side, side)} {count}" for side, count in counts.items())


def _print_group(neurons: pd.DataFrame, group: Group, names: dict) -> None:
    count = len(group.indices)
    if group.source == "select" and not group.paper:
        origin = "наш отбор по таблице аннотаций [ЧЕСТНОСТЬ]"
    elif group.paper_total:
        missing = len(group.paper_missing)
        origin = f"из статьи {group.paper_total - missing} из {group.paper_total}"
        if missing:
            origin += f" (не найден в 783: {missing})"
    else:
        origin = "наш отбор"
    print(f"    {group.name_ru}: {_fmt(count)} — {origin}; сторона мухи: {_sides(neurons, group, names)}")


def run_build_data() -> int:
    started = time.monotonic()
    print("FlyBrain: подготовка данных в data/processed/\n")
    try:
        inputs = check_inputs()
        paths = {name: source.dest for name, source in inputs.items()}
        print(f"[OK] исходные файлы на месте, контрольные суммы совпадают ({len(inputs)} шт.)")

        root_ids = pd.read_csv(paths["Completeness_783.csv"], index_col=0).index.to_numpy(np.int64)
        connectivity = pd.read_parquet(paths["Connectivity_783.parquet"])
        annotations = read_annotations(paths["Supplemental_file1_neuron_annotations.tsv"])
        check_connectivity(connectivity, root_ids)
        n = len(root_ids)
        pre = connectivity["Presynaptic_Index"].to_numpy(np.int64)
        post = connectivity["Postsynaptic_Index"].to_numpy(np.int64)
        weight = connectivity["Excitatory x Connectivity"].to_numpy(np.int64)
        sign = model_signs(pre, connectivity["Excitatory"].to_numpy(np.int8), n)
        neurons = build_neuron_table(root_ids, annotations, sign)

        groups = resolve_groups(load_group_config(), neurons, load_paper_lists(paths["figures.ipynb"].parent))
        csr = build_connectivity_csr(post, pre, weight, n)
        del connectivity
    except (BuildError, GroupError) as exc:
        print(f"\n[ОШИБКА] {exc}", file=sys.stderr)
        return 1

    with open(NAMES_CONFIG, encoding="utf-8") as handle:
        names = yaml.safe_load(handle)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    neurons.to_parquet(PROCESSED_DIR / processed.NEURONS_FILE, index=False)
    np.savez(PROCESSED_DIR / processed.CONNECTIVITY_FILE, **csr)
    with open(PROCESSED_DIR / processed.GROUPS_FILE, "w", encoding="utf-8") as handle:
        json.dump({"format": processed.FORMAT_VERSION, **groups_to_json(groups, neurons)},
                  handle, ensure_ascii=False, indent=1)

    annotated = int(neurons["super_class"].notna().sum())
    sources = neurons["pos_source"].value_counts()
    signs = neurons["sign"].value_counts()
    synapses = int(np.abs(csr["weight"]).sum(dtype=np.int64))
    matrix_mb = (PROCESSED_DIR / processed.CONNECTIVITY_FILE).stat().st_size / 1024**2

    print(f"\nТаблица нейронов: {_fmt(n)} ({processed.NEURONS_FILE})")
    print(f"  с аннотацией {_fmt(annotated)}; без аннотации {n - annotated} — у них нет координат, "
          "на картинке их нет")
    print(f"  точка — тело клетки: {_fmt(sources.get('soma', 0))}; "
          f"опорная точка на отростке: {_fmt(sources.get('anchor', 0))}")
    print(f"  знак в модели Shiu: возбуждающих {_fmt(signs.get(1, 0))}, тормозящих {_fmt(signs.get(-1, 0))}, "
          f"без выходных связей {_fmt(signs.get(0, 0))}")
    mismatches = _count(sign_nt_mismatches(neurons), "нейрона", "нейронов", "нейронов")
    print(f"  у {mismatches} знак в модели не совпадает с медиатором из таблицы аннотаций — "
          "в модели используем знак Shiu (D-025)")
    print(f"Матрица связей: {_count(len(csr['indices']), 'пара', 'пары', 'пар')}, "
          f"{_count(synapses, 'синапс', 'синапса', 'синапсов')} ({processed.CONNECTIVITY_FILE}, {matrix_mb:.0f} МБ)")
    print("Группы нейронов сверены со статьёй Shiu et al. 2024 и с данными 783:")
    for section, title in (("stimuli", "Стимулы"), ("readouts", "Исполнители")):
        print(f"  {title}:")
        for group in groups[section].values():
            _print_group(neurons, group, names)

    stimuli, readouts = groups["stimuli"], groups["readouts"]
    highlights = [
        Highlight(f"Сахар — {len(stimuli['sugar'].indices)} вкусовых нейронов хоботка\n(список Shiu et al. 2024)",
                  "сахар", stimuli["sugar"].indices, "#ff453a", 14, (-120, -40)),
        Highlight(f"MN9 — мотонейрон хоботка ({len(readouts['MN9'].indices)})",
                  "MN9", readouts["MN9"].indices, "#ffd60a", 42, (120, -45)),
        Highlight(f"GF — гигантское волокно ({len(readouts['GF'].indices)})",
                  "GF", readouts["GF"].indices, "#40c8ff", 42, (110, 60)),
    ]
    draw_preview(neurons, highlights, PROCESSED_DIR / processed.PREVIEW_FILE)
    print(f"Картинка: data/processed/{processed.PREVIEW_FILE}")

    build_info = {
        "format": processed.FORMAT_VERSION,
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs_sha256": {name: source.sha256 for name, source in inputs.items()},
        "configs_sha256": {path.name: sha256_of(path) for path in (GROUPS_CONFIG, NAMES_CONFIG)},
        "neurons": n,
        "pairs": int(len(csr["indices"])),
        "synapses": synapses,
    }
    with open(PROCESSED_DIR / processed.BUILD_INFO_FILE, "w", encoding="utf-8") as handle:
        json.dump(build_info, handle, ensure_ascii=False, indent=1)

    shadow_left, shadow_right = len(stimuli["shadow_left"].indices), len(stimuli["shadow_right"].indices)
    print(f"\nГотово за {time.monotonic() - started:.0f} с.")
    sugar = _count(len(stimuli["sugar"].indices), "нейрон", "нейрона", "нейронов")
    print(f"сахар: {sugar}, горечь: {len(stimuli['bitter'].indices)}, "
          f"усик: {len(stimuli['antenna'].indices)}, тень: {shadow_left} слева / {shadow_right} справа")
    return 0
