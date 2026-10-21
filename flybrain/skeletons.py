"""Формы нейронов из скелетов FlyWire (коммит C12b, D-037): подготовка, чтение по номеру нейрона и проверка координат.

Исходный файл `sk_lod1_783_healed_ds2.parquet` (docs/DATA.md, раздел 3) — 268 млн точек скелетов всех нейронов,
в нанометрах, разбитых всего на 4 больших блока. Прочитать из него один нейрон по щелчку — значит распаковать блок
в 1,5 ГБ, поэтому один раз перекладываем точки в `data/processed/skeletons/`: нейрон за нейроном в порядке
внутренних номеров, плюс указатель «номер → где лежат его точки». Координаты переводятся в те же микрометры
с отражённой осью X, что и облако (D-011, D-025): точка нейрона и его форма лежат в одной системе.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.paths import PROCESSED_DIR, PROJECT_ROOT

SKELETON_DIR = PROCESSED_DIR / "skeletons"
NODES_FILE = "nodes.bin"  # записи NODE_DTYPE подряд, нейрон за нейроном в порядке внутренних номеров
INDEX_FILE = "index.npz"  # start, count — где лежат точки каждого нейрона (по внутреннему номеру)
INFO_FILE = "info.json"  # справка; пишется последней: есть файл — подготовка закончена
FORMAT_VERSION = 1

NM_PER_UM = 1000.0
# Точка скелета: положение в мкм (X отражён) и номера точки и её «родителя» внутри нейрона, как в файле FlyWire
# (у корня родитель −1). Номера переводятся в позиции при отдаче нейрона странице.
NODE_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("node", "<i4"), ("parent", "<i4")])
BATCH_ROWS = 8_000_000  # столько точек читаем за раз: ~0,5 ГБ памяти

# Проверка «точка нейрона лежит на его форме» (D-037). Пороги записаны до первого прогона на настоящих данных.
ON_SKELETON_UM = 5.0  # точка не дальше этого от ближайшей точки скелета — «лежит на форме» (тело клетки — 2–5 мкм)
PASS_SHARE = 0.95  # проверка пройдена, если так лежит не меньше этой доли нейронов
PASS_MEDIAN_UM = 2.0  # и половина нейронов не дальше этого
CONTROL_MIN_MEDIAN_UM = 50.0  # контроль: без отражения X точки должны уйти далеко, иначе проверка ничего не доказывает

HEADER = np.dtype([("version", "<u4"), ("count", "<u4"), ("distance_um", "<f4"), ("reserved", "<u4")])
WIRE_VERSION = 1


def neurons_fingerprint(neurons: pd.DataFrame) -> str:
    """Сумма порядка номеров FlyWire: подготовленные скелеты подходят только к той же таблице нейронов."""
    return hashlib.sha256(neurons["root_id"].to_numpy(dtype="<i8").tobytes()).hexdigest()[:16]


def _internal_index(root_ids: np.ndarray, sorted_ids: np.ndarray, sorter: np.ndarray) -> np.ndarray:
    """Внутренний номер для каждого номера FlyWire; −1 — нейрона нет в модели."""
    position = np.searchsorted(sorted_ids, root_ids)
    position[position == len(sorted_ids)] = 0
    found = sorted_ids[position] == root_ids
    return np.where(found, sorter[position], -1)


def _batches(source: Path, columns: list[str]):
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(source)
    yield parquet.metadata.num_rows
    for batch in parquet.iter_batches(batch_size=BATCH_ROWS, columns=columns):
        yield batch


def prepare(source: Path, neurons: pd.DataFrame, out_dir: Path = SKELETON_DIR, source_sha256: str = "",
            log=print) -> dict:
    """Перекладывает скелеты в `out_dir` (два прохода по файлу) и возвращает справку. Старые файлы заменяются."""
    started = time.monotonic()
    root_ids = neurons["root_id"].to_numpy(dtype=np.int64)
    sorter = np.argsort(root_ids).astype(np.int64)
    sorted_ids = root_ids[sorter]
    n = len(root_ids)

    # Проход 1: сколько точек у каждого нейрона модели.
    counts = np.zeros(n, dtype=np.int64)
    batches = _batches(source, ["neuron"])
    total_rows = next(batches)
    skipped = 0
    seen = 0
    for batch in batches:
        index = _internal_index(batch.column(0).to_numpy(), sorted_ids, sorter)
        skipped += int((index < 0).sum())
        counts += np.bincount(index[index >= 0], minlength=n)
        seen += batch.num_rows
        log(f"\r    проход 1 из 2: {100 * seen // total_rows}%", end="")
    log("")
    start = np.zeros(n, dtype=np.int64)
    np.cumsum(counts[:-1], out=start[1:])
    kept = int(counts.sum())

    # Проход 2: точки — на свои места. Внутри нейрона порядок строк файла сохраняется.
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / INFO_FILE).unlink(missing_ok=True)  # пока перекладываем, подготовка считается незаконченной
    nodes_part = out_dir / (NODES_FILE + ".part")
    nodes = np.memmap(nodes_part, dtype=NODE_DTYPE, mode="w+", shape=(max(kept, 1),))
    filled = np.zeros(n, dtype=np.int64)
    batches = _batches(source, ["neuron", "node_id", "parent_id", "x", "y", "z"])
    next(batches)
    seen = 0
    for batch in batches:
        columns = {name: batch.column(name).to_numpy() for name in batch.schema.names}
        index = _internal_index(columns["neuron"], sorted_ids, sorter)
        rows = np.flatnonzero(index >= 0)
        order = rows[np.argsort(index[rows], kind="stable")]
        owner = index[order]
        first = np.searchsorted(owner, owner, side="left")  # где в этой пачке начинаются точки того же нейрона
        destination = start[owner] + filled[owner] + (np.arange(len(order)) - first)
        record = np.empty(len(order), dtype=NODE_DTYPE)
        record["x"] = -columns["x"][order] / NM_PER_UM  # X отражён, как у облака (D-011)
        record["y"] = columns["y"][order] / NM_PER_UM
        record["z"] = columns["z"][order] / NM_PER_UM
        record["node"] = columns["node_id"][order]
        record["parent"] = columns["parent_id"][order]
        nodes[destination] = record
        filled += np.bincount(owner, minlength=n)
        seen += batch.num_rows
        log(f"\r    проход 2 из 2: {100 * seen // total_rows}%", end="")
    log("")
    nodes.flush()
    del nodes
    if not np.array_equal(filled, counts):
        raise RuntimeError("при перекладке скелетов число точек не сошлось — файл прочитан не так, как ожидалось")

    nodes_part.replace(out_dir / NODES_FILE)
    np.savez(out_dir / INDEX_FILE, start=start, count=counts.astype(np.int32))
    info = {
        "format": FORMAT_VERSION,
        "source": source.name,
        "source_sha256": source_sha256,
        "neurons_fingerprint": neurons_fingerprint(neurons),
        "neurons_total": n,
        "neurons_with_skeleton": int((counts > 0).sum()),
        "nodes": kept,
        "nodes_not_in_model": skipped,
        "seconds": round(time.monotonic() - started, 1),
    }
    return info


def write_info(info: dict, out_dir: Path = SKELETON_DIR) -> None:
    with open(out_dir / INFO_FILE, "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=2)


def read_info(out_dir: Path = SKELETON_DIR) -> dict | None:
    try:
        with open(out_dir / INFO_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


@dataclass
class Skeleton:
    positions: np.ndarray  # (n, 3) float32, мкм, X отражён
    parents: np.ndarray  # (n,) int32: позиция «родителя» в этом же массиве, −1 у корня


class SkeletonStore:
    """Подготовленные скелеты: чтение формы одного нейрона по внутреннему номеру."""

    def __init__(self, out_dir: Path, info: dict):
        self.info = info
        with np.load(out_dir / INDEX_FILE) as index:
            self.start = index["start"]
            self.count = index["count"]
        total = int(self.count.sum())
        self.nodes = np.memmap(out_dir / NODES_FILE, dtype=NODE_DTYPE, mode="r", shape=(max(total, 1),))

    @classmethod
    def open(cls, neurons: pd.DataFrame, out_dir: Path = SKELETON_DIR) -> tuple["SkeletonStore | None", str | None]:
        """(хранилище, None) или (None, почему форм нет) — без скелетов страница работает как раньше."""
        info = read_info(out_dir)
        if info is None:
            return None, "скелеты не скачаны"
        if info.get("format") != FORMAT_VERSION:
            return None, "скелеты подготовлены старой версией программы"
        if info.get("neurons_fingerprint") != neurons_fingerprint(neurons):
            return None, "скелеты подготовлены для другой таблицы нейронов"
        try:
            return cls(out_dir, info), None
        except (OSError, ValueError, KeyError) as exc:
            return None, f"файлы скелетов не читаются ({exc})"

    def has(self, index: int) -> bool:
        return 0 <= index < len(self.count) and self.count[index] > 0

    def get(self, index: int) -> Skeleton:
        begin = int(self.start[index])
        part = np.asarray(self.nodes[begin:begin + int(self.count[index])])
        positions = np.column_stack([part["x"], part["y"], part["z"]]).astype(np.float32)
        node_ids = part["node"]
        order = np.argsort(node_ids, kind="stable")
        where = np.searchsorted(node_ids, part["parent"], sorter=order)
        where = np.minimum(where, len(order) - 1)
        parents = order[where].astype(np.int32)
        parents[node_ids[parents] != part["parent"]] = -1  # корень (−1) или родитель, которого нет в файле
        return Skeleton(positions, parents)

    def wire(self, index: int, point: np.ndarray | None) -> bytes:
        """Ответ странице: заголовок HEADER, затем float32 x, y, z на точку, затем int32 позиция родителя."""
        skeleton = self.get(index)
        header = np.zeros(1, dtype=HEADER)
        header["version"] = WIRE_VERSION
        header["count"] = len(skeleton.parents)
        header["distance_um"] = distance_to(skeleton.positions, point)
        return header.tobytes() + skeleton.positions.astype("<f4").tobytes() + skeleton.parents.astype("<i4").tobytes()


def distance_to(positions: np.ndarray, point: np.ndarray | None) -> float:
    """Расстояние (мкм) от точки до ближайшей точки скелета; NaN, если точки нет."""
    if point is None or len(positions) == 0 or not np.all(np.isfinite(point)):
        return float("nan")
    return float(np.sqrt(((positions - np.asarray(point, dtype=np.float32)) ** 2).sum(axis=1).min()))


def check_alignment(store: SkeletonStore, neurons: pd.DataFrame, log=print) -> dict:
    """Проверка координат (D-037): лежит ли точка каждого нейрона облака на его собственной форме.

    Для контроля то же расстояние считается, если забыть отразить X: тогда точки должны уйти далеко —
    значит, проверка действительно ловит ошибку в координатах.
    """
    xyz = neurons[["x_um", "y_um", "z_um"]].to_numpy(dtype=np.float32)
    distance = np.full(len(neurons), np.nan)
    control = np.full(len(neurons), np.nan)
    candidates = np.flatnonzero(store.count > 0)
    for done, index in enumerate(candidates):
        point = xyz[index]
        if not np.all(np.isfinite(point)):
            continue
        begin = int(store.start[index])
        part = np.asarray(store.nodes[begin:begin + int(store.count[index])])
        positions = np.column_stack([part["x"], part["y"], part["z"]])
        distance[index] = distance_to(positions, point)
        control[index] = distance_to(positions, point * np.array([-1, 1, 1], dtype=np.float32))
        if done % 5000 == 0:
            log(f"\r    проверка координат: {100 * done // len(candidates)}%", end="")
    log("\r    проверка координат: 100%")

    def summary(values: np.ndarray) -> dict:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return {"neurons": 0}
        return {
            "neurons": int(len(values)),
            "median_um": round(float(np.median(values)), 2),
            "p95_um": round(float(np.percentile(values, 95)), 2),
            "p99_um": round(float(np.percentile(values, 99)), 2),
            "max_um": round(float(values.max()), 1),
            "share_on_skeleton": round(float((values <= ON_SKELETON_UM).mean()), 4),
        }

    sources = neurons["pos_source"].to_numpy()
    result = {
        "threshold_um": ON_SKELETON_UM,
        "all": summary(distance),
        "soma": summary(distance[sources == "soma"]),
        "anchor": summary(distance[sources == "anchor"]),
        "control_not_mirrored": summary(control),
    }
    every = result["all"]
    result["passed"] = bool(
        every.get("neurons", 0) > 0
        and every["share_on_skeleton"] >= PASS_SHARE
        and every["median_um"] <= PASS_MEDIAN_UM
        and result["control_not_mirrored"]["median_um"] >= CONTROL_MIN_MEDIAN_UM
    )
    finite = np.isfinite(distance)
    worst = np.argsort(-np.where(finite, distance, -1))[:10]
    result["worst"] = [{"index": int(i), "root_id": str(int(neurons["root_id"].iloc[i])),
                        "pos_source": str(sources[i]), "distance_um": round(float(distance[i]), 1)}
                       for i in worst if finite[i]]
    return result


def _fmt(number: float) -> str:
    return f"{number:,.0f}".replace(",", " ")


def _print_check(check: dict) -> None:
    every, control = check["all"], check["control_not_mirrored"]
    print(f"\nПроверка координат: точка нейрона на его форме (не дальше {check['threshold_um']:g} мкм)")
    for key, title in (("all", "все нейроны"), ("soma", "точка — тело клетки"), ("anchor", "точка на отростке")):
        part = check[key]
        if part.get("neurons"):
            print(f"  {title}: {_fmt(part['neurons'])}; на форме {100 * part['share_on_skeleton']:.2f}%, "
                  f"расстояние: половина ≤ {part['median_um']:g} мкм, 95% ≤ {part['p95_um']:g} мкм, "
                  f"самое большое {part['max_um']:g} мкм")
    if control.get("neurons"):
        print(f"  контроль — если забыть отразить X: на форме {100 * control['share_on_skeleton']:.2f}%, "
              f"половина ≤ {control['median_um']:g} мкм")
    verdict = "[OK] координаты совпадают" if check["passed"] else "[ВНИМАНИЕ] координаты НЕ совпадают"
    print(f"  {verdict} (нужно: на форме ≥ {100 * PASS_SHARE:g}% нейронов, половина ≤ {PASS_MEDIAN_UM:g} мкм; "
          f"без отражения X половина ≥ {CONTROL_MIN_MEDIAN_UM:g} мкм)")


def run_fetch_skeletons() -> int:
    """`flybrain fetch-skeletons`: скачать скелеты (5,4 ГБ) с проверкой SHA-256, подготовить и проверить координаты."""
    import shutil

    from flybrain import processed
    from flybrain.fetch import FetchError, fetch_one, load_sources

    if not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists():
        print("[ОШИБКА] сначала нужны подготовленные данные: uv run flybrain fetch-data, затем uv run flybrain build-data")
        return 1
    (source,) = [s for s in load_sources(section="optional") if s.name.startswith("sk_")]
    neurons = processed.load_neurons()
    info = read_info()
    if (info and info.get("format") == FORMAT_VERSION and info.get("source_sha256") == source.sha256
            and info.get("neurons_fingerprint") == neurons_fingerprint(neurons)):
        print("Скелеты уже скачаны и подготовлены (data/processed/skeletons/).")
        _print_check(info["check"])
        return 0 if info["check"]["passed"] else 1

    print(f"FlyBrain: формы нейронов — {source.description}\n")
    if not source.dest.exists():
        partial = source.dest.with_name(source.dest.name + ".part")
        have = partial.stat().st_size if partial.exists() else 0
        need = (source.size_bytes - have) + source.size_bytes  # докачать + подготовленная копия того же размера
        source.dest.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(source.dest.parent).free
        if free < need * 1.05:
            print(f"[ОШИБКА] на диске свободно {free / 1024**3:.1f} ГБ, а нужно около {need / 1024**3:.1f} ГБ.")
            return 1
        print(f"Скачиваю {source.size_bytes / 1024**3:.1f} ГБ. Это долго (при 1 МБ/с — около полутора часов); "
              "окно можно закрыть — при следующем запуске скачивание продолжится с того же места.")
    print(f"{source.name}")
    try:
        outcome = fetch_one(source)
    except FetchError as exc:
        print(f"\n[ОШИБКА] {exc}")
        return 1
    print(f"  [OK] {outcome}, контрольная сумма совпадает ({source.sha256[:12]}…)")

    print("\nПерекладываю скелеты для быстрого чтения (несколько минут)…")
    try:
        info = prepare(source.dest, neurons, source_sha256=source.sha256)  # fetch_one уже сверил сумму
    except PermissionError as exc:  # Windows не даёт заменить файл, который читает запущенный сервер
        print(f"\n[ОШИБКА] файл занят ({exc.filename}). Закройте окно сервера FlyBrain (run.bat) и запустите команду снова.")
        return 1
    print(f"  нейронов модели с формой: {_fmt(info['neurons_with_skeleton'])} из {_fmt(info['neurons_total'])}; "
          f"точек: {_fmt(info['nodes'])} (ещё {_fmt(info['nodes_not_in_model'])} — у нейронов вне модели)")
    store = SkeletonStore(SKELETON_DIR, info)
    info["check"] = check_alignment(store, neurons)
    del store
    write_info(info)
    _print_check(info["check"])
    print("\nГотово. Перезапустите сервер (run.bat): в панели «Вид» появится флажок «Формы нейронов».")
    print(f"Исходный файл {source.dest.relative_to(PROJECT_ROOT).as_posix()} больше не нужен для работы; "
          "его можно удалить, чтобы освободить 5 ГБ.")
    return 0 if info["check"]["passed"] else 1
