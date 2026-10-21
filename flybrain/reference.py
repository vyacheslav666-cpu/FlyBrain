"""`flybrain reference`: эталонные прогоны неизменённого `model.py` Shiu на Brian2 (C4, D-006, D-029).

Каждый повтор — отдельный вызов `run_trial` из `model.py` в отдельном процессе; результат сразу пишется
в свой файл. Поэтому прогон можно прервать (Ctrl+C) и продолжить той же командой: готовые повторы
не пересчитываются. Когда все повторы условия готовы, они собираются в один файл в формате Shiu
(`construct_dataframe` из того же `model.py`).
"""

import ctypes
import importlib.util
import json
import os
import platform
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from flybrain import processed
from flybrain.fetch import load_sources, sha256_of
from flybrain.paths import CONFIG_DIR, PROCESSED_DIR, PROJECT_ROOT

REFERENCE_CONFIG = CONFIG_DIR / "reference.yaml"
NAMES_CONFIG = CONFIG_DIR / "names_ru.yaml"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
SHUFFLED_FILE = "Connectivity_783_shuffled.parquet"
INFO_FILE = "reference_info.json"
RATES_FILE = "rates.csv"
REQUIRED_INPUTS = ("model.py", "Completeness_783.csv", "Connectivity_783.parquet")
# Меняется, когда меняется формат файлов повторов; старые повторы тогда не подходят.
TRIAL_FORMAT = 1
# «Активный нейрон» — средняя частота за все повторы не меньше 1 Гц (D-018).
ACTIVE_HZ = 1.0
# Пик памяти одного повтора Brian2 — замер C4 (2,5–2,9 ГБ, D-029); нужен только для предупреждения.
GB_PER_JOB = 2.8


class ReferenceError(Exception):
    """Эталонный прогон нельзя запустить или продолжить."""


@dataclass
class Condition:
    key: str
    name_ru: str
    excite_group: str
    excite_hz: float
    excite2_group: str | None
    excite2_hz: float
    connectivity: str  # "original" или "shuffled"
    honesty_ru: str | None


def load_reference_config(path: Path = REFERENCE_CONFIG) -> dict:
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    conditions = []
    for entry in config["conditions"]:
        excite2 = entry.get("excite2") or {}
        conditions.append(Condition(
            key=entry["key"], name_ru=entry["name_ru"],
            excite_group=entry["excite"]["group"], excite_hz=float(entry["excite"]["hz"]),
            excite2_group=excite2.get("group"), excite2_hz=float(excite2.get("hz", 0)),
            connectivity=entry.get("connectivity", "original"), honesty_ru=entry.get("honesty_ru"),
        ))
    keys = [condition.key for condition in conditions]
    if len(set(keys)) != len(keys):
        raise ReferenceError("в config/reference.yaml повторяются ключи условий")
    config["conditions"] = conditions
    return config


def trial_seed(key: str, trial: int) -> int:
    """Зерно случайных чисел повтора: своё у каждого повтора и одно и то же при каждом запуске."""
    return zlib.crc32(f"{key}:{trial}".encode("utf-8"))


def shuffle_connectivity(connectivity: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Контроль Б7: получатели всех связей переставлены случайно (D-029).

    Отправитель и вес связи остаются на месте, поэтому у каждого нейрона сохраняются число выходов
    и их веса; столбец получателей лишь переставлен, поэтому сохраняется и число входов.
    """
    order = np.random.default_rng(seed).permutation(len(connectivity))
    shuffled = connectivity.copy()
    for column in ("Postsynaptic_Index", "Postsynaptic_ID"):
        if column in shuffled:
            shuffled[column] = connectivity[column].to_numpy()[order]
    return shuffled


# ---------- работа в отдельном процессе ----------

def _load_shiu(model_path: str):
    """Загружает `model.py` Shiu как модуль, не меняя в нём ни строчки."""
    spec = importlib.util.spec_from_file_location("shiu_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _peak_memory_gb() -> float:
    """Наибольший объём оперативной памяти, занятый этим процессом."""
    if platform.system() == "Windows":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        # Описатель процесса — 64-битный; без явных типов ctypes обрезал бы его до 32 бит.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.PeakWorkingSetSize / 1024**3
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def _root_cause(exc: BaseException) -> str:
    while exc.__cause__ is not None:
        exc = exc.__cause__
    return f"{type(exc).__name__}: {exc}"


def run_one_trial(task: dict) -> dict:
    """Один повтор: `run_trial` Shiu с его же параметрами; меняются только частоты стимулов."""
    started = time.monotonic()
    import brian2
    from brian2.devices.device import auto_target

    shiu = _load_shiu(task["model_path"])
    params = dict(shiu.default_params)
    params["r_poi"] = task["r_poi_hz"] * brian2.Hz
    params["r_poi2"] = task["r_poi2_hz"] * brian2.Hz
    params["t_run"] = task["duration_ms"] * brian2.ms
    brian2.seed(task["seed"])
    try:
        spikes = shiu.run_trial(task["exc"], task["exc2"], [], task["path_comp"], task["path_con"], params)
    except Exception as exc:
        # Ошибка Brian2 держит ссылки на скомпилированный код, и передать её в главный процесс нельзя
        # («cannot pickle 'PyCapsule'»): передаём текст первопричины.
        raise RuntimeError(_root_cause(exc)) from None

    neurons = np.array(sorted(spikes), dtype=np.int32)
    times = [np.asarray(spikes[i] / brian2.second, dtype=np.float64) for i in neurons]
    neuron = np.repeat(neurons, [len(t) for t in times]).astype(np.int32)
    t = np.concatenate(times) if times else np.zeros(0)
    seconds = time.monotonic() - started
    memory = _peak_memory_gb()
    # Каким способом Brian2 реально считал: «cython» (скомпилированный код) или запасной «numpy».
    codegen = str(auto_target().class_name)
    target = Path(task["out_file"])
    part = target.with_name(target.name + ".part")
    with open(part, "wb") as handle:
        np.savez(handle, neuron=neuron, t=t, meta=np.array(task["meta"]),
                 seconds=np.array(seconds), peak_memory_gb=np.array(memory), codegen=np.array(codegen))
    os.replace(part, target)
    return {"key": task["key"], "trial": task["trial"], "seconds": seconds, "peak_memory_gb": memory,
            "spikes": int(len(t)), "codegen": codegen}


# ---------- подготовка ----------

def check_inputs() -> dict[str, Path]:
    """Файлы Shiu скачаны и не изменены (D-006, D-009): эталон считается только неизменённым кодом."""
    sources = {source.name: source for source in load_sources()}
    paths = {}
    for name in REQUIRED_INPUTS:
        source = sources[name]
        where = source.dest.relative_to(PROJECT_ROOT).as_posix()
        if not source.dest.exists():
            raise ReferenceError(f"нет файла {where}. Сначала выполните: uv run flybrain fetch-data")
        if sha256_of(source.dest) != source.sha256:
            raise ReferenceError(f"файл {where} не совпадает с закреплённым. Выполните: uv run flybrain fetch-data")
        paths[name] = source.dest
    if not (PROCESSED_DIR / processed.GROUPS_FILE).exists():
        raise ReferenceError("нет data/processed/groups.json. Сначала выполните: uv run flybrain build-data")
    return paths


def resolve_indices(groups: dict, root_ids: np.ndarray, key: str) -> list[int]:
    """Внутренние номера группы-стимула; сверяются с номерами FlyWire в списке нейронов Shiu."""
    group = groups["stimuli"].get(key)
    if group is None:
        raise ReferenceError(f"в groups.json нет группы-стимула «{key}»")
    indices = np.array(group["indices"], dtype=np.int64)
    if not np.array_equal(root_ids[indices], np.array(group["root_ids"], dtype=np.int64)):
        raise ReferenceError(f"группа «{key}» не сходится с Completeness_783.csv. "
                             "Выполните заново: uv run flybrain build-data")
    return [int(i) for i in indices]


def ensure_shuffled(path_con: Path, out_dir: Path, seed: int) -> Path:
    target = out_dir / SHUFFLED_FILE
    if not target.exists():
        print(f"Готовлю перемешанные связи для контроля (зерно {seed})…", flush=True)
        shuffled = shuffle_connectivity(pd.read_parquet(path_con), seed)
        part = target.with_name(target.name + ".part")
        shuffled.to_parquet(part, index=False)
        os.replace(part, target)
    return target


def read_trial_meta(path: Path) -> str | None:
    try:
        with np.load(path) as arrays:
            return str(arrays["meta"])
    except (OSError, ValueError, KeyError):
        return None


def load_trial(path: Path) -> dict:
    with np.load(path) as arrays:
        return {name: arrays[name] for name in arrays.files}


# ---------- вывод ----------

def _where(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def condition_rates(trials: list[dict], n_neurons: int, duration_s: float) -> np.ndarray:
    """Средняя частота каждого нейрона за все повторы, Гц (как `get_rate` в utils.py Shiu)."""
    counts = np.zeros(n_neurons, dtype=np.int64)
    for trial in trials:
        counts += np.bincount(trial["neuron"], minlength=n_neurons)
    return counts / (len(trials) * duration_s)


def print_summary(config: dict, rows: list[tuple[Condition, np.ndarray, int]], groups: dict,
                  neurons: pd.DataFrame, names: dict) -> None:
    readouts = [(key, groups["readouts"][key]) for key in config["summary_readouts"]]
    columns = []
    side_order = {"left": 0, "right": 1}
    for key, group in readouts:
        for index in sorted(group["indices"], key=lambda i: side_order.get(str(neurons.at[i, "side"]), 2)):
            side = str(neurons.at[index, "side"])
            columns.append((f"{key} {names['sides'].get(side, side)[:1].upper()}", index))
    width = max(len(row[0].name_ru) for row in rows) + 2
    header = "Условие".ljust(width) + "повт.".rjust(6) + "откл.".rjust(7) \
        + "".join(title.rjust(max(8, len(title) + 2)) for title, _ in columns)
    print("\nЧастоты ключевых нейронов, Гц (среднее за повторы; Л/П — сторона мухи, D-011)")
    print(header)
    print("-" * len(header))
    for condition, rates, n_trials in rows:
        stimulated = set(condition_stimulated(condition, groups))
        responded = int(sum(1 for i in np.flatnonzero(rates >= ACTIVE_HZ) if i not in stimulated))
        line = condition.name_ru.ljust(width) + str(n_trials).rjust(6) + str(responded).rjust(7)
        for title, index in columns:
            line += f"{rates[index]:.1f}".rjust(max(8, len(title) + 2))
        print(line)
    print(f"«откл.» — нейроны со средней частотой ≥ {ACTIVE_HZ:g} Гц (D-018), не считая стимулируемых.")


def condition_stimulated(condition: Condition, groups: dict) -> list[int]:
    keys = [condition.excite_group] + ([condition.excite2_group] if condition.excite2_group else [])
    return [i for key in keys for i in groups["stimuli"][key]["indices"]]


# ---------- команда ----------

def run_reference(trials: int | None = None, jobs: int = 6, only: list[str] | None = None,
                  out_dir: Path = REFERENCE_DIR) -> int:
    started = time.monotonic()
    print("FlyBrain: эталонные прогоны оригинального кода Shiu (Brian2)\n")
    try:
        config = load_reference_config()
        n_trials = trials or int(config["trials"])
        duration_ms = float(config["duration_ms"])
        conditions = config["conditions"]
        if only:
            unknown = sorted(set(only) - {condition.key for condition in conditions})
            if unknown:
                raise ReferenceError(f"нет таких условий: {', '.join(unknown)}. "
                                     f"Есть: {', '.join(c.key for c in conditions)}")
            conditions = [condition for condition in conditions if condition.key in only]
        paths = check_inputs()
        print("[OK] model.py, список нейронов и связи Shiu на месте, контрольные суммы совпадают")
        groups = processed.load_groups()
        root_ids = pd.read_csv(paths["Completeness_783.csv"], index_col=0).index.to_numpy(np.int64)
        out_dir.mkdir(parents=True, exist_ok=True)
        for leftover in out_dir.rglob("*.part"):
            leftover.unlink()
        connectivity = {"original": paths["Connectivity_783.parquet"]}
        if any(condition.connectivity == "shuffled" for condition in conditions):
            connectivity["shuffled"] = ensure_shuffled(paths["Connectivity_783.parquet"], out_dir,
                                                       int(config["shuffle_seed"]))
        connectivity_sha = {kind: sha256_of(path) for kind, path in connectivity.items()}
        model_sha = sha256_of(paths["model.py"])

        tasks, done = [], {}
        for condition in conditions:
            if condition.connectivity not in connectivity:
                raise ReferenceError(f"условие {condition.key}: неизвестные связи «{condition.connectivity}»")
            exc = resolve_indices(groups, root_ids, condition.excite_group)
            exc2 = resolve_indices(groups, root_ids, condition.excite2_group) if condition.excite2_group else []
            trial_dir = out_dir / "trials" / condition.key
            trial_dir.mkdir(parents=True, exist_ok=True)
            done[condition.key] = 0
            for trial in range(n_trials):
                meta = json.dumps({
                    "format": TRIAL_FORMAT, "condition": condition.key, "trial": trial,
                    "seed": trial_seed(condition.key, trial), "duration_ms": duration_ms,
                    "r_poi_hz": condition.excite_hz, "r_poi2_hz": condition.excite2_hz,
                    "exc": exc, "exc2": exc2, "model_sha256": model_sha,
                    "connectivity_sha256": connectivity_sha[condition.connectivity],
                }, sort_keys=True)
                out_file = trial_dir / f"trial_{trial:02d}.npz"
                if out_file.exists():
                    if read_trial_meta(out_file) != meta:
                        raise ReferenceError(
                            f"в {_where(out_file)} лежит повтор с другими настройками. "
                            f"Удалите папку {_where(trial_dir)} и запустите снова")
                    done[condition.key] += 1
                    continue
                tasks.append({
                    "key": condition.key, "trial": trial, "seed": trial_seed(condition.key, trial),
                    "model_path": str(paths["model.py"]), "path_comp": str(paths["Completeness_783.csv"]),
                    "path_con": str(connectivity[condition.connectivity]), "exc": exc, "exc2": exc2,
                    "r_poi_hz": condition.excite_hz, "r_poi2_hz": condition.excite2_hz,
                    "duration_ms": duration_ms, "meta": meta, "out_file": str(out_file),
                })
    except ReferenceError as exc:
        print(f"\n[ОШИБКА] {exc}", file=sys.stderr)
        return 1

    total = n_trials * len(conditions)
    print(f"Условий: {len(conditions)}, повторов по {duration_ms / 1000:g} с: {n_trials} на условие, "
          f"всего {total}; уже готово {total - len(tasks)}, осталось {len(tasks)}")
    print(f"Результаты: {_where(out_dir)}")
    stats = []
    if tasks:
        jobs = max(1, min(jobs, len(tasks)))
        free_gb = _free_memory_gb()
        print(f"Параллельных повторов: {jobs}; каждому нужно около {GB_PER_JOB:g} ГБ памяти"
              + (f", свободно сейчас {free_gb:.1f} ГБ" if free_gb is not None else ""))
        if free_gb is not None and free_gb < jobs * GB_PER_JOB:
            print(f"[ВНИМАНИЕ] свободной памяти может не хватить. Закройте лишние программы "
                  f"или уменьшите число повторов: --jobs {max(1, int(free_gb // GB_PER_JOB))}")
        print("Прервать можно в любой момент (Ctrl+C); готовые повторы сохранятся, "
              "продолжение — той же командой.\n", flush=True)
        status = _run_pool(tasks, jobs, conditions, done, n_trials, stats)
        if status:
            return status

    with open(NAMES_CONFIG, encoding="utf-8") as handle:
        names = yaml.safe_load(handle)
    neurons = processed.load_neurons()
    shiu = _load_shiu(str(paths["model.py"]))
    i2flyid = dict(enumerate(root_ids.tolist()))
    rows, rate_columns, timings, memories, codegens = [], {}, {}, {}, set()
    for condition in conditions:
        trial_files = sorted((out_dir / "trials" / condition.key).glob("trial_*.npz"))[:n_trials]
        trials_data = [load_trial(path) for path in trial_files]
        rates = condition_rates(trials_data, len(root_ids), duration_ms / 1000)
        rows.append((condition, rates, len(trials_data)))
        rate_columns[condition.key] = rates
        timings[condition.key] = [float(trial["seconds"]) for trial in trials_data]
        memories[condition.key] = max(float(trial["peak_memory_gb"]) for trial in trials_data)
        codegens.update(str(trial["codegen"]) for trial in trials_data)
        # Все повторы условия — одним файлом в формате Shiu (одна строка — один спайк).
        spike_trains = []
        for trial in trials_data:
            bounds = np.flatnonzero(np.diff(trial["neuron"])) + 1
            spike_trains.append({int(chunk[0]): times for chunk, times in
                                 zip(np.split(trial["neuron"], bounds), np.split(trial["t"], bounds))
                                 if len(chunk)})
        frame = shiu.construct_dataframe(spike_trains, condition.key, i2flyid)
        frame.to_parquet(out_dir / f"{condition.key}.parquet", compression="brotli")

    table = pd.DataFrame(rate_columns)
    table.insert(0, "flywire_id", root_ids.astype(str))
    table = table[(table[list(rate_columns)] > 0).any(axis=1)]
    table.to_csv(out_dir / RATES_FILE, index_label="index", float_format="%.4g", encoding="utf-8")
    _write_info(out_dir, config, conditions, n_trials, duration_ms, model_sha, connectivity_sha,
                timings, memories, sorted(codegens))
    print_summary(config, rows, groups, neurons, names)
    print(f"\nФайлы: по одному .parquet на условие (формат Shiu), частоты всех нейронов — {RATES_FILE}, "
          f"справка — {INFO_FILE}")
    print(f"Готово за {_hms(time.monotonic() - started)}.")
    return 0


def _run_pool(tasks: list[dict], jobs: int, conditions: list[Condition], done: dict[str, int],
              n_trials: int, stats: list[dict]) -> int:
    names = {condition.key: condition.name_ru for condition in conditions}
    total, remaining = n_trials * len(conditions), len(tasks)
    finished_before = total - remaining
    started = time.monotonic()
    # Один повтор — один процесс (max_tasks_per_child=1): память Brian2 гарантированно освобождается.
    executor = ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1)
    pending: set[Future] = set()
    queue = list(tasks)
    try:
        while queue or pending:
            while queue and len(pending) < jobs:
                pending.add(executor.submit(run_one_trial, queue.pop(0)))
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                result = future.result()
                stats.append(result)
                done[result["key"]] += 1
                count = finished_before + len(stats)
                elapsed = time.monotonic() - started
                left = elapsed / len(stats) * (total - count)
                if result["codegen"] != "cython":
                    print("[ВНИМАНИЕ] Brian2 считает без компилятора C++ (запасной способ numpy) — "
                          "это в разы медленнее. Проверьте: uv run flybrain doctor", flush=True)
                print(f"[{count:>3}/{total}] {names[result['key']]}: повтор {result['trial'] + 1} готов "
                      f"за {result['seconds']:.0f} с, память {result['peak_memory_gb']:.1f} ГБ "
                      f"({done[result['key']]}/{n_trials}) · прошло {_hms(elapsed)} · осталось ≈ {_hms(left)}",
                      flush=True)
    except KeyboardInterrupt:
        executor.shutdown(wait=False, cancel_futures=True)
        print(f"\nПрервано. Сохранено повторов: {finished_before + len(stats)} из {total}. "
              "Чтобы продолжить, выполните ту же команду ещё раз.", flush=True)
        return 130
    except BrokenProcessPool:
        _stop_workers(executor)
        print(f"\n[ОШИБКА] процесс расчёта неожиданно завершился (часто — не хватило памяти, или нажат Ctrl+C). "
              f"Сохранено повторов: {finished_before + len(stats)} из {total}. Запустите ту же команду ещё раз, "
              "можно с меньшим --jobs.", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:  # ошибка внутри повтора: остальные повторы не теряются
        _stop_workers(executor)
        hint = ""
        if "Visual Studio" in str(exc) or "compiler" in str(exc).lower():
            hint = ("\nBrian2 не нашёл компилятор C++ (он нужен, когда встречается новая частота стимула). "
                    "Запускайте команду из PowerShell или cmd: в Git Bash Windows не передаёт программам "
                    "переменную ProgramFiles(x86), по которой ищется Visual Studio. Проверка: uv run flybrain doctor")
        print(f"\n[ОШИБКА] повтор завершился с ошибкой: {type(exc).__name__}: {exc}{hint}\n"
              f"Сохранено повторов: {finished_before + len(stats)} из {total}.", file=sys.stderr, flush=True)
        return 1
    executor.shutdown()
    return 0


def _stop_workers(executor: ProcessPoolExecutor) -> None:
    """Сразу остановить процессы-повторы: иначе после ошибки команда может так и не завершиться.

    Готовые повторы не теряются: файл повтора появляется под своим именем только целиком (`.part` → `os.replace`).
    """
    for process in list((executor._processes or {}).values()):
        process.terminate()
    # Ждём, пока остановленные процессы закроются: без этого служебный поток пула печатает лишнюю ошибку.
    executor.shutdown(wait=True, cancel_futures=True)


def _free_memory_gb() -> float | None:
    if platform.system() != "Windows":
        return None

    class Status(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = Status()
    status.dwLength = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 1024**3


def _write_info(out_dir: Path, config: dict, conditions: list[Condition], n_trials: int, duration_ms: float,
                model_sha: str, connectivity_sha: dict, timings: dict, memories: dict,
                codegens: list[str]) -> None:
    import brian2
    import Cython

    info = {
        "written_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code": "model.py Shiu, коммит 91bdd1e, без изменений (D-006)",
        "model_sha256": model_sha,
        "connectivity_sha256": connectivity_sha,
        "shuffle_seed": int(config["shuffle_seed"]),
        "seed_rule": "zlib.crc32('<условие>:<номер повтора>') → brian2.seed() перед каждым повтором",
        "trials_per_condition": n_trials,
        "duration_ms": duration_ms,
        "brian2": brian2.__version__,
        "cython": Cython.__version__,
        "python": platform.python_version(),
        "codegen": codegens,
        "conditions": {
            condition.key: {
                "name_ru": condition.name_ru,
                "excite": {"group": condition.excite_group, "hz": condition.excite_hz},
                "excite2": ({"group": condition.excite2_group, "hz": condition.excite2_hz}
                            if condition.excite2_group else None),
                "connectivity": condition.connectivity,
                "honesty_ru": condition.honesty_ru,
                "trial_seconds_mean": round(float(np.mean(timings[condition.key])), 1),
                "peak_memory_gb": round(memories[condition.key], 2),
            }
            for condition in conditions
        },
    }
    with open(out_dir / INFO_FILE, "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=False, indent=1)
