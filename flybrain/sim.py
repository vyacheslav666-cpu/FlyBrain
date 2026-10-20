"""`flybrain sim`: прогон движка PyTorch на настоящем коннектоме и топ активных нейронов (C5)."""

import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from flybrain import processed
from flybrain.engine import Engine, EngineError
from flybrain.paths import CONFIG_DIR, PROCESSED_DIR

NAMES_CONFIG = CONFIG_DIR / "names_ru.yaml"
# Ключевые «исполнители» из статьи и опыта с тенью — печатаются всегда (группы readouts, C3).
KEY_READOUTS = ("MN9", "aDN1", "aDN2", "aBN1", "GF")
ACTIVE_HZ = 1.0  # «активный нейрон» — средняя частота не меньше 1 Гц (D-018)
PROGRESS_BLOCKS = 56  # прогресс печатается примерно каждые 100 мс жизни мухи (56 блоков по 1,8 мс)


def _number(value: float, digits: int = 1) -> str:
    """Число по-русски: десятичная запятая, тысячи через пробел."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def _device_name(engine: Engine) -> str:
    if engine.device.type == "cuda":
        return f"видеокарта {torch.cuda.get_device_name(engine.device)}"
    return "процессор (видеокарта не найдена или не выбрана; это в десятки раз медленнее)"


def run_sim(stimuli: list[str], hz: float, trials: int, duration_ms: float, device: str = "auto",
            seed: int | None = None, top: int = 20) -> int:
    print("FlyBrain: симуляция на движке PyTorch (модель Shiu, параметры D-005)\n")
    if not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists():
        print("[ОШИБКА] нет подготовленных данных. Сначала выполните:\n"
              "  uv run flybrain fetch-data\n  uv run flybrain build-data", file=sys.stderr)
        return 1
    groups = processed.load_groups()
    unknown = [key for key in stimuli if key not in groups["stimuli"]]
    if unknown:
        print(f"[ОШИБКА] нет такого стимула: {', '.join(unknown)}. Есть: {', '.join(groups['stimuli'])}",
              file=sys.stderr)
        return 1
    if trials < 1 or duration_ms <= 0:
        print("[ОШИБКА] число повторов и длительность должны быть больше нуля", file=sys.stderr)
        return 1

    started = time.monotonic()
    connectivity = processed.load_connectivity()
    neurons = processed.load_neurons()
    with open(NAMES_CONFIG, encoding="utf-8") as handle:
        names = yaml.safe_load(handle)
    try:
        engine = Engine(connectivity["indptr"], connectivity["indices"], connectivity["weight"],
                        n_trials=trials, device=device, seed=seed)
        stimulated = []
        for key in stimuli:
            engine.set_stimulus(key, groups["stimuli"][key]["indices"], hz)
            stimulated.extend(groups["stimuli"][key]["indices"])
        engine.prepare()
    except (EngineError, RuntimeError) as exc:
        message = str(exc).splitlines()[0]
        if "out of memory" in message.lower():
            message = f"не хватило памяти видеокарты на {trials} повторов; уменьшите --trials"
        print(f"[ОШИБКА] {message}", file=sys.stderr)
        return 1
    setup_seconds = time.monotonic() - started

    print(f"Сеть: {_number(engine.n_neurons, 0)} нейронов, {_number(len(connectivity['indices']), 0)} пар связей")
    print(f"Считает: {_device_name(engine)}")
    for key in stimuli:
        group = groups["stimuli"][key]
        print(f"Стимул: {group['name_ru']} — {group['count']} нейронов, {_number(hz, 0)} Гц "
              "(случайные импульсы, прямое включение нейронов)")
        if group.get("honesty_ru"):
            print(f"  [ЧЕСТНОСТЬ] {group['honesty_ru']}")
    together = " (считаются одновременно)" if trials > 1 else ""
    print(f"Повторов: {trials} по {_number(duration_ms / 1000, 3).rstrip('0').rstrip(',')} с{together}. "
          f"Подготовка заняла {_number(setup_seconds)} с.\n", flush=True)

    total_steps = int(round(duration_ms / engine.params.dt))
    chunk = PROGRESS_BLOCKS * engine.block_steps
    run_started = time.monotonic()
    done = 0
    while done < total_steps:
        steps = min(chunk, total_steps - done)
        engine.run_steps(steps)
        engine.synchronize()
        done += steps
        print(f"\r  посчитано {_number(engine.time_ms, 0)} из {_number(duration_ms, 0)} мс "
              f"({100 * done // total_steps} %)", end="", flush=True)
    rates = engine.rates_hz()  # (повторы, нейроны)
    run_seconds = time.monotonic() - run_started
    print("\n")

    _print_top(rates, neurons, names, set(stimulated), top, trials)
    _print_readouts(rates, groups, neurons, names)
    mean = rates.mean(axis=0)
    stim_set = sorted(set(stimulated))
    active = mean >= ACTIVE_HZ
    others = int(active.sum() - active[stim_set].sum())
    print(f"\nСтимулируемые нейроны ({len(stim_set)}): в среднем {_number(mean[stim_set].mean())} Гц.")
    print(f"Активных нейронов (≥ {ACTIVE_HZ:g} Гц в среднем, D-018): {int(active.sum())}, "
          f"из них не стимулируемых {others}.")
    simulated = trials * duration_ms / 1000
    print(f"\nскорость: {_number(simulated / run_seconds, 2)}× реального времени "
          f"({trials} × {_number(duration_ms / 1000, 3).rstrip('0').rstrip(',')} с жизни мухи за "
          f"{_number(run_seconds)} с счёта; в пересчёте на один повтор)")
    return 0


def _label(row: pd.Series, names: dict) -> str:
    cell_type = None if pd.isna(row["cell_type"]) else str(row["cell_type"])
    if cell_type is None:
        return "без типа"
    entry = names["cell_types"].get(cell_type)
    return entry["short"] if entry else cell_type


def _print_top(rates: np.ndarray, neurons: pd.DataFrame, names: dict, stimulated: set[int], top: int,
               trials: int) -> None:
    mean = rates.mean(axis=0)
    order = [int(i) for i in np.argsort(-mean, kind="stable") if int(i) not in stimulated and mean[i] > 0][:top]
    print(f"Топ-{top} активных нейронов (без стимулируемых), средняя частота за {trials} повт.:")
    header = f"{'№':>3}  {'Гц':>6}  {'имя':<10} {'тип FlyWire':<14} {'сторона мухи':<13} {'номер FlyWire':<19} класс"
    print(header)
    print("-" * (len(header) + 20))
    for place, index in enumerate(order, start=1):
        row = neurons.iloc[index]
        side = "—" if pd.isna(row["side"]) else names["sides"].get(row["side"], row["side"])
        super_class = "missing" if pd.isna(row["super_class"]) else row["super_class"]
        cell_type = "—" if pd.isna(row["cell_type"]) else str(row["cell_type"])
        print(f"{place:>3}  {_number(mean[index]):>6}  {_label(row, names):<10} {cell_type:<14} {side:<13} "
              f"{int(row['root_id']):<19} {names['super_classes'].get(super_class, super_class)}")
    if not order:
        print("  (никто, кроме стимулируемых нейронов, не выстрелил)")


def _print_readouts(rates: np.ndarray, groups: dict, neurons: pd.DataFrame, names: dict) -> None:
    mean = rates.mean(axis=0)
    parts = []
    for key in KEY_READOUTS:
        group = groups["readouts"].get(key)
        if group is None:
            continue
        sides = []
        for index in group["indices"]:
            side = neurons.at[index, "side"]
            sides.append(f"{names['sides'].get(side, side)} {_number(mean[index])}")
        parts.append(f"{key}: " + ", ".join(sides))
    print("\nКлючевые нейроны, Гц (сторона мухи, D-011): " + "; ".join(parts))
