"""`flybrain fetch-data`: скачивание данных с проверкой SHA-256 (D-009, docs/DATA.md)."""

import hashlib
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import yaml

from flybrain.paths import CONFIG_DIR, PROJECT_ROOT

SOURCES_FILE = CONFIG_DIR / "sources.yaml"
CHUNK = 1024 * 1024


class FetchError(Exception):
    """Данные не удалось получить или они не прошли проверку."""


@dataclass(frozen=True)
class Source:
    name: str
    description: str
    url: str
    sha256: str
    size_bytes: int
    dest: Path


def load_sources(path: Path = SOURCES_FILE, root: Path = PROJECT_ROOT) -> list[Source]:
    with open(path, encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)["files"]
    return [
        Source(
            name=entry["name"],
            description=entry["description"],
            url=entry["url"],
            sha256=entry["sha256"].lower(),
            size_bytes=int(entry["size_bytes"]),
            dest=root / entry["dest"],
        )
        for entry in entries
    ]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _megabytes(size: float) -> str:
    return f"{size / 1024**2:.1f} МБ"


def _download(source: Source, target: Path) -> str:
    """Скачивает во временный файл `target` и возвращает SHA-256 скачанного."""
    digest = hashlib.sha256()
    done = 0
    started = time.monotonic()
    last_print = 0.0
    request = urllib.request.Request(source.url, headers={"User-Agent": "flybrain-fetch-data"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, open(target, "wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last_print > 0.5 and source.size_bytes > 100_000:
                    last_print = now
                    percent = min(99, 100 * done // max(source.size_bytes, 1))
                    print(f"\r    скачано {_megabytes(done)} (~{percent}%)", end="", flush=True)
    except OSError as exc:
        raise FetchError(f"не удалось скачать {source.url}: {exc}") from exc
    if source.size_bytes > 100_000:
        print(f"\r    скачано {_megabytes(done)} за {time.monotonic() - started:.0f} с" + " " * 10)
    return digest.hexdigest()


def fetch_one(source: Source) -> str:
    """Возвращает «уже было» или «скачано»; при несовпадении суммы бросает FetchError."""
    if source.dest.exists():
        if sha256_of(source.dest) == source.sha256:
            return "уже было"
        print(f"    файл {source.dest.name} повреждён или изменён — скачиваю заново")
    source.dest.parent.mkdir(parents=True, exist_ok=True)
    partial = source.dest.with_name(source.dest.name + ".part")
    actual = _download(source, partial)
    if actual != source.sha256:
        partial.unlink(missing_ok=True)
        raise FetchError(
            f"контрольная сумма {source.name} не совпала.\n"
            f"    ожидалась {source.sha256}\n    получена  {actual}\n"
            "    Файл у автора изменился или скачался с ошибкой; ничего не сохранено."
        )
    partial.replace(source.dest)
    return "скачано"


def _summary(sources: list[Source]) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    by_name = {source.name: source.dest for source in sources}
    neurons = pd.read_csv(by_name["Completeness_783.csv"], index_col=0)
    pairs = pq.read_table(by_name["Connectivity_783.parquet"], columns=["Connectivity"])
    synapses = int(pairs.column("Connectivity").to_pandas().sum())
    annotations = pd.read_csv(
        by_name["Supplemental_file1_neuron_annotations.tsv"],
        sep="\t", usecols=["root_id"], dtype={"root_id": "int64"},
    )
    in_model = int(annotations["root_id"].isin(neurons.index).sum())

    def fmt(number: int) -> str:
        return f"{number:,}".replace(",", " ")

    print("\nСводка:")
    print(f"  нейронов: {fmt(len(neurons))}")
    print(f"  пар связей: {fmt(pairs.num_rows)}")
    print(f"  синапсов: {fmt(synapses)}")
    print(f"  аннотаций: {fmt(len(annotations))}, из них нейронов модели {fmt(in_model)}")


def run_fetch_data() -> int:
    sources = load_sources()
    print("FlyBrain: скачивание данных в data/raw/\n")
    try:
        for source in sources:
            print(f"{source.name} — {source.description}")
            outcome = fetch_one(source)
            print(f"  [OK] {outcome}, контрольная сумма совпадает ({source.sha256[:12]}…)")
        _summary(sources)
    except FetchError as exc:
        print(f"\n[ОШИБКА] {exc}", file=sys.stderr)
        return 1
    print("\nВсе контрольные суммы совпадают.")
    return 0
