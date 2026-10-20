import hashlib
import re

import pytest

from flybrain import fetch
from flybrain.fetch import FetchError, Source, fetch_one, load_sources


def _make_source(tmp_path, payload: bytes, sha256: str | None = None) -> Source:
    origin = tmp_path / "origin.bin"
    origin.write_bytes(payload)
    return Source(
        name="origin.bin",
        description="проверочный файл",
        url=origin.as_uri(),
        sha256=sha256 or hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        dest=tmp_path / "raw" / "origin.bin",
    )


def test_sources_file_is_complete():
    sources = load_sources()
    names = {source.name for source in sources}
    assert {
        "Completeness_783.csv",
        "Connectivity_783.parquet",
        "model.py",
        "utils.py",
        "figures.ipynb",
        "sez_neurons.pickle",
        "Supplemental_file1_neuron_annotations.tsv",
    } <= names
    for source in sources:
        assert re.fullmatch(r"[0-9a-f]{64}", source.sha256)
        assert re.search(r"/[0-9a-f]{40}/", source.url), "адрес должен быть закреплён коммитом"
        assert "data/raw" in source.dest.as_posix()


def test_download_then_skip(tmp_path, monkeypatch):
    source = _make_source(tmp_path, b"hello fly brain")
    assert fetch_one(source) == "скачано"
    assert source.dest.read_bytes() == b"hello fly brain"

    def fail(*args, **kwargs):
        raise AssertionError("повторный запуск не должен ничего качать")

    monkeypatch.setattr(fetch, "_download", fail)
    assert fetch_one(source) == "уже было"


def test_wrong_checksum_saves_nothing(tmp_path):
    source = _make_source(tmp_path, b"data", sha256="0" * 64)
    with pytest.raises(FetchError, match="не совпала"):
        fetch_one(source)
    assert not source.dest.exists()
    assert not list(source.dest.parent.glob("*.part"))
