"""Формы нейронов (C12b, D-037): перекладка скелетов, чтение, проверка координат, ответ сервера, докачка."""

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from flybrain import fetch, processed, skeletons
from flybrain.fetch import Source, fetch_one
from flybrain.server import create_app
from flybrain.skeletons import HEADER, SkeletonStore, check_alignment, prepare, write_info


def _neurons(root_ids, points, sources=None) -> pd.DataFrame:
    xyz = np.array(points, dtype=np.float32)
    return pd.DataFrame({
        "root_id": np.array(root_ids, dtype=np.int64),
        "x_um": xyz[:, 0], "y_um": xyz[:, 1], "z_um": xyz[:, 2],
        "pos_source": sources or ["soma"] * len(root_ids),
    })


def _write_skeletons(path, rows, row_group_size=4) -> None:
    """rows: (нейрон, node_id, parent_id, x, y, z в нм) — как в файле FlyWire."""
    neuron, node, parent, x, y, z = map(list, zip(*rows))
    table = pa.table({
        "node_id": pa.array(node, pa.int32()),
        "parent_id": pa.array(parent, pa.int64()),
        "radius": pa.array([0] * len(rows), pa.int32()),
        "x": pa.array(x, pa.float32()), "y": pa.array(y, pa.float32()), "z": pa.array(z, pa.float32()),
        "neuron": pa.array(neuron, pa.int64()),
    })
    pq.write_table(table, path, row_group_size=row_group_size)


# Нейрон 20 разрезан между блоками и пачками, 10 стоит после него, 99 нет в модели, у 40 формы нет.
ROWS = [
    (20, 1, -1, 300_000, 100_000, 50_000),
    (20, 2, 1, 301_000, 100_000, 50_000),
    (99, 1, -1, 1, 1, 1),
    (20, 4, 2, 302_000, 101_000, 50_000),
    (10, 7, 9, 500_000, 200_000, 10_000),
    (20, 3, 2, 301_000, 99_000, 50_000),
    (10, 9, -1, 501_000, 200_000, 10_000),
]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(skeletons, "BATCH_ROWS", 3)  # несколько пачек даже на семи строках
    source = tmp_path / "sk.parquet"
    _write_skeletons(source, ROWS)
    neurons = _neurons([10, 20, 30, 40], [[-501, 200, 10], [-300, 100, 50], [0, 0, 0], [np.nan] * 3])
    out = tmp_path / "skeletons"
    info = prepare(source, neurons, out, source_sha256="abc", log=lambda *a, **k: None)
    write_info(info, out)
    opened, why_not = SkeletonStore.open(neurons, out)
    assert why_not is None
    return opened, neurons, info


def test_prepare_counts_and_skips_foreign_neurons(store):
    opened, _, info = store
    assert opened.count.tolist() == [2, 4, 0, 0]
    assert (info["nodes"], info["nodes_not_in_model"], info["neurons_with_skeleton"]) == (6, 1, 2)
    assert not opened.has(2) and not opened.has(3) and opened.has(1)


def test_coordinates_in_micrometres_with_mirrored_x(store):
    opened, _, _ = store
    skeleton = opened.get(1)
    # порядок строк файла внутри нейрона сохраняется; X отражён, как у облака (D-011)
    np.testing.assert_allclose(skeleton.positions, [[-300, 100, 50], [-301, 100, 50], [-302, 101, 50], [-301, 99, 50]])


def test_parents_become_positions(store):
    opened, _, _ = store
    assert opened.get(1).parents.tolist() == [-1, 0, 1, 1]  # node 4 и node 3 — дети node 2
    assert opened.get(0).parents.tolist() == [1, -1]  # родитель может стоять ниже ребёнка


def test_wire_format(store):
    opened, neurons, _ = store
    data = opened.wire(1, np.array([-301, 100, 53], dtype=np.float32))
    header = np.frombuffer(data[:HEADER.itemsize], dtype=HEADER)[0]
    assert (header["version"], header["count"]) == (1, 4)
    assert header["distance_um"] == pytest.approx(3.0)
    positions = np.frombuffer(data[HEADER.itemsize:HEADER.itemsize + 48], dtype="<f4").reshape(4, 3)
    parents = np.frombuffer(data[HEADER.itemsize + 48:], dtype="<i4")
    assert positions[2].tolist() == [-302, 101, 50]
    assert parents.tolist() == [-1, 0, 1, 1]
    assert np.isnan(np.frombuffer(opened.wire(1, None)[8:12], dtype="<f4")[0])


def test_alignment_check_catches_unmirrored_x(store):
    opened, neurons, _ = store
    check = check_alignment(opened, neurons, log=lambda *a, **k: None)
    assert check["all"]["neurons"] == 2
    assert check["all"]["max_um"] == pytest.approx(0.0)
    assert check["control_not_mirrored"]["median_um"] > 500
    assert check["passed"]
    shifted = neurons.assign(x_um=-neurons["x_um"])  # как если бы облако забыло отражение
    assert not check_alignment(opened, shifted, log=lambda *a, **k: None)["passed"]


def test_store_refuses_other_neuron_table(tmp_path, store):
    _, neurons, _ = store
    out = tmp_path / "skeletons"
    assert SkeletonStore.open(neurons.iloc[::-1], out) == (None, "скелеты подготовлены для другой таблицы нейронов")
    assert SkeletonStore.open(neurons, tmp_path / "nothing") == (None, "скелеты не скачаны")


def test_server_without_skeletons_works_as_before(tmp_path):
    with TestClient(create_app(skeleton_dir=tmp_path / "none")) as client:
        assert client.get("/api/meta").json()["skeletons"] == {"available": False, "reason": "скелеты не скачаны"}
        answer = client.get("/api/neuron/0/skeleton")
        assert answer.status_code == 404
        assert client.get("/api/neuron/0").status_code == 200


def test_server_serves_skeleton_of_real_neuron(tmp_path):
    neurons = processed.load_neurons()
    mn9 = int(np.flatnonzero(neurons["root_id"].to_numpy() == 720575940660219265)[0])
    x, y, z = neurons.loc[mn9, ["x_um", "y_um", "z_um"]].astype(float)
    nm = lambda value: value * 1000  # noqa: E731
    source = tmp_path / "sk.parquet"
    _write_skeletons(source, [
        (720575940660219265, 1, -1, nm(-x), nm(y), nm(z)),  # в файле X не отражён
        (720575940660219265, 2, 1, nm(-x) + 2000, nm(y), nm(z)),
    ])
    out = tmp_path / "skeletons"
    write_info(prepare(source, neurons, out, log=lambda *a, **k: None), out)
    with TestClient(create_app(skeleton_dir=out)) as client:
        assert client.get("/api/meta").json()["skeletons"]["available"] is True
        data = client.get(f"/api/neuron/{mn9}/skeleton").content
        header = np.frombuffer(data[:HEADER.itemsize], dtype=HEADER)[0]
        assert header["count"] == 2
        assert header["distance_um"] == pytest.approx(0.0, abs=1e-3)
        assert client.get(f"/api/neuron/{mn9 + 1}/skeleton").status_code == 404
        assert client.get("/api/neuron/999999999/skeleton").status_code == 404


class _RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    cut_after = None  # обрывает первую отдачу после стольких байт
    ranges = []

    def do_GET(self):  # noqa: N802
        rng = self.headers.get("Range")
        type(self).ranges.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        body = self.payload[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if type(self).cut_after is not None:
            self.wfile.write(body[:type(self).cut_after])
            type(self).cut_after = None
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_interrupted_download_resumes(tmp_path, monkeypatch):
    payload = bytes(range(256)) * 4000
    _RangeHandler.payload, _RangeHandler.cut_after, _RangeHandler.ranges = payload, 300_000, []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(fetch, "CHUNK", 64 * 1024)
    try:
        source = Source(name="big.bin", description="", url=f"http://127.0.0.1:{server.server_port}/big.bin",
                        sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload),
                        dest=tmp_path / "raw" / "big.bin")
        with pytest.raises(fetch.FetchError, match="продолжит"):
            fetch_one(source)
        part = source.dest.with_name("big.bin.part")
        have = part.stat().st_size
        assert 0 < have < len(payload)
        assert fetch_one(source) == "скачано"
        assert source.dest.read_bytes() == payload
        assert _RangeHandler.ranges == [None, f"bytes={have}-"]  # второй раз — только недостающий хвост
    finally:
        server.shutdown()


def test_skeleton_source_is_pinned():
    (source,) = fetch.load_sources(section="optional")
    assert source.name == "sk_lod1_783_healed_ds2.parquet"
    assert source.size_bytes == 5_355_543_468
    assert len(source.sha256) == 64 and "zenodo.org/api/records/10877326/" in source.url
    assert "data/raw" in source.dest.as_posix()


def test_complete_partial_file_is_not_downloaded_again(tmp_path, monkeypatch):
    payload = b"skeleton bytes" * 100
    dest = tmp_path / "raw" / "big.bin"
    dest.parent.mkdir(parents=True)
    dest.with_name("big.bin.part").write_bytes(payload)
    source = Source(name="big.bin", description="", url="http://127.0.0.1:9/never", size_bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(), dest=dest)

    def fail(*args, **kwargs):
        raise AssertionError("целиком скачанный .part не нужно качать заново")

    monkeypatch.setattr(fetch, "_download", fail)
    assert fetch_one(source) == "докачано раньше"
    assert dest.read_bytes() == payload
