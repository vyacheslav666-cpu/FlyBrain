"""Тесты страницы и сервера (коммит C7): формат данных для браузера, ответы сервера, сторонние файлы."""

import hashlib
import struct

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from flybrain import processed, webdata
from flybrain.paths import PROCESSED_DIR, PROJECT_ROOT
from flybrain.server import create_app

LEGEND = [{"key": "optic", "color": "#4f8dff"}, {"key": "motor", "color": "#ffe14d"}]
NAMES = {"super_classes": {"optic": "зрительная доля", "motor": "моторный"}}


def _tiny_neurons() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x_um": [-100.0, -200.0, np.nan],
            "y_um": [10.0, 20.0, np.nan],
            "z_um": [1.0, 2.0, np.nan],
            "super_class": ["motor", "optic", None],
            "pos_source": ["soma", "anchor", "none"],
        }
    )


def test_cloud_layout_is_header_positions_classes():
    blob = webdata.build_cloud(_tiny_neurons(), LEGEND)
    count, version = struct.unpack_from("<II", blob)
    assert (count, version) == (3, webdata.CLOUD_FORMAT_VERSION)
    assert len(blob) == 8 + 3 * 12 + 3
    xyz = np.frombuffer(blob, dtype="<f4", count=9, offset=8).reshape(3, 3)
    assert xyz[0].tolist() == [-100.0, 10.0, 1.0]
    assert xyz[2].tolist() == [0.0, 0.0, 0.0]  # нет координат — нули
    codes = np.frombuffer(blob, dtype=np.uint8, offset=8 + 36)
    assert codes.tolist() == [1, 0, webdata.NO_POSITION]


def test_unknown_class_is_an_error_not_silently_hidden():
    neurons = _tiny_neurons()
    neurons.loc[0, "super_class"] = "mystery"
    with pytest.raises(ValueError, match="mystery"):
        webdata.build_cloud(neurons, LEGEND)


def test_meta_counts_and_center():
    meta = webdata.build_meta(_tiny_neurons(), LEGEND, NAMES)
    assert (meta["total"], meta["shown"], meta["hidden"]) == (3, 2, 1)
    assert (meta["soma"], meta["anchor"]) == (1, 1)
    assert meta["center"] == [-150.0, 15.0, 1.5]
    assert [c["count"] for c in meta["classes"]] == [1, 1]
    assert meta["classes"][1]["name"] == "моторный"


def test_legend_file_covers_all_classes_of_names_file():
    legend = webdata.load_legend()
    names = webdata.load_yaml(webdata.NAMES_FILE)["super_classes"]
    keys = {entry["key"] for entry in legend}
    assert keys <= set(names)
    assert len(keys) == len(legend)
    for entry in legend:
        assert len(entry["color"]) == 7 and entry["color"].startswith("#")


def test_vendor_files_match_recorded_checksums():
    # Чужие файлы three.js не должны меняться незаметно (web/vendor/README.md).
    readme = (PROJECT_ROOT / "web" / "vendor" / "README.md").read_text(encoding="utf-8")
    for name in ("three.module.js", "OrbitControls.js", "LICENSE-three.txt"):
        digest = hashlib.sha256((PROJECT_ROOT / "web" / "vendor" / name).read_bytes()).hexdigest()
        assert digest in readme, f"контрольная сумма {name} не совпала с записью в README"


needs_data = pytest.mark.skipif(
    not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists(), reason="нет data/processed — запустите build-data"
)


@needs_data
def test_server_serves_page_data_and_scripts():
    with TestClient(create_app()) as client:
        assert client.get("/api/ping").json()["app"] == "flybrain"
        page = client.get("/")
        assert page.status_code == 200 and "FlyBrain" in page.text

        meta = client.get("/api/meta").json()
        assert (meta["total"], meta["shown"], meta["hidden"]) == (138_639, 138_625, 14)
        assert sum(c["count"] for c in meta["classes"]) == meta["shown"]

        cloud = client.get("/api/cloud.bin")
        assert cloud.headers["content-encoding"] == "gzip"
        assert len(cloud.content) == 8 + 138_639 * 13  # httpx уже распаковал
        count, _ = struct.unpack_from("<II", cloud.content)
        assert count == 138_639

        script = client.get("/web/main.js")
        assert script.status_code == 200 and "javascript" in script.headers["content-type"]
        assert client.get("/web/vendor/three.module.js").status_code == 200
