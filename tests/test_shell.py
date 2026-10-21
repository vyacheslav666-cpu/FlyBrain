"""Тесты силуэта мозга и страницы (коммит C12): сетка силуэта считается на Node, ссылки страницы на элементы."""

import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from flybrain import processed
from flybrain.paths import PROCESSED_DIR, PROJECT_ROOT
from flybrain.server import create_app

CHECK = PROJECT_ROOT / "tests" / "js" / "shell_check.mjs"
needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="нет Node.js: силуэт проверяется им")
needs_data = pytest.mark.skipif(
    not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists(), reason="нет data/processed — запустите build-data"
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["node", str(CHECK), *args], capture_output=True, text=True, encoding="utf-8", timeout=120)


@needs_node
def test_shell_of_hollow_ball_is_one_closed_solid_ball():
    # Полая «кора» + одинокие точки + нейрон без координат: пустота заливается, одиночки не дают кусков.
    result = _run()
    assert result.returncode == 0, result.stdout + result.stderr


@needs_node
@needs_data
def test_shell_of_real_cloud_covers_the_points(tmp_path):
    with TestClient(create_app()) as client:
        cloud = tmp_path / "cloud.bin"
        cloud.write_bytes(client.get("/api/cloud.bin").content)
    result = _run("--cloud", str(cloud))
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_element_the_scripts_look_up_exists_on_the_page():
    # Опечатка в идентификаторе ломает страницу только в браузере; ловим её здесь.
    page = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'\bid="([^"]+)"', page))
    missing = []
    for script in sorted((PROJECT_ROOT / "web").glob("*.js")):
        text = script.read_text(encoding="utf-8")
        for name in re.findall(r'getElementById\("([^"]+)"\)|\$\("([^"]+)"\)', text):
            name = name[0] or name[1]
            if name not in ids:
                missing.append(f"{script.name}: {name}")
    assert not missing, f"в index.html нет элементов: {missing}"


@needs_node
def test_found_neurons_fit_between_panels():
    # После поиска камера отходит так, чтобы найденные нейроны не оказались под боковыми панелями (D-040).
    check = PROJECT_ROOT / "tests" / "js" / "framing_check.mjs"
    result = subprocess.run(["node", str(check)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
