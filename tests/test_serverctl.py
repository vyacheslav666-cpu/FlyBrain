"""Тесты замены устаревшего сервера (D-028): версия кода и осторожная остановка процесса на порту."""

import socket
import subprocess
import sys
import time

import pytest
from fastapi.testclient import TestClient

from flybrain import server
from flybrain.serverctl import Listener, find_listeners, looks_like_flybrain_server, stop_old_server
from flybrain.version import code_version

OURS = r'"C:\Users\x\AppData\python.exe"  "C:\Projects\FlyBrain\.venv\Scripts\flybrain.exe" serve'


# ---------- версия кода ----------

def _tree(root):
    for name, text in {"flybrain/a.py": "x = 1\n", "web/main.js": "//\n", "config/n.yaml": "a: 1\n"}.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


def test_version_is_stable_and_changes_with_code(tmp_path):
    _tree(tmp_path)
    first = code_version(tmp_path)
    assert first == code_version(tmp_path) and len(first) == 12
    (tmp_path / "flybrain" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert code_version(tmp_path) != first


def test_version_ignores_line_endings_vendor_and_caches(tmp_path):
    _tree(tmp_path)
    first = code_version(tmp_path)
    (tmp_path / "flybrain" / "a.py").write_bytes(b"x = 1\r\n")  # git поменял переводы строк
    (tmp_path / "web" / "vendor").mkdir()
    (tmp_path / "web" / "vendor" / "three.js").write_text("чужой код", encoding="utf-8")
    (tmp_path / "flybrain" / "__pycache__").mkdir()
    (tmp_path / "flybrain" / "__pycache__" / "a.py").write_text("мусор", encoding="utf-8")
    assert code_version(tmp_path) == first


def test_version_notices_new_config_and_new_web_file(tmp_path):
    _tree(tmp_path)
    first = code_version(tmp_path)
    (tmp_path / "web" / "extra.js").write_text("//", encoding="utf-8")
    assert code_version(tmp_path) != first


def test_ping_reports_version_of_this_code():
    from flybrain.paths import PROCESSED_DIR

    if not (PROCESSED_DIR / "build_info.json").exists():
        pytest.skip("нет data/processed — запустите build-data")
    with TestClient(server.create_app()) as client:
        assert client.get("/api/ping").json() == {"app": "flybrain", "version": server.CODE_VERSION}


# ---------- опознание нашего сервера ----------

@pytest.mark.parametrize("command", [OURS, OURS + " --port 8766", "python -m flybrain serve", "FLYBRAIN.EXE serve"])
def test_recognises_our_server(command):
    assert looks_like_flybrain_server(command)


@pytest.mark.parametrize("command", [
    None, "", "python.exe -m http.server 8765", "nginx.exe",
    r"C:\Projects\FlyBrain\.venv\Scripts\flybrain.exe doctor", "notepad.exe flybrain serve.txt",
    "other_flybrain.exe serve",
])
def test_does_not_recognise_foreign_processes(command):
    assert not looks_like_flybrain_server(command)


# ---------- решение, кого останавливать (без настоящих процессов) ----------

def _stop(listeners, busy=lambda: False):
    killed = []
    result = stop_old_server(1, port_is_busy=busy, finder=lambda port: listeners, killer=killed.append)
    return result, killed


def test_stops_only_our_server():
    result, killed = _stop([Listener(10, OURS)])
    assert result.ok and killed == [10]


def test_refuses_foreign_process_and_kills_nothing():
    result, killed = _stop([Listener(11, "python.exe -m http.server 8765")])
    assert not result.ok and killed == [] and "не трогаю" in result.message


def test_refuses_when_any_listener_is_foreign():
    result, killed = _stop([Listener(10, OURS), Listener(11, "nginx.exe")])
    assert not result.ok and killed == []


def test_refuses_when_nobody_found():
    result, killed = _stop([])
    assert not result.ok and killed == []


def test_reports_when_port_stays_busy(monkeypatch):
    monkeypatch.setattr("flybrain.serverctl._STOP_WAIT_SECONDS", 0.3)
    result, killed = _stop([Listener(10, OURS)], busy=lambda: True)
    assert killed == [10] and not result.ok and "не освободился" in result.message


# ---------- настоящие процессы ----------

LISTENER = ("import socket, time; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
            "print(s.getsockname()[1], flush=True); time.sleep(40)")


def _start_listener(*extra):
    process = subprocess.Popen([sys.executable, "-c", LISTENER, *extra], stdout=subprocess.PIPE, text=True)
    return process, int(process.stdout.readline())


def _busy(port):
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _cleanup(process, port):
    subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
    for _ in range(50):
        if not _busy(port):
            break
        time.sleep(0.1)


@pytest.mark.skipif(sys.platform != "win32", reason="остановка сервера рассчитана на Windows (D-003)")
def test_real_process_looking_like_ours_is_stopped():
    process, port = _start_listener("flybrain", "serve")
    try:
        assert any("flybrain" in item.command_line for item in find_listeners(port))
        result = stop_old_server(port, port_is_busy=lambda: _busy(port))
        assert result.ok, result.message
        assert not _busy(port)
    finally:
        _cleanup(process, port)


@pytest.mark.skipif(sys.platform != "win32", reason="остановка сервера рассчитана на Windows (D-003)")
def test_real_foreign_process_is_left_alone():
    process, port = _start_listener()
    try:
        result = stop_old_server(port, port_is_busy=lambda: _busy(port))
        assert not result.ok and _busy(port)
    finally:
        _cleanup(process, port)
