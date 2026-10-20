"""Остановка устаревшей копии FlyBrain на нашем порту (D-028). Чужие программы не трогаем."""

import json
import os
import re
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass

# Так выглядит наш сервер в списке процессов: «...\flybrain.exe serve» или «python -m flybrain serve».
_OUR_SERVER = re.compile(r'(?<![\w.-])flybrain(?:\.exe)?"?\s+serve(?=[\s"]|$)', re.IGNORECASE)
_STOP_WAIT_SECONDS = 10


@dataclass(frozen=True)
class Listener:
    pid: int
    command_line: str


def looks_like_flybrain_server(command_line: str | None) -> bool:
    return bool(command_line) and _OUR_SERVER.search(command_line) is not None


def find_listeners(port: int) -> list[Listener]:
    """Кто слушает порт на этой машине. Вызываем PowerShell: системные утилиты пишут по-русски, их разбирать хуже."""
    script = (
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
        f"$found = @(Get-NetTCPConnection -LocalPort {int(port)} -State Listen -ErrorAction SilentlyContinue | "
        "Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { "
        "[pscustomobject]@{ pid = $_; cmd = (Get-CimInstance Win32_Process -Filter \"ProcessId=$_\").CommandLine } }); "
        "ConvertTo-Json -InputObject $found -Compress"
    )
    done = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, timeout=30, check=True,
    )
    rows = json.loads(done.stdout.decode("utf-8").strip() or "[]")
    return [Listener(int(row["pid"]), row.get("cmd") or "") for row in rows]


@dataclass(frozen=True)
class StopResult:
    ok: bool
    message: str


def stop_old_server(
    port: int,
    port_is_busy: Callable[[], bool],
    finder: Callable[[int], list[Listener]] = find_listeners,
    killer: Callable[[int], None] = lambda pid: os.kill(pid, signal.SIGTERM),  # на Windows это немедленное завершение
) -> StopResult:
    """Останавливает процесс, слушающий порт, но только если все слушающие процессы — наши `flybrain serve`."""
    try:
        listeners = finder(port)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        return StopResult(False, f"не удалось узнать, какая программа занимает порт ({error})")
    if not listeners:
        return StopResult(False, "не нашёл программу, которая занимает порт")
    foreign = [item for item in listeners if not looks_like_flybrain_server(item.command_line)]
    if foreign:
        names = ", ".join(f"{item.pid}: {item.command_line or 'неизвестно'}" for item in foreign)
        return StopResult(False, f"порт занят программой, которая не похожа на FlyBrain ({names}); её я не трогаю")

    for item in listeners:
        try:
            killer(item.pid)
        except OSError as error:
            return StopResult(False, f"процесс {item.pid} не завершился ({error})")
    deadline = time.monotonic() + _STOP_WAIT_SECONDS
    while time.monotonic() < deadline:
        if not port_is_busy():
            return StopResult(True, "старая копия остановлена")
        time.sleep(0.1)
    return StopResult(False, f"порт не освободился за {_STOP_WAIT_SECONDS} с")
