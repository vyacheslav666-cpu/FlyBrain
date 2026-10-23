"""`flybrain doctor`: самопроверка машины. Печатает результаты по-русски."""

import locale
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from flybrain.paths import PROJECT_ROOT

OK = "OK"
WARN = "ВНИМАНИЕ"
FAIL = "ОШИБКА"

EXPECTED_PYTHON = (3, 12)
MAX_PATH_LEN = 100
MIN_FREE_GB = 8  # данные и результаты M1 (docs/SETUP.md)
COMFY_FREE_GB = 15  # с запасом под новые прогоны и отчёты
EXPECTED_GPU = "RTX 5070"

_COLORS = {OK: "32", WARN: "33", FAIL: "31"}


@dataclass
class Check:
    status: str
    name: str
    detail: str


def check_python() -> Check:
    version = platform.python_version()
    if sys.version_info[:2] == EXPECTED_PYTHON:
        return Check(OK, "Python", f"{version} (нужен {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]})")
    return Check(
        FAIL,
        "Python",
        f"{version}, а проекту нужен {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}. "
        "Запускайте через `uv run flybrain doctor`.",
    )


def check_project_path() -> Check:
    path = str(PROJECT_ROOT)
    problems = []
    if not path.isascii():
        problems.append("в пути есть не латинские буквы")
    if len(path) >= MAX_PATH_LEN:
        problems.append(f"путь длиннее {MAX_PATH_LEN - 1} символов ({len(path)})")
    if problems:
        return Check(FAIL, "Путь проекта", f"{path}: {'; '.join(problems)} (D-001)")
    return Check(OK, "Путь проекта", f"{path} — только латиница, {len(path)} символов")


def check_gpu() -> Check:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - любая ошибка импорта нам важна
        return Check(FAIL, "Видеокарта", f"не удалось загрузить PyTorch: {exc}")
    if not torch.cuda.is_available():
        return Check(
            WARN,
            "Видеокарта",
            f"PyTorch {torch.__version__} не видит видеокарту NVIDIA; "
            "движок мозга сможет считать только на процессоре, намного медленнее",
        )
    props = torch.cuda.get_device_properties(0)
    gigabytes = round(props.total_memory / 1024**3)
    try:
        # Реальный расчёт: убеждаемся, что для этой карты в PyTorch есть готовый код.
        ones = torch.ones(256, 256, device="cuda")
        probe = ones @ ones
        torch.cuda.synchronize()
        if float(probe[0, 0]) != 256.0:
            return Check(FAIL, "Видеокарта", f"{props.name}: пробный расчёт дал неверный ответ")
    except Exception as exc:  # noqa: BLE001
        return Check(FAIL, "Видеокарта", f"{props.name}: пробный расчёт не удался: {exc}")
    detail = f"{props.name}, {gigabytes} ГБ (PyTorch {torch.__version__}, пробный расчёт прошёл)"
    if EXPECTED_GPU not in props.name:
        return Check(WARN, "Видеокарта", f"{detail}; ожидалась NVIDIA GeForce RTX 5070 Laptop GPU, 8 ГБ")
    if gigabytes < 8:
        return Check(WARN, "Видеокарта", f"{detail}; памяти меньше ожидаемых 8 ГБ")
    return Check(OK, "Видеокарта", detail)


def _find_cpp_compiler() -> tuple[str, str] | None:
    """Возвращает (название, путь) найденного компилятора C++ или None."""
    program_files = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")
    if program_files:
        vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            result = subprocess.run(
                [
                    str(vswhere), "-all", "-products", "*", "-utf8",
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find", r"VC\Tools\MSVC\**\bin\Hostx64\x64\cl.exe",
                ],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if lines:
                # Версия компилятора записана в пути (…\MSVC\14.29.30133\…): берём самую свежую.
                newest = max(lines, key=lambda line: Path(line).parts[-5])
                return "Microsoft Visual C++", newest
    for name in ("cl", "g++"):
        found = shutil.which(name)
        if found:
            return name, found
    return None


def check_cpp_compiler() -> Check:
    found = _find_cpp_compiler()
    if found is None:
        return Check(
            FAIL,
            "Компилятор C++",
            "не найден. Он нужен Brian2 для эталонных прогонов. Установите "
            "«Visual Studio Build Tools» с набором «Разработка классических приложений на C++».",
        )
    name, path = found
    return Check(OK, "Компилятор C++", f"{name}: {path}")


def check_disk() -> Check:
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / 1024**3
    detail = f"свободно {free_gb:.0f} ГБ на диске проекта"
    if free_gb < MIN_FREE_GB:
        return Check(FAIL, "Место на диске", f"{detail}; нужно не меньше {MIN_FREE_GB} ГБ")
    if free_gb < COMFY_FREE_GB:
        return Check(WARN, "Место на диске", f"{detail}; с запасом лучше {COMFY_FREE_GB} ГБ")
    return Check(OK, "Место на диске", f"{detail} (нужно около {MIN_FREE_GB} ГБ)")


def check_utf8() -> Check:
    system_encoding = locale.getpreferredencoding(False)
    sample = "Вячеслав: мозг мухи"
    try:
        with tempfile.TemporaryDirectory() as folder:
            probe = Path(folder) / "проба.txt"
            probe.write_text(sample, encoding="utf-8")
            if probe.read_text(encoding="utf-8") != sample:
                return Check(FAIL, "Кодировка", "кириллица в UTF-8-файле прочиталась неверно")
    except OSError as exc:
        return Check(FAIL, "Кодировка", f"не удалось записать файл с кириллицей в имени: {exc}")
    mode = "включён" if sys.flags.utf8_mode else "выключен (его включает run.bat)"
    return Check(
        OK,
        "Кодировка",
        f"файлы читаем и пишем в UTF-8 явно; системная кодировка {system_encoding}; режим UTF-8 {mode}",
    )


CHECKS = (
    check_python,
    check_project_path,
    check_gpu,
    check_cpp_compiler,
    check_disk,
    check_utf8,
)


def _paint(text: str, status: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{_COLORS[status]}m{text}\033[0m"


def run_doctor() -> int:
    if sys.platform == "win32":
        os.system("")  # включает цвета в консоли Windows
    print("FlyBrain: проверка машины\n")
    results = []
    for check in CHECKS:
        try:
            result = check()
        except Exception as exc:  # noqa: BLE001 - одна сломанная проверка не должна скрыть остальные
            result = Check(FAIL, check.__name__, f"проверка сама упала: {exc}")
        results.append(result)
        tag = f"[{result.status}]"
        print(f"{_paint(tag, result.status)}{' ' * (10 - len(tag))} {result.name}: {result.detail}")

    failed = sum(r.status == FAIL for r in results)
    warned = sum(r.status == WARN for r in results)
    print()
    if failed:
        print(_paint(f"Не готово: ошибок {failed}, предупреждений {warned}. Исправьте красные пункты.", FAIL))
        return 1
    if warned:
        print(_paint(f"Работать можно, но есть предупреждения: {warned}.", WARN))
        return 0
    print(_paint("Всё в порядке: машина готова.", OK))
    return 0
