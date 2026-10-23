"""Версия кода для `/api/ping` (D-028): по ней `flybrain serve` отличает свежий сервер от устаревшего."""

import hashlib
from pathlib import Path

from flybrain.paths import PROJECT_ROOT

# Что сервер читает при запуске или отдаёт странице. Сторонние файлы three.js (web/vendor) не берём: они неизменны.
_WATCHED = (
    ("flybrain", "**/*.py"),
    ("web", "*.html"),
    ("web", "*.js"),
    ("web", "*.css"),
    ("config", "*.yaml"),
    ("data/processed", "build_info.json"),  # после нового `build-data` сервер тоже устарел
)


def code_version(root: Path = PROJECT_ROOT) -> str:
    """Короткая контрольная сумма кода, настроек и справки о подготовке данных.

    Переводы строк приводятся к одному виду: смена `LF`/`CRLF` при работе git версию не меняет.
    """
    files = sorted({path for folder, pattern in _WATCHED for path in (root / folder).glob(pattern)
                    if path.is_file() and "__pycache__" not in path.parts})
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return digest.hexdigest()[:12]
