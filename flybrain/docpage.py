"""Документы проекта в браузере (C13, D-039): `docs/*.md` → простая HTML-страница.

Окно «О проекте» ссылается на отчёт `docs/VALIDATION.md` и инструкцию `docs/USER_GUIDE.md`. Сервер локальный,
поэтому показываем их сам: Markdown превращаем в HTML библиотекой markdown-it-py, картинки и соседние документы
берутся по тем же относительным ссылкам, что и в файле.
"""

import html
import re
from pathlib import Path

from flybrain.paths import PROJECT_ROOT

DOCS_DIR = PROJECT_ROOT / "docs"
# Только файлы прямо в docs/ с простым именем: никаких «..» и подпапок.
DOC_NAME = re.compile(r"^[A-Za-z0-9_]+\.md$")
# Короткие итоги проверки для окна «О проекте»; пишет `flybrain validate` (модуль report), путь — от docs/.
VALIDATION_SUMMARY = "validation/summary.json"

PAGE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — FlyBrain</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; background: #fcfcfb; color: #1b1b1a; font: 15px/1.55 "Segoe UI", system-ui, sans-serif; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 24px 20px 48px; }}
  h1 {{ font-size: 26px; line-height: 1.25; }}
  h2 {{ margin-top: 32px; padding-top: 12px; border-top: 1px solid #e1e0d9; font-size: 20px; }}
  h3 {{ font-size: 16px; }}
  a {{ color: #1f5fb0; }}
  code {{ padding: 1px 4px; border-radius: 4px; background: #efeee9; font: 13px Consolas, monospace; }}
  pre {{ overflow-x: auto; padding: 10px 12px; border-radius: 6px; background: #efeee9; }}
  pre code {{ padding: 0; background: none; }}
  .table {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; margin: 12px 0; font-size: 14px; }}
  th, td {{ padding: 5px 9px; border: 1px solid #d6d5cc; vertical-align: top; text-align: left; }}
  th {{ background: #f1f0ea; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #e1e0d9; }}
  blockquote {{ margin: 12px 0; padding: 2px 14px; border-left: 3px solid #c3c2b7; color: #52514e; }}
  .back {{ font-size: 13px; color: #52514e; }}
</style>
</head>
<body><main>
<p class="back"><a href="/">← к мозгу</a> · файл <code>docs/{name}</code></p>
{body}
</main></body>
</html>
"""


def doc_path(name: str, docs_dir: Path = DOCS_DIR) -> Path | None:
    """Путь к документу или None, если такого нет или имя не подходит."""
    if not DOC_NAME.match(name):
        return None
    path = docs_dir / name
    return path if path.is_file() else None


def render_doc(path: Path) -> str:
    from markdown_it import MarkdownIt

    text = path.read_text(encoding="utf-8")
    # HTML внутри Markdown не пропускаем: документ показывается как текст, а не как код страницы.
    renderer = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])
    body = renderer.render(text)
    body = body.replace("<table>", '<div class="table"><table>').replace("</table>", "</table></div>")
    heading = next((line[2:].strip() for line in text.splitlines() if line.startswith("# ")), path.stem)
    return PAGE.format(title=html.escape(heading), name=html.escape(path.name), body=body)
