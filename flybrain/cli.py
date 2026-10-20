"""Консольная команда `flybrain` и её подкоманды."""

import argparse
import sys

from flybrain import __version__


def _use_utf8_output() -> None:
    # Системная кодировка здесь cp1251; вывод в файл или по трубе всегда делаем в UTF-8 (D-021).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_output()
    parser = argparse.ArgumentParser(
        prog="flybrain",
        description="FlyBrain: симуляция мозга дрозофилы на коннектоме FlyWire 783.",
    )
    parser.add_argument("--version", action="version", version=f"flybrain {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="команда")
    subparsers.add_parser("doctor", help="проверить, что машина готова к работе")
    subparsers.add_parser("fetch-data", help="скачать данные в data/raw/ и сверить контрольные суммы")
    subparsers.add_parser(
        "build-data", help="собрать таблицу нейронов, матрицу связей, группы и картинку в data/processed/"
    )
    serve = subparsers.add_parser("serve", help="запустить сервер и открыть 3D-облако нейронов в браузере")
    serve.add_argument("--port", type=int, default=8765, help="порт сервера (по умолчанию 8765)")
    serve.add_argument("--no-browser", action="store_true", help="не открывать браузер автоматически")

    args = parser.parse_args(argv)
    if args.command == "doctor":
        from flybrain.doctor import run_doctor

        return run_doctor()
    if args.command == "fetch-data":
        from flybrain.fetch import run_fetch_data

        return run_fetch_data()
    if args.command == "build-data":
        from flybrain.build import run_build_data

        return run_build_data()
    if args.command == "serve":
        from flybrain.server import run_serve

        return run_serve(port=args.port, open_browser=not args.no_browser)
    parser.print_help()
    return 0
