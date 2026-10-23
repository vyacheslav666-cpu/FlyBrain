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
    reference = subparsers.add_parser(
        "reference", help="эталонные прогоны оригинального кода Shiu на Brian2 в data/reference/ (около 25 минут)"
    )
    reference.add_argument("--jobs", type=int, default=6, help="сколько повторов считать одновременно (по умолчанию 6)")
    reference.add_argument("--trials", type=int, help="повторов на условие (по умолчанию 30 — из config/reference.yaml)")
    reference.add_argument("--only", help="только эти условия, через запятую (например sugar_100hz)")
    reference.add_argument("--out", help="папка для результатов (по умолчанию data/reference)")
    sim = subparsers.add_parser("sim", help="прогнать модель на движке PyTorch и показать самые активные нейроны")
    sim.add_argument("--stim", required=True,
                     help="группы-стимулы через запятую: sugar, bitter, water, ir94e, jon_ce, jon_f, jon_dm, "
                          "antenna, shadow_left, shadow_right")
    sim.add_argument("--hz", type=float, default=100.0, help="частота стимула, Гц (по умолчанию 100)")
    sim.add_argument("--trials", type=int, default=30, help="повторов одновременно (по умолчанию 30)")
    sim.add_argument("--duration-ms", type=float, default=1000.0, help="длительность повтора, мс (по умолчанию 1000)")
    sim.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                     help="где считать: auto — видеокарта, если есть (по умолчанию)")
    sim.add_argument("--seed", type=int, help="зерно случайных чисел (для точного повторения)")
    sim.add_argument("--top", type=int, default=20, help="сколько самых активных нейронов показать (по умолчанию 20)")
    validate = subparsers.add_parser(
        "validate", help="проверить движок против эталона Brian2 и выводов статьи, написать docs/VALIDATION.md"
    )
    validate.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                          help="где считать движок: auto — видеокарта, если есть (по умолчанию)")
    validate.add_argument("--rerun", action="store_true",
                          help="пересчитать движок заново, даже если готовые результаты подходят")
    psy = subparsers.add_parser(
        "psy", help="эксперимент P1 «психоделик»: доза против отклика мозга, таблицы и картинки в docs/psy/")
    psy.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                     help="где считать движок: auto — видеокарта, если есть (по умолчанию)")
    psy.add_argument("--rerun", action="store_true", help="пересчитать всё заново, даже если готовые расчёты подходят")
    psy.add_argument("--only", help="только эти части, через запятую: runaway, main, thresholds, spont, broad")
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
    if args.command == "reference":
        from pathlib import Path

        from flybrain.reference import REFERENCE_DIR, run_reference

        only = [key.strip() for key in args.only.split(",") if key.strip()] if args.only else None
        return run_reference(trials=args.trials, jobs=args.jobs, only=only,
                             out_dir=Path(args.out).resolve() if args.out else REFERENCE_DIR)
    if args.command == "sim":
        from flybrain.sim import run_sim

        stimuli = [key.strip() for key in args.stim.split(",") if key.strip()]
        return run_sim(stimuli, hz=args.hz, trials=args.trials, duration_ms=args.duration_ms,
                       device=args.device, seed=args.seed, top=args.top)
    if args.command == "validate":
        from flybrain.validate import run_validate

        return run_validate(device=args.device, rerun=args.rerun)
    if args.command == "psy":
        from flybrain.psy import run_psy

        parts = [part.strip() for part in args.only.split(",") if part.strip()] if args.only else None
        return run_psy(device=args.device, rerun=args.rerun, parts=parts)
    if args.command == "serve":
        from flybrain.server import run_serve

        return run_serve(port=args.port, open_browser=not args.no_browser)
    parser.print_help()
    return 0
