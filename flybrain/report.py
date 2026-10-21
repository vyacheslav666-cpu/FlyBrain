"""Отчёт `docs/VALIDATION.md` и картинки `docs/validation/*.png` по итогам `flybrain validate` (C6, D-031).

Текст пишется простыми словами: что ждали по статье, по какому правилу решали, сырые числа эталона
и движка, отметка и что она значит. Файл каждый раз собирается заново — вручную его не правят.
"""

import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from flybrain.paths import PROJECT_ROOT
from flybrain.reference import ACTIVE_HZ
from flybrain.validate import BOUNDARY, BOUNDARY_SE, CORR_MIN, FAIL, INFO, ITEMS, KEY_ABS_HZ, KEY_REL, MARKS, \
    MARK_WORDS, OVERLAP_MIN, PAPER_B2, PASS, Agreement, Results, Verdict, number

REPORT_FILE = PROJECT_ROOT / "docs" / "VALIDATION.md"
FIGURES_DIR = PROJECT_ROOT / "docs" / "validation"
FIGURE_RATES = "a_rates.png"
FIGURE_B2 = "b2_counts.png"
FIGURE_KEYS = "b_key_neurons.png"

# Цвета — палитра по умолчанию навыка dataviz, светлая тема; три цвета проверены скриптом validate_palette.js.
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR_REFERENCE, COLOR_ENGINE, COLOR_PAPER = "#2a78d6", "#eb6834", "#1baf7a"

# Что ждём по статье, по какому правилу решаем, и пояснения — для каждого пункта (правила D-031).
TEXTS = {
    "Б1": {
        "expected": "MN9 включается с обеих сторон, на противоположной стороне сильнее. Сахарные нейроны в списке "
                    "статьи — на левой стороне мухи (D-011), значит, противоположная — правая.",
        "rule": "сахар 100 Гц: оба MN9 активны (≥ 1 Гц), и правый сильнее левого.",
    },
    "Б2": {
        "expected": "чем чаще стреляют сахарные нейроны, тем больше нейронов мозга откликается: от десятков до "
                    f"сотен (в статье, на старой версии 630: {PAPER_B2[10]} при 10 Гц и {PAPER_B2[200]} при 200 Гц).",
        "rule": "«откликнулся» — частота больше 0 Гц, как определено в самой статье; стимулируемые нейроны не "
                "считаем. При 10 Гц — от 10 до 99, при 200 Гц — от 100 до 999, и число растёт с каждой частотой "
                "(10 → 50 → 100 → 200 Гц).",
        "notes": ["**[ЧЕСТНОСТЬ]** Для Б2 «откликнулся» — это «больше 0 Гц», а не «≥ 1 Гц», как в пункте А. Так "
                  "в статье: «Activated neurons are defined as neurons that have greater than 0 Hz firing» "
                  "(Shiu et al., препринт bioRxiv 10.1101/2023.05.02.539144, раздел Results). Б2 сравнивает с "
                  "числами статьи, поэтому и считаем по её правилу. Исправлено после того, как эталон сверили "
                  "с текстом статьи (D-031); пороги А не менялись. Счёт по правилу ≥ 1 Гц тоже показан выше.",
                  "Входят ли сами стимулируемые нейроны в 45 и 455 у статьи, из текста не ясно; поэтому выше "
                  "оба числа — без них и с ними."],
    },
    "Б3": {
        "expected": "сильная горечь подавляет ответ MN9 на сахар.",
        "rule": "MN9 из статьи отвечает на сахар 100 Гц (≥ 1 Гц), а при «сахар 100 + горечь 200 Гц» — не больше "
                "половины этого ответа.",
    },
    "Б4": {
        "expected": "вкусовые нейроны воды включают MN9.",
        "rule": "вода 100 Гц: хотя бы один MN9 активен (≥ 1 Гц). Частота 100 Гц записана в плане C4 заранее; "
                "под результат её не меняем (D-018).",
        "notes": ["Главный вывод статьи про воду другой: вода 40 Гц и сахар 40 Гц по отдельности MN9 заметно не "
                  "включают, а вместе включают. Его проверяет пункт Д1 ниже."],
    },
    "Б5": {
        "expected": "на включение нейронов Джонстонова органа (усик) отвечают нейроны груминга aDN1, aDN2 и aBN1.",
        "rule": "все JON 100 Гц: aDN1, aDN2 и aBN1 с номерами из статьи активны (≥ 1 Гц).",
    },
    "Б6": {
        "expected": "JO-CE сильно включает aBN1, а JO-F — нет (в плане: «почти нет»).",
        "rule": "aBN1 из статьи активен от JO-CE 100 Гц (≥ 1 Гц) и не активен от JO-F 100 Гц (< 1 Гц). Порог тот "
                "же, что у «активного нейрона» в D-018.",
        "notes": ["Порог 1 Гц — это в среднем один спайк за секунду. Если от JO-F среднее легло прямо на порог, "
                  "отметка решается случайностью, и правило «на границе» честно показывает это жёлтым. Сами "
                  "числа — выше: по ним видно, во сколько раз JO-F слабее JO-CE."],
    },
    "Б7": {
        "expected": "контроль: если связи перемешать случайно, MN9 на сахар не отвечает.",
        "rule": "сахар 100 Гц на перемешанных связях: оба MN9 не активны (< 1 Гц).",
        "notes": ["**[ЧЕСТНОСТЬ]** Способ перемешивания — наш (у Shiu он не записан): получатели всех связей "
                  "переставлены случайно, у каждого нейрона сохранены число входов, число выходов и веса выходов "
                  "(D-029). Движок считает на том же перемешанном файле, что и эталон."],
    },
    "В": {
        "expected": "в статье этот опыт не проверялся. По коннектому зрительные нейроны LPLC2 и LC4 дают "
                    "гигантскому волокну (GF, нейрон побега) больше всего входов (D-013).",
        "rule": "порога нет: записываем, что произошло.",
        "notes": ["**[ЧЕСТНОСТЬ]** Это не настоящая тень, а прямое включение 162 зрительных нейронов левой "
                  "стороны; группу и «исполнителя» (GF, тип DNp01) выбрали мы по таблице аннотаций. У живой мухи "
                  "гигантское волокно получает много сигналов через электрические синапсы, а их в коннектоме нет."],
    },
    "Д1": {
        "expected": "вода 40 Гц и сахар 40 Гц по отдельности MN9 заметно не включают, а вместе — сильно включают "
                    "(рис. 4 статьи).",
        "rule": "MN9 из статьи при «сахар 40 + вода 40 Гц» активен (≥ 1 Гц) и как минимум вдвое сильнее, чем от "
                "каждого стимула по отдельности. «По отдельности» устроено как в сетке опытов Shiu: у второй "
                "группы стимулов частота 0 Гц.",
        "notes": ["**[ЧЕСТНОСТЬ] Пункт добавлен после того, как увидели эталон** (вода 100 Гц не включает MN9 "
                  "уже в оригинальном Brian2). В зафиксированных заранее пунктах D-018 его не было. Правило и "
                  "множитель «вдвое» (наш выбор для слов статьи «заметно» и «сильно») записаны до эталонного "
                  "прогона этих трёх условий (D-031)."],
    },
}

# Пункты на картинке ключевых нейронов (подписи столбиков — ключи Verdict.values) и короткие заголовки панелей.
FIGURE_PANELS = {
    "Б1": "Б1 · сахар 100 Гц: MN9",
    "Б3": "Б3 · MN9 (правый): сахар 100 без горечи и с ней",
    "Б4": "Б4 · вода 100 Гц: MN9",
    "Б5": "Б5 · все JON 100 Гц: груминг",
    "Б6": "Б6 · aBN1 (левый): JO-CE и JO-F",
    "Б7": "Б7 · сахар 100 Гц, связи перемешаны: MN9",
    "В": "В · тень слева 100 Гц: GF",
    "Д1": "Д1 · MN9 (правый): сахар 40, вода 40, вместе",
}
# Условия, добавленные после того, как увидели эталон (пункт Д1): в таблице А они помечены.
ADDED_CONDITIONS = next(item.conditions for item in ITEMS if item.key == "Д1")


def _mark(state: str) -> str:
    return f"{MARKS[state]} {MARK_WORDS[state]}"


def _corr(agreement: Agreement) -> str:
    return "нельзя посчитать" if math.isnan(agreement.corr) else number(agreement.corr, 3)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _same_within_noise(reference: Verdict, engine: Verdict) -> bool | None:
    """Отличаются ли числа эталона и движка в проверках «на границе» меньше чем на две погрешности.

    Проверки двух вердиктов идут в одном порядке (их строит одна и та же функция). None — сравнивать нечего.
    """
    pairs = [(r, e) for r, e in zip(reference.tests, engine.tests)
             if BOUNDARY in (r.state, e.state) and r.state != e.state]
    if not pairs:
        return None
    return all(abs(r.margin - e.margin) < BOUNDARY_SE * math.hypot(r.se, e.se) for r, e in pairs)


def _interpretation(reference: Verdict, engine: Verdict) -> str:
    r, e = reference.state, engine.state
    if r == INFO:
        return ""
    if r == e == PASS:
        return "Вывод статьи повторяется и в оригинальном коде Shiu (Brian2) на версии 783, и в нашем движке."
    if r == e == FAIL:
        return ("Не выполняется уже в оригинальном коде Shiu (Brian2) на данных версии 783, и движок даёт то же "
                "самое. Значит, дело не в нашем движке, а в самой модели на этих данных и условиях опыта. "
                "Модель не подгоняем (D-005, D-018); это результат.")
    if r == PASS and e == FAIL:
        return ("Эталон выполняет, а движок — нет. Это расхождение движка с оригиналом; смотрите пункт А для "
                "этих условий.")
    if r == FAIL and e == PASS:
        return ("Движок выполняет, а эталон — нет. Это расхождение движка с оригиналом, засчитывать вывод "
                "статьи нельзя.")
    parts = []
    for name, verdict in (("эталоне", reference), ("движке", engine)):
        boundary = [test.text for test in verdict.tests if test.state == BOUNDARY]
        if boundary:
            parts.append(f"в {name} — «{'», «'.join(boundary)}»")
    text = ("На границе: " + "; ".join(parts) + ". Здесь число отстоит от порога меньше чем на две погрешности, "
            "и по 30 повторам нельзя уверенно сказать, по какую сторону порога оно лежит. Жёлтая отметка не "
            "считается выполненной.") if parts else ""
    if FAIL in (r, e):
        failed = "эталоне" if r == FAIL else "движке"
        text += f" В {failed} пункт не выполнен."
    same = _same_within_noise(reference, engine)
    if same is True:
        text += (" Отметки у эталона и движка разные, но сами числа отличаются меньше чем на две погрешности, то "
                 "есть в пределах случайности: расхождения движка с эталоном это не показывает (см. и пункт А).")
    elif same is False:
        text += (" Числа эталона и движка здесь отличаются больше чем на две погрешности — это расхождение "
                 "движка, его нужно разобрать.")
    return text.strip()


def _tests_lines(verdict: Verdict) -> list[str]:
    return [f"  - {MARKS[test.state]} {test.text}" for test in verdict.tests]


def _summary_lines(results: Results) -> list[str]:
    lines = []
    agreements = results.agreements
    good = [c for c in results.conditions if agreements[c.key].ok]
    lines.append(f"- **Движок против эталона (А):** все три порога выполнены в {len(good)} условиях из "
                 f"{len(results.conditions)}.")
    for condition in results.conditions:
        agreement = agreements[condition.key]
        if agreement.ok:
            continue
        failed = []
        if not agreement.corr_ok:
            failed.append(f"корреляция {_corr(agreement)}")
        if not agreement.overlap_ok:
            failed.append(f"совпадение {number(100 * agreement.overlap, 0)} %")
        if not agreement.keys_ok:
            failed.append("ключевой нейрон вне допуска")
        noise = results.noise[condition.key]
        lines.append(f"  - не прошло «{condition.name_ru}»: {', '.join(failed)}; у движка против самого себя "
                     f"с другим зерном — корреляция {_corr(noise)}, совпадение {number(100 * noise.overlap, 0)} % "
                     f"(активных нейронов здесь всего {agreement.n_union}).")
    groups = {}
    for item in ITEMS:
        reference, engine = results.verdicts[item.key]
        groups.setdefault((reference.state, engine.state), []).append(item.key)
    phrases = {  # (одно, несколько)
        (PASS, PASS): ("повторился и в эталоне Brian2, и в движке", "повторились и в эталоне Brian2, и в движке"),
        (FAIL, FAIL): ("не повторился уже в оригинальном Brian2 на версии 783 — значит, дело не в движке",
                       "не повторились уже в оригинальном Brian2 на версии 783 — значит, дело не в движке"),
        (PASS, FAIL): ("эталон выполняет, движок — нет: расхождение движка", "эталон выполняет, движок — нет: "
                                                                          "расхождение движка"),
        (FAIL, PASS): ("движок выполняет, эталон — нет: расхождение движка", "движок выполняет, эталон — нет: "
                                                                          "расхождение движка"),
    }
    verdicts = results.verdicts
    for states, keys in groups.items():
        if states[0] == INFO:
            continue
        if states in phrases:
            lines.append(f"- **{', '.join(keys)}:** {phrases[states][len(keys) > 1]}.")
            continue
        for key in keys:  # на границе: сразу сырые числа, чтобы граница была видна без чтения раздела
            reference, engine = verdicts[key]
            lines.append(f"- **{key}:** эталон — {MARK_WORDS[states[0]]}, движок — {MARK_WORDS[states[1]]}. "
                         f"Эталон: {reference.numbers[0]}; движок: {engine.numbers[0]}.")
            same = _same_within_noise(reference, engine)
            if same is not None:
                lines[-1] += (" Числа эталона и движка отличаются в пределах случайности." if same else
                              " Числа эталона и движка отличаются больше чем на две погрешности.")
    lines.append(f"- **В (тень):** {_shadow_words(*verdicts['В'], short=True)}")
    return lines


def _a_table(results: Results) -> list[str]:
    lines = [
        "| Условие | А1 корреляция частот (≥ 0,95) | А2 совпадение активных (≥ 80 %) | А3 ключевые нейроны "
        "(±15 % или ±3 Гц) | Итог | Шумовой предел: корреляция / совпадение |",
        "|---|---|---|---|---|---|",
    ]
    for condition in results.conditions:
        agreement, noise = results.agreements[condition.key], results.noise[condition.key]
        worst = agreement.worst_key()
        if worst is None:
            keys = "все 10 молчат в обоих расчётах"
        else:
            keys = (f"ближе всех к допуску {worst.key.label}: {number(worst.base)} → {number(worst.test)} Гц "
                    f"(допуск ±{number(worst.allowed)})")
        name = condition.name_ru + (" *(доп.)*" if condition.key in ADDED_CONDITIONS else "")
        lines.append(
            f"| {name} | {MARKS[PASS if agreement.corr_ok else FAIL]} {_corr(agreement)} "
            f"| {MARKS[PASS if agreement.overlap_ok else FAIL]} {number(100 * agreement.overlap, 0)} % "
            f"({agreement.n_common} из {agreement.n_union}; эталон {agreement.n_base}, движок {agreement.n_test}) "
            f"| {MARKS[PASS if agreement.keys_ok else FAIL]} {keys} "
            f"| {_mark(PASS if agreement.ok else FAIL)} "
            f"| {_corr(noise)} / {number(100 * noise.overlap, 0)} % |")
    return lines


def _item_section(item, results: Results) -> list[str]:
    reference, engine = results.verdicts[item.key]
    texts = TEXTS[item.key]
    lines = [f"### {item.key}. {item.title}", "",
             f"**Ожидание по статье.** {_cap(texts['expected'])}", "",
             f"**Правило.** {_cap(texts['rule'])}", ""]
    for name, verdict in (("Эталон Brian2", reference), ("Движок", engine)):
        lines.append(f"- **{name}** — {_mark(verdict.state)}")
        lines += [f"  - {line}" for line in verdict.numbers]
        lines += _tests_lines(verdict)
    lines.append("")
    meaning = _interpretation(reference, engine)
    if meaning:
        lines += [f"**Что это значит.** {meaning}", ""]
    if item.key == "В":
        lines += [_shadow_words(reference, engine), ""]
    for note in texts.get("notes", []):
        lines += [note, ""]
    if item.key == "Б2":
        lines += [f"![Число откликнувшихся нейронов от частоты сахара](validation/{FIGURE_B2})", ""]
    return lines


def _shadow_words(reference: Verdict, engine: Verdict, short: bool = False) -> str:
    def words(verdict: Verdict) -> str:
        left, right = verdict.values["GF Л"][0], verdict.values["GF П"][0]
        on_left, on_right = left >= ACTIVE_HZ, right >= ACTIVE_HZ
        if on_left and on_right:
            stronger = "левое, на стороне стимула" if left > right else "правое, на другой стороне от стимула"
            return (f"включились оба гигантских волокна — левое ({number(left)} Гц) и правое "
                    f"({number(right)} Гц); сильнее {stronger}")
        if on_left or on_right:
            name, rate, other = ("левое", left, right) if on_left else ("правое", right, left)
            return f"включилось только {name} гигантское волокно ({number(rate)} Гц; другое — {number(other)} Гц)"
        return f"гигантское волокно не включилось ни с одной стороны ({number(left)} и {number(right)} Гц)"

    text = f"в эталоне Brian2 {words(reference)}; в движке {words(engine)}."
    if short:
        return _cap(text)
    return (f"**Что произошло.** {_cap(text)} Это отклик модели на прямое включение LPLC2 и LC4 одной стороны; "
            "насколько он похож на реакцию живой мухи, этот опыт не говорит.")


def _technical_lines(results: Results) -> list[str]:
    info = results.info
    reference = info["reference"]
    seconds = [run.seconds for run in results.engine.values() if not math.isnan(run.seconds)]
    lines = [
        f"- Повторов на условие: {info['n_trials']} по {number(info['duration_ms'] / 1000, 0)} с; и у эталона, и у "
        f"движка. Условий: {len(results.conditions)} (11 из плана C4 и 3 добавленных для Д1).",
        f"- Эталон: `model.py` Shiu без изменений (коммит 91bdd1e), Brian2 {reference.get('brian2')}, Cython "
        f"{reference.get('cython')}, способ расчёта {', '.join(reference.get('codegen', []))}; зерно каждого "
        f"повтора — `{reference.get('seed_rule')}`; данные записаны {reference.get('written_at')}.",
        f"- Движок: PyTorch {torch.__version__}, {info['device']}; код движка — версия `{info['engine_version']}` "
        "(контрольная сумма файлов `flybrain/engine/`); 32-битные числа (D-030); зерно — "
        "`zlib.crc32('engine-<main|repeat>:<условие>')`.",
        "- Шумовой предел: тот же движок, то же условие, другое зерно случайных чисел (`repeat`). Показывает, "
        "насколько расходятся два набора по 30 повторов у одной и той же модели. На отметки не влияет.",
        f"- Связи: исходные — SHA-256 `{info['connectivity_sha256']['original'][:16]}…` (`Connectivity_783.parquet`); "
        f"перемешанные — `{info['connectivity_sha256'].get('shuffled', '—')[:16]}…` (тот же файл, что у эталона).",
    ]
    if seconds:
        lines.append(f"- Один расчёт движка (30 повторов сразу): {number(min(seconds), 0)}–"
                     f"{number(max(seconds), 0)} с.")
    return lines


def render_markdown(results: Results, written_at: str) -> str:
    lines = [
        "# Проверка движка FlyBrain (главный критерий M1)",
        "",
        f"*Отчёт собран командой `uv run flybrain validate` {written_at}. Файл каждый раз пишется заново — "
        "вручную его не правят. Правила проверки — [D-031](DECISIONS.md), пороги А — [D-018](DECISIONS.md).*",
        "",
        "**Что сравниваем.**",
        "- **Эталон** — оригинальный код Shiu et al. (Brian2) на коннектоме FlyWire 783, по 30 повторов на "
        "условие (коммит C4).",
        "- **Движок** — наш расчёт той же модели на видеокарте (коммит C5), тоже по 30 повторов, со своими "
        "случайными числами. Поэтому спайк в спайк расчёты не совпадают и не должны; сравниваем средние.",
        "- **Статья** — Shiu et al., *Nature* 2024 (препринт bioRxiv 10.1101/2023.05.02.539144). Её числа "
        "посчитаны на старой версии коннектома 630, поэтому с ней сравниваем только качественно.",
        "- Модель не подгоняли: параметры — D-005, пороги — D-018.",
        "",
        "## Итог",
        "",
        "| Пункт | Что проверяем | Эталон Brian2 | Движок |",
        "|---|---|---|---|",
    ]
    good = sum(agreement.ok for agreement in results.agreements.values())
    total = len(results.agreements)
    lines.append(f"| А | движок совпадает с эталоном (все условия) | — | "
                 f"{_mark(PASS if good == total else FAIL)}: {good} из {total} |")
    for item in ITEMS:
        reference, engine = results.verdicts[item.key]
        lines.append(f"| {item.key} | {item.title} | {_mark(reference.state)} | {_mark(engine.state)} |")
    lines += [
        "",
        f"Отметки: {_mark(PASS)} · {_mark(FAIL)} · {_mark(BOUNDARY)} — число ближе к порогу, чем две погрешности, "
        f"по 30 повторам нельзя сказать уверенно; **не считается выполненным** · {_mark(INFO)} — порога нет.",
        "",
        "**Главное простыми словами.**",
        "",
        *_summary_lines(results),
        "",
        "**Как читать числа.** Частота — среднее за 30 повторов по 1 с. «± X» — погрешность этого среднего "
        "(насколько оно могло бы сдвинуться при других случайных импульсах стимула). «Активный нейрон» — "
        f"не меньше {number(ACTIVE_HZ, 0)} Гц, то есть в среднем хотя бы один спайк за секунду (D-018). "
        "Стороны «левый/правый» — со стороны самой мухи (D-011).",
        "",
        "## А. Движок против эталона",
        "",
        f"Для каждого условия три порога D-018: **А1** — корреляция средних частот не меньше {number(CORR_MIN, 2)} "
        "(считается по нейронам, активным хотя бы в одном расчёте); **А2** — совпадение наборов активных нейронов "
        f"не меньше {number(100 * OVERLAP_MIN, 0)} %: общие активные, делённые на активных хотя бы в одном расчёте; **А3** — "
        f"10 ключевых нейронов (MN9, aDN1, aDN2, aBN1, GF с обеих сторон) отличаются не больше чем на "
        f"{number(100 * KEY_REL, 0)} % или {number(KEY_ABS_HZ, 0)} Гц. Стимулируемые нейроны в А1 и А2 не считаются: их включаем "
        "мы сами.",
        "",
        *_a_table(results),
        "",
        "*(доп.)* — условия, добавленные для пункта Д1 после того, как увидели эталон. **Шумовой предел** — тот "
        "же расчёт движка с другим зерном случайных чисел: если он сам не дотягивает до порога, расхождение с "
        "эталоном объясняется случайностью, а не ошибкой в движке. На отметку он не влияет.",
        "",
        f"![Частоты нейронов: эталон против движка](validation/{FIGURE_RATES})",
        "",
        "*Каждая точка — нейрон, активный хотя бы в одном расчёте. Точки на серой линии — частоты совпадают. "
        "Шкала до 1 Гц обычная, дальше — логарифмическая (1, 10, 100 Гц на равных расстояниях).*",
        "",
        "## Б. Выводы статьи",
        "",
        "Одно и то же правило применяется к эталону и к движку. Так видно, где вывод статьи не повторяется "
        "уже в оригинальном коде (свойство модели и версии 783), а где расходится наш движок. «Номер из "
        "статьи» — конкретный нейрон, записанный в `figures.ipynb` Shiu; MN9 из статьи — правый, на стороне, "
        "противоположной стимулу.",
        "",
        f"![Ключевые нейроны по пунктам: эталон и движок](validation/{FIGURE_KEYS})",
        "",
        f"*Высота столбика — средняя частота; чёрточки — ± две погрешности (зона «на границе», правило D-031); "
        f"серая линия — порог «активного» {number(ACTIVE_HZ, 0)} Гц.*",
        "",
    ]
    for item in ITEMS:
        if item.key == "В":
            lines += ["## В. Эксперимент вне статьи", ""]
        if item.key == "Д1":
            lines += ["## Д. Дополнительно (добавлено после эталона)", "",
                      "Пункт «Д» не входил в зафиксированные заранее пороги D-018. Он добавлен, когда стало видно, "
                      "что вода 100 Гц (Б4) не включает MN9 уже в оригинальном Brian2: так проверяется главный "
                      "вывод статьи про воду.", ""]
        lines += _item_section(item, results)
    lines += [
        "## Честность: что задано не коннектомом и что уточнено после эталона",
        "",
        "- Пороги А и параметры модели не менялись (D-005, D-018).",
        "- **[ЧЕСТНОСТЬ]** Б2 считает «откликнувшихся» по правилу статьи (> 0 Гц) — исправлено после сверки "
        "эталона с текстом статьи (D-031).",
        f"- **[ЧЕСТНОСТЬ]** Отметка «на границе» (ближе {number(BOUNDARY_SE, 0)} погрешностей к порогу) введена в C6, "
        "после того как в эталоне увидели aBN1 = 1,0 Гц от JO-F. Она ничего не делает зелёным.",
        "- **[ЧЕСТНОСТЬ]** Пункт Д1 и три его условия добавлены после того, как увидели эталон; правило записано "
        "до их расчёта.",
        "- **[ЧЕСТНОСТЬ]** Способ перемешивания связей (Б7) — наш (D-029). Опыт с тенью (В), его группа нейронов "
        "и GF выбраны нами (D-013). Списки стимулов — из статьи, номера версии 630, не найденные в 783, не "
        "подменены (D-012).",
        "- Все стимулы — прямое включение нейронов, а не настоящий вкус, касание или тень.",
        "",
        "## Как получено",
        "",
        *_technical_lines(results),
        "",
    ]
    return "\n".join(lines)


# ---------- картинки ----------

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=8, length=3, color=AXIS)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _comma(value: float, _position=None) -> str:
    text = f"{value:g}"
    return text.replace(".", ",")


def figure_rates(results: Results, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    conditions = results.conditions
    columns = 5
    rows = math.ceil(len(conditions) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.15 * rows), facecolor=SURFACE)
    for ax in axes.flat[len(conditions):]:
        ax.set_visible(False)
    ticks = [0, 1, 10, 100]
    for ax, condition in zip(axes.flat, conditions):
        base, test = results.reference[condition.key].rates, results.engine[condition.key].rates
        counted = np.ones(len(base), dtype=bool)
        counted[results.stimulated[condition.key]] = False
        shown = ((base >= ACTIVE_HZ) | (test >= ACTIVE_HZ)) & counted
        _style(ax)
        ax.set_xscale("symlog", linthresh=1, linscale=0.4)
        ax.set_yscale("symlog", linthresh=1, linscale=0.4)
        ax.plot([0, 400], [0, 400], color=MUTED, linewidth=1, zorder=1)
        ax.scatter(base[shown], test[shown], s=11, color=COLOR_REFERENCE, alpha=0.7, linewidths=0.4,
                   edgecolors=SURFACE, zorder=2)
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FuncFormatter(_comma))
            axis.set_minor_locator(FixedLocator([]))
        agreement = results.agreements[condition.key]
        ax.set_title(condition.name_ru, fontsize=9, color=INK, loc="left")
        ax.text(0.03, 0.97, f"r = {_corr(agreement)}\nсовпадение {number(100 * agreement.overlap, 0)} %\n"
                            f"нейронов: {agreement.n_union}\nА: {'да' if agreement.ok else 'нет'}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color=INK_2)
    fig.supxlabel("эталон Brian2, Гц", color=INK_2, fontsize=10)
    fig.supylabel("движок, Гц", color=INK_2, fontsize=10)
    fig.suptitle("Пункт А: средние частоты нейронов, эталон против движка (по 30 повторов)", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def figure_b2(results: Results, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter

    reference, engine = results.verdicts["Б2"]
    freqs = sorted(reference.values["counts"])
    fig, ax = plt.subplots(figsize=(7.2, 4.4), facecolor=SURFACE)
    _style(ax)
    # Логарифмическая шкала: «десятки» и «сотни» — две равные полосы, и малые числа при 10 Гц не слипаются.
    ax.set_yscale("log")
    ax.set_ylim(5, 2000)
    ax.yaxis.set_major_locator(FixedLocator([10, 30, 100, 300, 1000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    for low, word in ((10, "десятки"), (100, "сотни")):
        ax.text(1.01, low * math.sqrt(10), word, transform=ax.get_yaxis_transform(), fontsize=8, color=MUTED,
                va="center")
    for verdict, color, label in ((reference, COLOR_REFERENCE, "эталон Brian2 (версия 783)"),
                                  (engine, COLOR_ENGINE, "движок (версия 783)")):
        ax.plot(freqs, [verdict.values["counts"][f] for f in freqs], color=color, linewidth=2, marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3,
                solid_capstyle="round", solid_joinstyle="round")
    ax.scatter(list(PAPER_B2), list(PAPER_B2.values()), marker="D", s=46, color=COLOR_PAPER, edgecolors=SURFACE,
               linewidths=1.5, label="статья (версия 630)", zorder=4)
    # Табличка с числами: у бирюзового цвета статьи низкий контраст, поэтому её числа написаны текстом.
    first, last = freqs[0], freqs[-1]
    table = "\n".join([f"{'':<8}{first:>6} Гц{last:>7} Гц",
                       *(f"{name:<8}{values[first]:>9}{values[last]:>10}" for name, values in (
                           ("эталон", reference.values["counts"]), ("движок", engine.values["counts"]),
                           ("статья", PAPER_B2)))])
    ax.text(0.98, 0.04, table, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=INK_2,
            family="DejaVu Sans Mono")
    ax.set_xticks(freqs)
    ax.set_xlabel("частота сахарных нейронов, Гц", color=INK_2, fontsize=9)
    ax.set_ylabel("откликнулось нейронов (> 0 Гц)", color=INK_2, fontsize=9)
    ax.set_title("Пункт Б2: откликнувшиеся нейроны (без стимулируемых)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def _bars(ax, reference: Verdict, engine: Verdict, labels: list[str], label_size: float = 7) -> float:
    """Пары столбиков «эталон / движок» с чёрточками ± две погрешности и линией порога; возвращает верх."""
    width = 0.36
    x = np.arange(len(labels))
    _style(ax)
    ax.grid(axis="x", visible=False)
    top = 0.0
    for shift, verdict, color in ((-width / 2 - 0.01, reference, COLOR_REFERENCE),
                                  (width / 2 + 0.01, engine, COLOR_ENGINE)):
        means = np.array([verdict.values[label][0] for label in labels])
        errors = np.array([BOUNDARY_SE * verdict.values[label][1] for label in labels])
        ax.bar(x + shift, means, width=width, color=color, zorder=2)
        ax.errorbar(x + shift, means, yerr=errors, fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5, zorder=3)
        for xi, mean, error in zip(x + shift, means, errors):
            ax.text(xi, mean + error, number(mean, 2 if 0 < mean < 2 else 1), ha="center", va="bottom",
                    fontsize=label_size, color=INK_2)
        top = max(top, float((means + errors).max()))
    ax.axhline(ACTIVE_HZ, color=MUTED, linewidth=1, zorder=1)
    ax.set_xticks(x, labels, fontsize=8, color=INK_2)
    return top


def figure_keys(results: Results, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.8), facecolor=SURFACE)
    for ax, (key, title) in zip(axes.flat, FIGURE_PANELS.items()):
        reference, engine = results.verdicts[key]
        labels = list(reference.values)
        top = _bars(ax, reference, engine, labels)
        ax.set_ylim(0, max(2.0, top * 1.18))
        ax.set_title(f"{title}\nэталон: {MARK_WORDS[reference.state]} · движок: {MARK_WORDS[engine.state]}",
                     fontsize=9, color=INK, loc="left")
        ax.set_ylabel("Гц", color=INK_2, fontsize=8)
        if key == "Б6":
            # Граница видна только вблизи: JO-F отдельно на шкале 0–2 Гц, с порогом 1 Гц.
            inset = ax.inset_axes([0.5, 0.4, 0.46, 0.48])
            _bars(inset, reference, engine, ["JO-F"], label_size=8)
            inset.set_ylim(0, 2)
            inset.set_title("JO-F крупно (0–2 Гц)", fontsize=8, color=INK_2, loc="left")
            inset.set_facecolor(SURFACE)
    fig.legend(handles=[Patch(color=COLOR_REFERENCE, label="эталон Brian2"), Patch(color=COLOR_ENGINE, label="движок")],
               loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=False, fontsize=9, labelcolor=INK_2,
               ncols=2)
    fig.suptitle(f"Пункты Б, В, Д: частоты нейронов, по которым решается пункт (серая линия — порог "
                 f"{number(ACTIVE_HZ, 0)} Гц; чёрточки — ± две погрешности)", color=INK, fontsize=11, x=0.01,
                 ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def write_report(results: Results) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figure_rates(results, FIGURES_DIR / FIGURE_RATES)
    figure_b2(results, FIGURES_DIR / FIGURE_B2)
    figure_keys(results, FIGURES_DIR / FIGURE_KEYS)
    text = render_markdown(results, datetime.now().astimezone().strftime("%d.%m.%Y в %H:%M"))
    with open(REPORT_FILE, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return REPORT_FILE
