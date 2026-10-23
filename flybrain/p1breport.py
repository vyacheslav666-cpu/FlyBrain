"""Таблицы и картинки уточнения P1b: `docs/psy/p1b.md` и `docs/psy/p1b_*.png` (D-058…D-061).

Файл собирается заново при каждом `flybrain psy` — вручную его не правят. Отчёт P1 (`docs/psy/results.md`)
этим модулем не меняется. Выводы словами — в `docs/EXPERIMENT_PSY.md`, раздел «P1b».
"""

import json
from pathlib import Path

import numpy as np

from flybrain.psy import Results, readout_threshold
from flybrain.psymech import IGNITED_HZ, QUIET_HZ, VARIANTS, circuit_facts, rate_series
from flybrain.psyreport import CATEGORICAL, DOCS_PSY, INK, INK_2, MUTED, SURFACE, _dose, _figure, \
    _finish, _n, _pct, _stimulus_word, _style, _table
from flybrain.serotonin import saturation
from flybrain.validate import engine_version, hz, number

P1B_FILE = DOCS_PSY / "p1b.md"
P1B_SUMMARY = DOCS_PSY / "p1b_summary.json"
GAP_MV = 7.0  # зазор «покой (−52 мВ) — порог (−45 мВ)» в модели Shiu (D-005)
MODE_RU = {"p1": "P1: все мишени возбуждаются", "kc_inhibit": "P1b: KC тормозятся"}
MODE_SHORT = {"p1": "P1", "kc_inhibit": "KC тормозятся"}
COLOR_MODE = {"p1": CATEGORICAL[0], "kc_inhibit": CATEGORICAL[1]}
PROTOCOL_RU = {"noise": "с шумом σ, без стимула", "stimulus": "без шума, зажигание стимулом"}
PROTOCOL_SHORT = {"noise": "шум σ", "stimulus": "без шума, стимул"}
PHASE_RU = {"A": "доза разгона", "B": "доза 0", "C": "доза 0, шум выключен", "D": "после сброса потенциалов"}


# ---------- выборка расчётов ----------

def mode_specs(results: Results, part: str, mode: str) -> dict:
    """{(условие или σ, доза, частота): расчёт} режима; у «KC тормозятся» доза 0 — расчёт P1 (он тот же)."""
    crit = results.setup.config.criterion
    p1 = [s for s in results.runs if s.part == part and s.criterion == crit and s.mode == "p1"]
    if mode == "p1":
        chosen = p1
    else:
        chosen = [s for s in results.runs if s.part == f"kc_{part}" and s.mode == "kc_inhibit"] + \
                 [s for s in p1 if s.dose == 0]
    out = {}
    for s in chosen:
        out[(s.condition if s.condition else s.sigma, s.dose, s.hz)] = s
    return out


def runaway_of(results: Results, mode: str) -> float | None:
    specs = sorted(mode_specs(results, "runaway", mode).values(), key=lambda s: s.dose)
    return next((s.dose for s in specs if results.metrics[s].runaway), None)


# ---------- разделы ----------

def _circuit_lines(results: Results) -> list[str]:
    f = circuit_facts(results.setup)
    links = f["links"]
    names = {"kc": "KC", "apl": "APL", "dan": "DAN", "dpm": "DPM"}
    rows = [[f"{names[a]} → {names[b]}", _n(links[f'{a}->{b}'][0]), _n(links[f'{a}->{b}'][1])]
            for a, b in (k.split("->") for k in links)]
    per = f["per_kc_input"]
    apl = "; ".join(f"{a['side']}: медиатор {a['nt']}, знак в модели {a['sign']:+d}" for a in f["apl"])
    lines = ["## 1. Схема грибовидных тел", "",
             f"Клеток Кеньона (KC) — {_n(f['counts']['kc'])}, APL — {_n(f['counts']['apl'])} ({apl}), "
             f"дофаминовых нейронов (DAN) — {_n(f['counts']['dan'])}, DPM — {_n(f['counts']['dpm'])}. "
             f"Предсказанный медиатор KC: " + ", ".join(f"{k} {_n(v)}" for k, v in f["kc_nt"].items())
             + " (в литературе KC — ацетилхолин; в модели знак всё равно +1). Знак DPM в модели — "
             + ", ".join(f"{s:+d}" for s in f["dpm_sign"]) + "; DAN — "
             + ", ".join(f"{k:+d}: {_n(v)}" for k, v in f["dan_sign"].items()) + ".", ""]
    lines += _table(["связь", "пар нейронов", "синапсов со знаком модели"], rows)
    lines += ["В среднем на одну KC приходится синапсов (со знаком): от других KC "
              f"{number(per['kc'], 1)}, от проекционных нейронов усиковой доли {number(per['alpn'], 1)}, от APL "
              f"{number(per['apl'], 1)}, от DAN {number(per['dan'], 1)}, от DPM {number(per['dpm'], 1)}.", ""]
    return lines


def mechanism_table(results: Results) -> dict[str, list[tuple[float, int, int, float, float]]]:
    """{вариант: [(σ, повторов «не гаснет», всего, частота KC во время шума, Гц, спайков/с через 300–500 мс)]}."""
    setup = results.setup
    kc = setup.kc_mask
    out = {}
    for spec in sorted((s for s in results.runs if s.part == "mech"), key=lambda s: s.sigma):
        m = results.metrics[spec]
        rate_kc = float(results.runs[spec].totals[kc].sum() / kc.sum() / (spec.trials * spec.stimulus_ms / 1000))
        out.setdefault(spec.variant, []).append((spec.sigma, m.sustained_trials, spec.trials, rate_kc,
                                                  float(m.after_rate)))
    return out


def ignition_sigma(rows: list[tuple], share: float) -> float | None:
    return next((row[0] for row in rows if row[1] >= share * row[2] and row[1] > 0), None)


def _mechanism_lines(results: Results) -> list[str]:
    b = results.setup.config.p1b
    table = mechanism_table(results)
    if not table:
        return []
    sigmas = sorted({row[0] for rows in table.values() for row in rows})
    lines = ["## 2. Механизм «зажигания» (доза 0)", "",
             f"Доза 0, без стимула: шум мембраны σ на {_n(b.mech_noise_ms)} мс, затем шум выключен на "
             f"{_n(b.mech_off_ms)} мс. «Зажглось» — то же правило, что у разгона (D-056): через 300–500 мс после "
             f"выключения шума спайков не меньше 10 % от последних 200 мс с шумом. {b.mech_trials} повторов. "
             "**[ЧЕСТНОСТЬ]** Варианты связей — проверка свойства модели Shiu, а не новая модель (D-058).", ""]
    rows = []
    for variant in b.mech_variants:
        got = {row[0]: row[1:] for row in table.get(variant, [])}
        cells = [f"{got[s][0]} из {got[s][1]}; KC {number(got[s][2], 1)} Гц; после — {_n(got[s][3])}/с" if s in got
                 else "—" for s in sigmas]
        first = ignition_sigma(table.get(variant, []), 0.0)
        half = ignition_sigma(table.get(variant, []), 0.5)
        rows.append([VARIANTS[variant]] + cells + [
            "не зажигается" if first is None else f"{number(first, 2)} мВ",
            "—" if half is None else f"{number(half, 2)} мВ"])
    lines += ["В клетке: в скольких повторах активность после выключения шума не гаснет; средняя частота клеток "
              "Кеньона во время шума; спайков в секунду во всём мозге через 300–500 мс после выключения шума.", ""]
    lines += _table(["вариант связей"] + [f"σ = {number(s, 2)} мВ" for s in sigmas]
                    + ["наименьший σ с зажиганием", "σ, где зажигается хотя бы половина повторов"], rows)
    checks = sorted((s for s in results.runs if s.part == "mech_runaway"), key=lambda s: s.dose)
    if checks:
        base = mode_specs(results, "runaway", "p1")
        rows = []
        for s in checks:
            m, ref = results.metrics[s], base.get((s.condition, s.dose, s.hz))
            rows.append([_dose(s.dose),
                         "—" if ref is None else f"{results.metrics[ref].sustained_trials} из {ref.trials} "
                                                 f"({_n(results.metrics[ref].active)} активных)",
                         f"{m.sustained_trials} из {s.trials} ({_n(m.active)} активных)"])
        lines += [f"Разгон от дозы (P1, {results.setup.config.condition(checks[0].condition).name_ru}, без шума, правило "
                  f"D-056) в исходной модели и в варианте «{VARIANTS[checks[0].variant]}» — повторов, где активность "
                  "после стимула не гаснет:", ""]
        lines += _table(["доза d", "исходная модель", VARIANTS[checks[0].variant]], rows)
    # Кто горит после выключения шума в исходной модели — при наименьшем σ, где зажглось.
    base = sorted((s for s in results.runs if s.part == "mech" and s.variant == "base"), key=lambda s: s.sigma)
    lit = [s for s in base if results.metrics[s].sustained_trials]
    if lit:
        from flybrain.psyreport import _classes

        s = lit[0]
        lines += [f"Исходная модель, σ = {number(s.sigma, 2)} мВ: кто стреляет через 0–500 мс после выключения шума "
                  "(класс клеток, нейронов): " + _classes(results.metrics[s].after_classes) + ".", ""]
    return lines


def unit_rows(results: Results) -> tuple[list[list[str]], float, float, float, float]:
    setup = results.setup
    s = setup.synapses[setup.config.criterion]
    targets = s[s > 0]
    median, peak = float(np.median(targets)), float(targets.max())
    h_med, h_max = float(saturation(np.array([median]), setup.config.s_half)[0]), \
        float(saturation(np.array([peak]), setup.config.s_half)[0])
    b = setup.config.p1b
    doses = sorted({*setup.config.main_doses, *(b.kc_main_doses if b else ()), 7.0})
    rows = []
    for d in doses:
        med, top = d * setup.config.delta_v_mv * h_med, d * setup.config.delta_v_mv * h_max
        rows.append([_dose(d), f"{number(med, 2)} мВ ({_pct(med / GAP_MV)})", f"{number(top, 2)} мВ ({_pct(top / GAP_MV)})"])
    return rows, median, peak, h_med, h_max


def _units_lines(results: Results) -> list[str]:
    rows, median, peak, h_med, h_max = unit_rows(results)
    lines = ["## 3. Единицы дозы", "",
             f"Снижение порога = d · ΔV · h(s), ΔV = {number(results.setup.config.delta_v_mv, 1)} мВ (D-055). У мишени "
             f"с медианным числом серотониновых синапсов (s = {_n(median)}, h = {number(h_med, 3)}) и с наибольшим "
             f"(s = {_n(peak)}, h = {number(h_max, 3)}); в скобках — доля зазора «покой — порог» {_n(GAP_MV)} мВ. "
             "В режиме «KC тормозятся» у клеток Кеньона на столько же порог **повышается**.", ""]
    lines += _table(["доза d", "мишень с медианным s", "мишень с наибольшим s"], rows)
    return lines


def _compare_main_lines(results: Results) -> list[str]:
    setup, config = results.setup, results.setup.config
    p1, kc = mode_specs(results, "main", "p1"), mode_specs(results, "main", "kc_inhibit")
    if not kc:
        return []
    lines = ["## 4. Режим «KC тормозятся»: сравнение с P1", "",
             "У клеток Кеньона порог от дозы повышается на d · ΔV · h(s), у остальных мишеней — снижается, как в P1 "
             f"(D-060). {config.trials} повторов, строгий критерий, те же стимулы и метрики. В клетке — P1 / KC "
             "тормозятся.", ""]
    for mode in ("p1", "kc_inhibit"):
        dose = runaway_of(results, mode)
        lines.append(f"- Доза разгона ({MODE_RU[mode]}, сахар 40 Гц, 10 повторов): "
                     f"**{'не найдена в сетке' if dose is None else _dose(dose)}**.")
    lines.append("")
    specs = sorted(mode_specs(results, "runaway", "kc_inhibit").values(), key=lambda s: s.dose)
    if specs:
        rows = [[_dose(s.dose), _n(results.metrics[s].responders_b2), _n(results.metrics[s].active),
                 f"{results.metrics[s].sustained_trials} из {s.trials}"] for s in specs]
        lines += ["Поиск дозы разгона в режиме «KC тормозятся»:", ""]
        lines += _table(["доза d", "откликнулось (Б2)", "активных (≥ 1 Гц)", "повторов, где не гаснет"], rows)
    doses = sorted({k[1] for k in kc} | {k[1] for k in p1})
    for c in config.conditions:
        k = setup.key_index(c.readout_root_id)
        rows = []
        for d in doses:
            a, b = p1.get((c.key, d, c.hz)), kc.get((c.key, d, c.hz))
            if a is None and b is None:
                continue

            def cell(fn, spec=None):
                return "—" if spec is None else fn(results.metrics[spec])

            pair = lambda fn: f"{cell(fn, a)} / {cell(fn, b)}"  # noqa: E731
            rows.append([_dose(d), pair(lambda m: _n(m.responders_b2)), pair(lambda m: _n(m.active)),
                         pair(lambda m: number(m.n_eff, 0)), pair(lambda m: hz(*m.key_rates[k])),
                         pair(lambda m: _n(m.visual_system)), pair(lambda m: f"{m.sustained_trials}")])
        lines += [f"### {c.name_ru} — «исполнитель» {setup.keys[k].label}", ""]
        lines += _table(["доза d", "откликнулось (Б2)", "активных", "эффективное число нейронов",
                         setup.keys[k].label, "в зрительной системе", "повторов из 30, где не гаснет"], rows)
    return lines


def _compare_threshold_lines(results: Results) -> list[str]:
    setup, config = results.setup, results.setup.config
    p1, kc = mode_specs(results, "thresholds", "p1"), mode_specs(results, "thresholds", "kc_inhibit")
    if not kc:
        return []
    lines = ["## 5. Порог «исполнителей»: P1 и «KC тормозятся»", "",
             f"{config.threshold_trials} повторов на точку; порог — наименьшая частота стимула, при которой "
             "«исполнитель» активен (≥ 1 Гц).", ""]
    doses = sorted({key[1] for key in kc} | {key[1] for key in p1})
    rows = []
    for c in config.conditions:
        k = setup.key_index(c.readout_root_id)
        cells = []
        for d in doses:
            texts = []
            for specs in (p1, kc):
                points = [(f, results.metrics[specs[(c.key, d, f)]].key_rates[k][0]) for f in c.threshold_hz
                          if (c.key, d, f) in specs]
                kind, value = readout_threshold(points)
                prefix = {"между": "", "не выше": "≤ ", "выше": "> "}.get(kind)
                texts.append("—" if prefix is None else f"{prefix}{number(value, 0)}")
            cells.append(" / ".join(texts))
        rows.append([f"{_stimulus_word(c.name_ru)} → {setup.keys[k].label}"] + cells)
    lines += _table(["стимул → исполнитель, Гц: P1 / KC тормозятся"] + [f"d = {_dose(d)}" for d in doses], rows)
    return lines


def _compare_spont_lines(results: Results) -> list[str]:
    b = results.setup.config.p1b
    p1, kc = mode_specs(results, "spont", "p1"), mode_specs(results, "spont", "kc_inhibit")
    p1 = {k: v for k, v in p1.items() if k[0] == b.kc_spont_sigma_mv}
    if not kc:
        return []
    doses = sorted({key[1] for key in kc} | {key[1] for key in p1})
    rows = []
    for d in doses:
        a, c = p1.get((b.kc_spont_sigma_mv, d, 0.0)), kc.get((b.kc_spont_sigma_mv, d, 0.0))
        rows.append([_dose(d)] + [("—" if s is None else f"{_n(results.metrics[s].active)} "
                                  f"({number(results.metrics[s].mean_rate, 2)} Гц)") for s in (a, c)])
    return ["## 6. Без стимула, шум σ = " + number(b.kc_spont_sigma_mv, 0) + " мВ", "",
            "Активных нейронов (≥ 1 Гц; в скобках — средняя частота по мозгу), 10 повторов, первые 200 мс не "
            "считаются.", ""] + _table(["доза d", "P1", "KC тормозятся"], rows)


def _hysteresis_lines(hysteresis: list[dict], results: Results) -> list[str]:
    b = results.setup.config.p1b
    if not hysteresis:
        return ["## 7. Гистерезис", "", "Не посчитан: нет дозы разгона ни у одного режима.", ""]
    lines = ["## 7. Гистерезис: гаснет ли сеть, когда доза возвращается в 0", "",
             f"Фазы: A — доза разгона режима {_n(b.hyst_on_ms)} мс; B — доза 0 на {_n(b.hyst_back_ms)} мс (стимул "
             f"выключен, шум прежний); C — шум выключен на {_n(b.hyst_quiet_ms)} мс; D — потенциалы сброшены в покой, "
             f"шум снова включён на {_n(b.hyst_reset_ms)} мс. «Горит» — больше {_n(IGNITED_HZ)} спайков в секунду "
             f"во всём мозге в последние 100 мс фазы; «погасла» — меньше {_n(QUIET_HZ)} и до конца фазы не "
             f"поднимается. Протокол без шума зажигает сеть стимулом «{results.setup.config.condition(b.hyst_condition).name_ru}». "
             f"{b.hyst_trials} повторов.", ""]
    rows = []
    for h in hysteresis:
        p, s = h["plan"], h["summary"]
        lit_a = [k for k, v in enumerate(s["lit"]["A"]) if v]
        outs = [s["out_ms"][k] for k in lit_a]
        gone = [o for o in outs if o is not None]
        when = "—" if not lit_a else (
            f"погасла в {len(gone)} из {len(lit_a)}" + (f": через {number(min(gone), 0)}–{number(max(gone), 0)} мс"
                                                       if gone else ""))
        rows.append([MODE_SHORT[p["mode"]], PROTOCOL_RU[p["protocol"]], _dose(p["dose"]),
                     number(p["sigma"], 0)] + [f"{sum(s['lit'][ph])} из {len(s['lit'][ph])}" for ph in "ABCD"]
                    + [when])
    lines += _table(["режим", "протокол", "доза разгона", "σ, мВ"] + [f"горит: {PHASE_RU[ph]}" for ph in "ABCD"]
                    + ["после возврата дозы в 0 (среди зажёгшихся)"], rows)
    return lines


def render(results: Results, hysteresis: list[dict]) -> str:
    lines = ["# P1b «психоделик», уточнение: таблицы", "",
             "*Файл собирает команда `uv run flybrain psy` — вручную его не правят. Цель, оговорки и выводы — в "
             "[EXPERIMENT_PSY.md](../EXPERIMENT_PSY.md), раздел «P1b». Таблицы P1 — в [results.md](results.md), "
             "они не менялись.*", "",
             "**[ЧЕСТНОСТЬ]** Вычислительная гипотеза, а не модель ощущений мухи. Режим «KC тормозятся», варианты связей, "
             "протокол гистерезиса — наши (D-058…D-061).", ""]
    for part in (_circuit_lines, _mechanism_lines, _units_lines, _compare_main_lines, _compare_threshold_lines,
                 _compare_spont_lines):
        lines += part(results)
    lines += _hysteresis_lines(hysteresis, results)
    seconds = [r.seconds for s, r in results.runs.items() if s.part.startswith(("mech", "kc_"))]
    lines += ["## 8. Как получено", "",
              f"- Движок — тот же, что в P1 и в проверке C6 (версия кода `{engine_version()}`): P1b его не менял.",
              "- Режим «KC тормозятся» при дозе 0 и варианты связей «base» — это ровно расчёты P1 (тот же ключ и то же "
              "зерно); тест `tests/test_psy.py`.",
              f"- Расчётов P1b: {len(seconds)}" + (f"; один — от {number(min(seconds), 0)} до {number(max(seconds), 0)} с"
                                                   if seconds else "") + ". Код — `flybrain/psymech.py`, "
              "`flybrain/p1breport.py`, настройки — раздел `p1b` в `config/psy.yaml`.", ""]
    figures = [p for p in FIGURES if (DOCS_PSY / p).exists()]
    if figures:
        lines += ["## Картинки", ""] + [f"![{FIGURES[p]}]({p})\n\n*{FIGURES[p]}*\n" for p in figures]
    return "\n".join(lines).rstrip() + "\n"


# ---------- картинки ----------

FIGURES = {
    "p1b_mechanism.png": "Механизм «зажигания»: доля повторов, где активность не гаснет после выключения шума",
    "p1b_compare.png": "P1 и «KC тормозятся»: исполнители и незатухающая активность",
    "p1b_hysteresis.png": "Гистерезис: активность мозга, когда доза возвращается в 0",
}


def figure_mechanism(results: Results, path: Path) -> bool:
    """Тепловая карта «вариант связей × шум»: в скольких повторах из 10 активность после шума не гаснет."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    table = mechanism_table(results)
    if not table:
        return False
    variants = [v for v in results.setup.config.p1b.mech_variants if v in table]
    sigmas = sorted({row[0] for rows in table.values() for row in rows})
    share = np.full((len(variants), len(sigmas)), np.nan)
    labels = [["" for _ in sigmas] for _ in variants]
    for i, v in enumerate(variants):
        for sigma, k, n, rate_kc, _after in table[v]:
            j = sigmas.index(sigma)
            share[i, j] = k / n
            labels[i][j] = f"{k}/{n}\nKC {number(rate_kc, 1)}"
    cmap = LinearSegmentedColormap.from_list("seq", [SURFACE, "#9ec5f4", "#2a6fc4", "#123a6e"])
    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(variants) + 1.8), facecolor=SURFACE)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(variants)):
        for j in range(len(sigmas)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=7,
                    color=SURFACE if share[i, j] > 0.55 else INK_2)
    ax.set_xticks(range(len(sigmas)), [number(s, 2) for s in sigmas], fontsize=8, color=INK_2)
    ax.set_yticks(range(len(variants)), [VARIANTS[v] for v in variants], fontsize=8, color=INK_2)
    ax.set_xlabel("шум σ, мВ", color=INK_2, fontsize=9)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    ax.categorical_x = ax.categorical_y = True
    _finish(fig, plt, path, "Откуда «зажигание»: доза 0, шум 1 с, затем шум выключен",
            "в клетке: повторов, где после шума не гаснет; частота клеток Кеньона во время шума, Гц. "
            "[ЧЕСТНОСТЬ] Варианты — проверка модели Shiu (D-058)")
    return True


def figure_compare(results: Results, path: Path) -> bool:
    import matplotlib.pyplot as plt

    setup, config = results.setup, results.setup.config
    specs = {m: mode_specs(results, "main", m) for m in ("p1", "kc_inhibit")}
    if not specs["kc_inhibit"]:
        return False
    fig, axes = plt.subplots(2, len(config.conditions), figsize=(3.3 * len(config.conditions), 5.8),
                             facecolor=SURFACE, squeeze=False)
    for col, c in enumerate(config.conditions):
        k = setup.key_index(c.readout_root_id)
        for mode in ("p1", "kc_inhibit"):
            got = sorted((key[1], s) for key, s in specs[mode].items() if key[0] == c.key)
            doses = [d for d, _ in got]
            axes[0][col].plot(doses, [results.metrics[s].key_rates[k][0] for _, s in got], color=COLOR_MODE[mode],
                              linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label=MODE_SHORT[mode])
            axes[1][col].plot(doses, [100 * results.metrics[s].sustained_trials / s.trials for _, s in got],
                              color=COLOR_MODE[mode], linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                              label=MODE_SHORT[mode])
        axes[0][col].set_title(f"{c.name_ru}: {setup.keys[k].label}, Гц", fontsize=9, color=INK, loc="left")
        axes[1][col].set_title("повторов, где не гаснет, %", fontsize=9, color=INK, loc="left")
        axes[1][col].set_ylim(-4, 104)
        for row in (0, 1):
            _style(axes[row][col])
            axes[row][col].set_xlabel("доза d", color=INK_2, fontsize=9)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncols=2, frameon=False, fontsize=8, labelcolor=INK_2,
               bbox_to_anchor=(0.995, 0.0))
    _finish(fig, plt, path, "P1 и «KC тормозятся»: частота исполнителя и незатухающая активность",
            "синий — P1 (все мишени возбуждаются), оранжевый — у клеток Кеньона порог от дозы повышается (D-060)")
    return True


def figure_hysteresis(hysteresis: list[dict], path: Path) -> bool:
    import matplotlib.pyplot as plt

    if not hysteresis:
        return False
    fig, axes = _figure(plt, len(hysteresis), 3.4, 3.3)
    for ax, h in zip(axes, hysteresis):
        t, rate = rate_series(h)
        for k in range(rate.shape[1]):
            ax.plot(t, np.maximum(rate[:, k], 1), color=COLOR_MODE[h["plan"]["mode"]], linewidth=0.8, alpha=0.5)
        for name, at in h["phases"][:-1]:
            ax.axvline(at, color=MUTED, linewidth=1, linestyle="--")
        ax.set_yscale("log")
        _style(ax)
        p = h["plan"]
        ax.set_title(f"{MODE_SHORT[p['mode']]}, {PROTOCOL_SHORT[p['protocol']]}", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("время, мс", color=INK_2, fontsize=9)
    axes[0].set_ylabel("спайков/с во всём мозге", color=INK_2, fontsize=9)
    _finish(fig, plt, path, "Гистерезис: доза разгона → доза 0 → шум выключен → сброс потенциалов",
            "каждая линия — повтор; пунктир — смена фазы; ниже 1 спайка/с показано как 1")
    return True


def write_p1b_report(results: Results, hysteresis: list[dict]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    DOCS_PSY.mkdir(parents=True, exist_ok=True)
    written = []
    for name, make in (("p1b_mechanism.png", lambda p: figure_mechanism(results, p)),
                       ("p1b_compare.png", lambda p: figure_compare(results, p)),
                       ("p1b_hysteresis.png", lambda p: figure_hysteresis(hysteresis, p))):
        path = DOCS_PSY / name
        if make(path):
            written.append(path)
        elif path.exists():
            path.unlink()
    P1B_FILE.write_text(render(results, hysteresis), encoding="utf-8", newline="\n")
    summary = {"runaway_dose": {m: runaway_of(results, m) for m in ("p1", "kc_inhibit")},
               "mechanism": {v: [list(r) for r in rows] for v, rows in mechanism_table(results).items()},
               "hysteresis": [{"plan": h["plan"], **h["summary"]} for h in hysteresis]}
    P1B_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    return [P1B_FILE, P1B_SUMMARY, *written]
