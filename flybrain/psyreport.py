"""Таблицы и картинки эксперимента P1 «психоделик»: `docs/psy/results.md` и `docs/psy/*.png` (D-053…D-056).

Файл собирается заново при каждом `flybrain psy` — вручную его не правят. Выводы словами — в
`docs/EXPERIMENT_PSY.md` (его пишет человек, опираясь на эти таблицы).
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from flybrain.modality import MODALITIES, MODALITIES_RU
from flybrain.paths import PROJECT_ROOT
from flybrain.psy import ACTIVE_HZ, VISUAL_SUPER_CLASSES, Results, RunSpec, readout_threshold, runaway_dose
from flybrain.serotonin import CRITERIA, CRITERIA_RU, mentions, saturation, threshold_shift
from flybrain.validate import engine_version, hz, number

DOCS_PSY = PROJECT_ROOT / "docs" / "psy"
RESULTS_FILE = DOCS_PSY / "results.md"
SUMMARY_FILE = DOCS_PSY / "summary.json"

# Палитра навыка dataviz (светлая тема), как в docs/validation (flybrain/report.py).
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# Последовательная шкала одного оттенка (синий, от светлого к тёмному) — для доз и σ.
SEQUENTIAL = ("#9ec5f4", "#6ea8ec", "#3f88df", "#2a6fc4", "#1e5499", "#123a6e")
MODALITY_COLORS = dict(zip(MODALITIES, CATEGORICAL[:5]))
NONE_COLOR = "#c3c2b7"

# Литература о числе серотониновых нейронов (D-054).
LITERATURE = [
    ("около 100 во всей нервной системе (обзорная оценка)", "Alekseyenko et al., Curr Biol 2014", 100),
    ("83 на полушарие в мозге и подглоточном ганглии, то есть около 166 (окраска на серотонин и "
     "триптофангидроксилазу)", "Bao et al., Cell Tissue Res 2010", 166),
]
SUPER_CLASS_RU = {
    "central": "центральные", "optic": "зрительные доли", "visual_projection": "зрительные проекционные",
    "visual_centrifugal": "зрительные центробежные", "sensory": "сенсорные", "sensory_ascending": "сенсорные восходящие",
    "ascending": "восходящие", "descending": "нисходящие", "motor": "моторные", "endocrine": "эндокринные",
    "без аннотации": "без аннотации",
}
NT_RU = {"acetylcholine": "ацетилхолин", "gaba": "ГАМК", "glutamate": "глутамат", "dopamine": "дофамин",
         "serotonin": "серотонин", "octopamine": "октопамин"}


def _n(value: float, digits: int = 0) -> str:
    return number(value, digits)


def _pct(value: float, digits: int = 0) -> str:
    return f"{number(100 * value, digits)} %"


def _dose(value: float) -> str:
    return number(value, 2).rstrip("0").rstrip(",")


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines + [""]


# ---------- раздел 1: разведка данных ----------

def _survey_lines(results: Results) -> list[str]:
    setup = results.setup
    neurons, sets = setup.neurons, setup.sets
    nt = neurons["nt"].fillna("нет")
    lines = ["## 1. Разведка данных", ""]

    lines += ["### 1.1. Как хранится медиатор", "",
              "В данных проекта медиатор — **на уровне нейрона**, а не синапса: столбец `top_nt` таблицы аннотаций "
              "FlyWire (самый частый медиатор, предсказанный по снимкам синапсов этого нейрона, Eckstein et al. 2024) "
              "с уверенностью `top_nt_conf`, и столбец `known_nt` — медиатор из литературы со ссылкой "
              "`known_nt_source`. Предсказаний для отдельных синапсов в скачанных файлах нет. В файле связей Shiu "
              "есть только знак отправителя (+1 / −1), одинаковый для всех его синапсов.", ""]
    rows = []
    for name in ("acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine", "нет"):
        mask = (nt == name).to_numpy()
        sign = neurons["sign"].to_numpy()[mask]
        conf = sets.top_nt_conf[mask]
        rows.append([NT_RU.get(name, "не предсказан"), _n(mask.sum()), _n((sign == 1).sum()), _n((sign == -1).sum()),
                     _n((sign == 0).sum()), number(float(np.nanmedian(conf)), 2) if mask.any() and name != "нет"
                     else "—"])
    lines += ["### 1.2. Какой знак и вес получают синапсы в движке", "",
              "Знак берётся из файла связей Shiu (D-025), вес — число синапсов × 0,275 мВ для любого медиатора (D-005). "
              "Таблица — нейроны модели по предсказанному медиатору:", ""]
    lines += _table(["предсказанный медиатор", "нейронов", "знак +1 (возбуждает)", "знак −1 (тормозит)",
                     "знак 0 (нет выходов)", "медианная уверенность"], rows)
    lines += ["Дофамин, серотонин и октопамин в модели Shiu — **обычные быстрые возбуждающие синапсы** с тем же "
              "весом 0,275 мВ на синапс, что у ацетилхолина. Это логика Shiu (правило «ГАМК и глутамат тормозят, "
              "остальные возбуждают», D-005), а не наше решение; единичные исключения — от другой версии "
              "предсказания медиатора (D-025). Медленного модулирующего действия моноаминов в модели нет.", ""]

    total_lit = ", ".join(f"{text} ({source})" for text, source, _ in LITERATURE)
    rows = []
    for crit in CRITERIA:
        mask = sets.masks[crit]
        sign = neurons["sign"].to_numpy()[mask]
        low, high = LITERATURE[0][2], LITERATURE[1][2]
        rows.append([CRITERIA_RU[crit], _n(mask.sum()), f"{number(mask.sum() / high, 1)}–{number(mask.sum() / low, 1)}×",
                     _n((nt[mask] == "serotonin").sum()), f"{_n((sign == 1).sum())} / {_n((sign == -1).sum())}"])
    lines += ["### 1.3. Серотониновые нейроны", "", f"Литература: {total_lit}.", ""]
    lines += _table(["критерий", "нейронов в модели", "от числа в литературе", "из них предсказан серотонин",
                     "знак в модели: + / −"], rows)

    broad, strict, predicted = sets.masks["broad"], sets.masks["strict"], sets.masks["predicted"]
    known_pred = [sets.known_nt[i] for i in np.flatnonzero(predicted)]
    with_known = sum(1 for k in known_pred if k is not None)
    others = {name: sum(1 for k in known_pred if mentions(k, name)) for name in ("acetylcholine", "histamine")}
    sensory_pred = int((neurons["super_class"].to_numpy()[predicted] == "sensory").sum())
    lines += [f"- Предсказание и литература почти не совпадают: из {_n(broad.sum())} нейронов, у которых серотонин "
              f"указан в литературе, предсказан серотонин только у {_n((broad & predicted).sum())}. Их предсказания: "
              + ", ".join(f"{NT_RU.get(k, k)} {v}" for k, v in nt[broad].value_counts().items()) + ".",
              f"- Из {_n(predicted.sum())} нейронов с предсказанным серотонином {_n(sensory_pred)} — сенсорные. "
              f"Литературный медиатор указан у {_n(with_known)} из них: ацетилхолин — у {_n(others['acetylcholine'])}, "
              f"гистамин (его нет среди шести медиаторов, которые умеет предсказывать сеть) — у "
              f"{_n(others['histamine'])}, серотонин — у {_n((broad & predicted).sum())}.", ""]

    ctype = neurons["cell_type"].fillna("без типа").to_numpy()
    rows = []
    for t in sorted(set(ctype[broad])):
        members = np.flatnonzero(broad & (ctype == t))
        methods = sorted({m for i in members for m in sets.methods[i]})
        preds = pd.Series(nt.to_numpy()[members]).value_counts()
        rows.append([t, _n(len(members)), "✅" if strict[members].all() else "—", ", ".join(methods),
                     ", ".join(f"{NT_RU.get(k, k)} {v}" for k, v in preds.items()),
                     SUPER_CLASS_RU.get(neurons["super_class"].iloc[members[0]], neurons["super_class"].iloc[members[0]])])
    lines += ["Типы клеток с серотонином в литературе (столбец «строгий» — входят ли в основной набор):", ""]
    lines += _table(["тип клетки", "нейронов", "строгий", "чем подтверждён серотонин", "предсказанный медиатор",
                     "суперкласс"], rows)

    lines += ["### 1.4. Мишени: серотониновые синапсы на нейрон", "",
              "s_i — сколько синапсов нейрон i получает от серотониновых нейронов.", ""]
    bins = [(1, 1), (2, 4), (5, 19), (20, 99), (100, None)]
    rows = []
    for crit in CRITERIA:
        s = setup.synapses[crit]
        targets = s[s > 0]
        sources = sets.masks[crit]
        out_syn = int(np.abs(setup.csr["weight"][sources[setup.csr["indices"]]].astype(np.int64)).sum())
        row = [CRITERIA_RU[crit], _n(sources.sum()), _n(out_syn), _n(len(targets)),
               _n(np.median(targets)) if len(targets) else "—"]
        row += [_n(((s >= lo) & (s <= (hi if hi else s.max()))).sum()) for lo, hi in bins]
        rows.append(row)
    lines += _table(["критерий", "источников", "их выходных синапсов", "мишеней (s > 0)", "медиана s у мишеней",
                     "s = 1", "s = 2–4", "s = 5–19", "s = 20–99", "s ≥ 100"], rows)

    for crit in ("strict", "broad"):
        s = setup.synapses[crit]
        frame = pd.DataFrame({"sc": neurons["super_class"].fillna("без аннотации"), "s": s})
        grouped = frame.groupby("sc").agg(n=("s", "size"), targets=("s", lambda x: int((x > 0).sum())),
                                          strong=("s", lambda x: int((x >= 5).sum())), syn=("s", "sum"))
        grouped = grouped.sort_values("syn", ascending=False)
        rows = [[SUPER_CLASS_RU.get(k, k), _n(r.n), _n(r.targets), _pct(r.targets / r.n, 1), _n(r.strong),
                 _pct(r.syn / grouped.syn.sum(), 1)] for k, r in grouped.iterrows()]
        lines += [f"Мишени по суперклассам — критерий «{CRITERIA_RU[crit]}»:", ""]
        lines += _table(["суперкласс", "нейронов", "мишеней (s > 0)", "доля класса", "мишеней с s ≥ 5",
                         "доля серотониновых синапсов"], rows)
        top = pd.DataFrame({"t": neurons["cell_type"].fillna("без типа"), "s": s}).groupby("t").s.sum()
        top = top.sort_values(ascending=False).head(10)
        lines += ["Больше всего серотониновых синапсов получают типы: "
                  + ", ".join(f"{t} ({_n(v)})" for t, v in top.items()) + ".", ""]
    rows = [[k.label, k.root_id, *[_n(setup.synapses[c][k.index]) for c in CRITERIA]] for k in setup.keys]
    lines += ["Ключевые нейроны — их собственные серотониновые синапсы:", ""]
    lines += _table(["нейрон", "номер FlyWire", *[f"s ({CRITERIA_RU[c].split(':')[0]})" for c in CRITERIA]], rows)
    return lines


# ---------- раздел 2: механизм ----------

def _mechanism_lines(results: Results) -> list[str]:
    setup, config = results.setup, results.setup.config
    s = setup.synapses[config.criterion]
    lines = ["## 2. Механизм и параметры", "",
             f"Порог нейрона i: V_th,i = −45 мВ − d · ΔV · h(s_i), h(s) = s / (s + s_half); ΔV = "
             f"{number(config.delta_v_mv, 1)} мВ, s_half = {_n(config.s_half)}; основной критерий — "
             f"«{CRITERIA_RU[config.criterion]}». Веса связей не меняются; при d = 0 порог у всех −45 мВ, как у Shiu.",
             ""]
    rows = [[_n(v), number(float(saturation(np.array([v]), config.s_half)[0]), 2)] for v in (1, 2, 5, 20, 100, 1000)]
    lines += _table(["s (синапсов)", "h(s)"], rows)
    shift = threshold_shift(s, 1.0, config.delta_v_mv, config.s_half)
    lines += [f"При дозе 1 порог снижен у {_n((shift > 0).sum())} нейронов: больше чем на 0,5 мВ — у "
              f"{_n((shift > 0.5).sum())}, больше чем на 0,8 мВ — у {_n((shift > 0.8).sum())}; наибольшее снижение — "
              f"{number(shift.max(), 3)} мВ. Без шума нейрон в покое (−52 мВ) сам не выстрелит, пока порог не опустится "
              f"ниже покоя: d · ΔV · h > 7 мВ, то есть при d > {number(7 / (config.delta_v_mv * shift.max()), 2)} у самого "
              "«нагруженного» нейрона.", ""]
    return lines


# ---------- раздел 3: разгон ----------

def _runaway_lines(results: Results) -> list[str]:
    setup, config = results.setup, results.setup.config
    lines = ["## 3. Поиск дозы разгона", "",
             f"Стимул — {config.condition(config.runaway_condition).name_ru}, {config.runaway_trials} повторов: 1 с со "
             f"стимулом, затем {_n(config.after_ms)} мс без него. **Разгон** (правило D-056, записано до расчётов): за "
             f"{_n(config.after_window_ms[0] - config.stimulus_ms)}–{_n(config.after_window_ms[1] - config.stimulus_ms)} мс "
             "после выключения стимула спайков не стимулируемых нейронов не меньше "
             f"{_pct(config.min_ratio)} от последних {_n(config.late_window_ms[1] - config.late_window_ms[0])} мс стимула. "
             "Без шума модель Shiu после выключения стимула затихает, так что любая поздняя активность — "
             "самоподдерживающаяся.", ""]
    for crit in (config.criterion, config.sensitivity_criterion):
        specs = sorted(results.find(part="runaway", criterion=crit), key=lambda sp: sp.dose)
        if not specs:
            continue
        rows = []
        for spec in specs:
            m = results.metrics[spec]
            rows.append([_dose(spec.dose), _n(m.responders_b2), _n(m.active), number(m.n_eff, 0),
                         number(m.late_rate, 0), number(m.after_rate, 0),
                         "—" if m.after_ratio is None else (_pct(m.after_ratio, 1) if math.isfinite(m.after_ratio)
                                                            else "∞"),
                         f"{m.sustained_trials} из {spec.trials}", _n(m.after_neurons),
                         "🔥 да" if m.runaway else "нет"])
        dose = runaway_dose(results, crit)
        lines += [f"### Критерий «{CRITERIA_RU[crit]}»", "",
                  f"Доза разгона: **{'не найдена в сетке' if dose is None else _dose(dose)}**.", ""]
        lines += _table(["доза d", "откликнулось (> 0 Гц, со стимулируемыми)", "активных (≥ 1 Гц)",
                         "эффективное число нейронов", "спайков/с в конце стимула", "спайков/с после",
                         "после / конец стимула", "повторов, где не гаснет", "нейронов после стимула",
                         "разгон"], rows)
        flagged = [s for s in specs if results.metrics[s].runaway][:2]
        for spec in flagged:
            lines += [f"Кто стреляет через 0–500 мс после стимула при дозе {_dose(spec.dose)} (класс клеток, нейронов): "
                      + _classes(results.metrics[spec].after_classes) + ".", ""]
    return lines


def _classes(counts: dict[str, int], top: int = 8) -> str:
    items = list(counts.items())
    text = ", ".join(f"{k} {_n(v)}" for k, v in items[:top])
    rest = sum(v for _, v in items[top:])
    return text + (f", прочие {_n(rest)}" if rest else "") if items else "никто"


# ---------- раздел 4–5: основной опыт ----------

def _main_specs(results: Results, criterion: str) -> dict[tuple[str, float], RunSpec]:
    return {(s.condition, s.dose): s for s in results.find(part="main", criterion=criterion)}


def _main_lines(results: Results) -> list[str]:
    setup, config = results.setup, results.setup.config
    specs = _main_specs(results, config.criterion)
    if not specs:
        return []
    doses = sorted({d for _, d in specs})
    lines = ["## 4. Основной опыт", "",
             f"{config.trials} повторов по {_n(config.stimulus_ms)} мс со стимулом, затем {_n(config.after_ms)} мс без "
             f"него; критерий — «{CRITERIA_RU[config.criterion]}». Доза "
             f"{_dose(config.runaway_dose) if config.runaway_dose is not None else '—'} — доза разгона, найденная для "
             f"стимула «{config.condition(config.runaway_condition).name_ru}» (раздел 3), отмечена 🔥. «Повторов, где "
             "активность не гаснет» — правило разгона D-056 в каждом повторе отдельно. "
             "«Откликнулся» — правило Б2: больше 0 Гц за все повторы, вместе со стимулируемыми; распределения ниже — "
             "без стимулируемых. Эффективное число нейронов — exp(H), H — энтропия Шеннона распределения спайков "
             "не стимулируемых нейронов (D-056).", ""]
    for c in config.conditions:
        k = setup.key_index(c.readout_root_id)
        rows = []
        for d in doses:
            spec = specs.get((c.key, d))
            if spec is None:
                continue
            m = results.metrics[spec]
            mark = " 🔥" if config.runaway_dose is not None and d == config.runaway_dose else ""
            rows.append([_dose(d) + mark, _n(m.responders_b2), _n(m.active), number(m.n_eff, 0), hz(*m.key_rates[k]),
                         number(m.modality_entropy, 2), _pct(m.foreign_share) if m.foreign_share is not None else "—",
                         _n(m.visual_system), f"{m.sustained_trials} из {spec.trials}"])
        lines += [f"### {c.name_ru} — «исполнитель» {setup.keys[k].label}", ""]
        lines += _table(["доза d", "откликнулось (Б2)", "активных (≥ 1 Гц)", "эффективное число нейронов",
                         f"{setup.keys[k].label}", "энтропия модальностей, бит", "чужие модальности",
                         "в зрительной системе", "повторов, где активность не гаснет"], rows)
        # Все ключевые нейроны — для справки.
        rows = []
        for d in doses:
            spec = specs.get((c.key, d))
            if spec is not None:
                rows.append([_dose(d)] + [hz(*r) for r in results.metrics[spec].key_rates])
        lines += ["Все ключевые нейроны:", ""] + _table(["доза d"] + [key.label for key in setup.keys], rows)
        spec = specs.get((c.key, config.runaway_dose)) if config.runaway_dose is not None else None
        if spec is not None:
            m = results.metrics[spec]
            lines += [f"При дозе разгона {_dose(spec.dose)} активность после стимула не гаснет в {m.sustained_trials} из "
                      f"{spec.trials} повторов. Кто стреляет через 0–500 мс после стимула (класс клеток, нейронов): "
                      + _classes(m.after_classes) + ".", ""]

    lines += ["## 5. Кто откликается: суперклассы и модальности", "",
              "Откликнувшиеся (> 0 Гц, без стимулируемых). Модальность — ведущий орган чувств по входу нейрона "
              "(модуль `flybrain/modality.py`, [ЧЕСТНОСТЬ]: наш расчёт по схеме связей). «Средние доли входа» — доли "
              "модальностей, взвешенные по числу спайков.", ""]
    for c in config.conditions:
        rows_sc, rows_mod = [], []
        classes = sorted({sc for d in doses if (c.key, d) in specs
                          for sc in results.metrics[specs[(c.key, d)]].super_classes},
                         key=lambda sc: list(SUPER_CLASS_RU).index(sc) if sc in SUPER_CLASS_RU else 99)
        for d in doses:
            spec = specs.get((c.key, d))
            if spec is None:
                continue
            m = results.metrics[spec]
            rows_sc.append([_dose(d)] + [_n(m.super_classes.get(sc, 0)) for sc in classes])
            rows_mod.append([_dose(d)] + [_n(m.modalities[mod]) for mod in (*MODALITIES, "none")]
                            + [" / ".join(_pct(v) for v in m.spike_shares)])
        lines += [f"### {c.name_ru}", ""]
        lines += _table(["доза d"] + [SUPER_CLASS_RU.get(sc, sc) for sc in classes], rows_sc)
        lines += _table(["доза d"] + [MODALITIES_RU[mod] for mod in (*MODALITIES, "none")]
                        + ["средние доли входа: " + " / ".join(MODALITIES_RU[mod] for mod in MODALITIES)], rows_mod)
    return lines


def similarity_by_dose(results: Results) -> dict[float, list[tuple[str, str, float, float]]]:
    """Сходство узоров активности между стимулами при каждой дозе: (стимул, стимул, корреляция, совпадение).

    Корреляция Пирсона log(1 + частота) по нейронам, откликнувшимся (> 0 Гц) хотя бы на один из двух стимулов;
    совпадение — индекс Жаккара наборов активных (≥ 1 Гц) нейронов. Стимулируемые обоих стимулов не считаются.
    """
    setup, config = results.setup, results.setup.config
    specs = _main_specs(results, config.criterion)
    out = {}
    for d in sorted({d for _, d in specs}):
        rates, stim = {}, set()
        for c in config.conditions:
            spec = specs.get((c.key, d))
            if spec is None:
                continue
            rates[c.key] = results.runs[spec].totals / (spec.trials * spec.stimulus_ms / 1000)
            stim |= set(setup.stimulated(c.group).tolist())
        keep = np.ones(setup.n, dtype=bool)
        keep[list(stim)] = False
        keys = list(rates)
        pairs = []
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                ra, rb = rates[a][keep], rates[b][keep]
                union = (ra > 0) | (rb > 0)
                corr = float(np.corrcoef(np.log1p(ra[union]), np.log1p(rb[union]))[0, 1]) if union.sum() > 2 else 0.0
                act_a, act_b = ra >= ACTIVE_HZ, rb >= ACTIVE_HZ
                either = (act_a | act_b).sum()
                jaccard = float((act_a & act_b).sum() / either) if either else 1.0
                pairs.append((a, b, corr, jaccard))
        out[d] = pairs
    return out


def _similarity_lines(results: Results) -> list[str]:
    config = results.setup.config
    table = similarity_by_dose(results)
    if not table:
        return []
    names = {c.key: c.name_ru.split(" (")[0] for c in config.conditions}
    lines = ["## 5а. Насколько похожи ответы на разные стимулы", "",
             "**[ЧЕСТНОСТЬ] Мера добавлена после того, как увидели результат** (при больших дозах все стимулы давали "
             "похожую картину); в D-056 её не было, это описание без порога. Для каждой пары стимулов: корреляция "
             "log(1 + частота) по нейронам, откликнувшимся хотя бы на один из двух, и совпадение наборов активных "
             "(≥ 1 Гц) нейронов (общие / активные хотя бы в одном). Стимулируемые нейроны всех стимулов не считаются. "
             "1 — ответы одинаковы.", ""]
    pairs = [(a, b) for a, b, _, _ in next(iter(table.values()))]
    rows = []
    for d, values in table.items():
        by_pair = {(a, b): (corr, jac) for a, b, corr, jac in values}
        rows.append([_dose(d)] + [f"{number(by_pair[p][0], 2)} / {_pct(by_pair[p][1])}" if p in by_pair else "—"
                                  for p in pairs]
                    + [number(float(np.mean([v[2] for v in values])), 2)])
    lines += _table(["доза d"] + [f"{names[a]} — {names[b]}: корреляция / совпадение" for a, b in pairs]
                    + ["средняя корреляция"], rows)
    return lines


def _threshold_points(results: Results, condition, dose: float, key_index: int) -> list[tuple[float, float, float]]:
    points = []
    for spec in results.find(part="thresholds", condition=condition.key, dose=dose):
        mean, se = results.metrics[spec].key_rates[key_index]
        points.append((spec.hz, mean, se))
    return sorted(points)


def _threshold_lines(results: Results) -> list[str]:
    setup, config = results.setup, results.setup.config
    if not results.find(part="thresholds"):
        return []
    lines = ["## 6. Порог «исполнителей» по частоте стимула", "",
             f"{config.threshold_trials} повторов на точку. Порог — наименьшая частота стимула, при которой "
             "«исполнитель» активен (≥ 1 Гц, D-018), линейно между соседними точками сетки.", ""]
    rows = []
    for c in config.conditions:
        k = setup.key_index(c.readout_root_id)
        for d in config.doses:
            points = _threshold_points(results, c, d, k)
            kind, value = readout_threshold([(f, r) for f, r, _ in points])
            prefix = {"между": "", "не выше": "≤ ", "выше": "> "}.get(kind)
            text = "—" if prefix is None else f"{prefix}{number(value, 0)} Гц"
            curve = "; ".join(f"{number(f, 0)} → {number(r, 1)}" for f, r, _ in points)
            rows.append([c.name_ru.split(" (")[0], setup.keys[k].label, _dose(d), text, curve])
    lines += _table(["стимул", "исполнитель", "доза d", "порог по частоте", "частота стимула → частота исполнителя, Гц"],
                    rows)
    return lines


def _spont_lines(results: Results) -> list[str]:
    config = results.setup.config
    specs = results.find(part="spont")
    if not specs:
        return []
    lines = ["## 7. Спонтанная активность (без стимула, с шумом мембраны)", "",
             f"{config.spont_trials} повторов; шум σ — стандартное отклонение потенциала в покое (D-055); первые "
             f"{_n(config.warmup_ms)} мс не считаются, окно — {_n(config.stimulus_ms)} мс. Без шума (σ = 0) мозг Shiu "
             "в покое молчит при любой дозе ниже порога самовозбуждения (раздел 2).", ""]
    rows = []
    for spec in sorted(specs, key=lambda s: (s.sigma, s.dose)):
        m = results.metrics[spec]
        rows.append([number(spec.sigma, 0), _dose(spec.dose), number(m.mean_rate, 3), _n(m.responders), _n(m.active),
                     number(m.n_eff, 0), number(m.modality_entropy, 2), _n(m.visual_system)])
    lines += _table(["σ, мВ", "доза d", "средняя частота, Гц", "выстрелили хоть раз", "активных (≥ 1 Гц)",
                     "эффективное число нейронов", "энтропия модальностей, бит", "в зрительной системе"], rows)
    return lines


def _broad_lines(results: Results) -> list[str]:
    setup, config = results.setup, results.setup.config
    strict, broad = _main_specs(results, config.criterion), _main_specs(results, config.sensitivity_criterion)
    if not broad:
        return []
    lines = ["## 8. Проверка устойчивости: широкий критерий серотониновых нейронов", "",
             f"Те же стимулы и дозы, но серотониновые нейроны — «{CRITERIA_RU[config.sensitivity_criterion]}» "
             f"({_n(setup.sets.masks[config.sensitivity_criterion].sum())} нейронов вместо "
             f"{_n(setup.sets.masks[config.criterion].sum())}). При d = 0 расчёты одинаковы.", ""]
    rows = []
    for c in config.conditions:
        k = setup.key_index(c.readout_root_id)
        for d in config.doses:
            a, b = strict.get((c.key, d)), broad.get((c.key, d)) if d > 0 else strict.get((c.key, d))
            if a is None or b is None:
                continue
            ma, mb = results.metrics[a], results.metrics[b]
            rows.append([c.name_ru.split(" (")[0], _dose(d), f"{_n(ma.responders_b2)} / {_n(mb.responders_b2)}",
                         f"{number(ma.n_eff, 0)} / {number(mb.n_eff, 0)}",
                         f"{hz(*ma.key_rates[k])} / {hz(*mb.key_rates[k])}",
                         f"{_n(ma.visual_system)} / {_n(mb.visual_system)}",
                         f"{ma.sustained_trials} / {mb.sustained_trials} из {a.trials}"])
    lines += _table(["стимул", "доза d", "откликнулось: строгий / широкий", "эффективное число: строгий / широкий",
                     "исполнитель: строгий / широкий", "в зрительной системе: строгий / широкий",
                     "повторов, где активность не гаснет: строгий / широкий"], rows)
    return lines


def _technical_lines(results: Results) -> list[str]:
    seconds = [r.seconds for r in results.runs.values()]
    return ["## 9. Как получено", "",
            f"- Движок — тот же, что прошёл проверку C6 (версия кода `{engine_version()}`); при d = 0 и σ = 0 расчёт "
            "совпадает с прежним движком спайк в спайк (тест `tests/test_engine.py`).",
            "- Зерно каждого расчёта — `zlib.crc32('psy:<ключ расчёта>')`; ключ записан в имени файла в `data/psy/`.",
            f"- Расчётов: {len(results.runs)}; один расчёт — от {number(min(seconds), 0)} до {number(max(seconds), 0)} с "
            "на видеокарте (время, когда расчёт был сделан).",
            "- Модальности входа — `flybrain/modality.py`, серотониновые нейроны и мишени — `flybrain/serotonin.py`, "
            "настройки — `config/psy.yaml`.", ""]


def render_markdown(results: Results) -> str:
    lines = ["# P1 «психоделик»: таблицы", "",
             "*Файл собирает команда `uv run flybrain psy` — вручную его не правят. Цель, оговорки и выводы — в "
             "[EXPERIMENT_PSY.md](../EXPERIMENT_PSY.md).*", "",
             "**[ЧЕСТНОСТЬ]** Вычислительная гипотеза: нейроны с серотониновым входом становятся возбудимее, как от "
             "агониста 5-HT2A. Это не модель того, что «чувствует» муха. Выбор серотониновых нейронов, форма эффекта, "
             "параметры и метрики — наши (D-053…D-056), а не данные коннектома.", ""]
    for part in (_survey_lines, _mechanism_lines, _runaway_lines, _main_lines, _similarity_lines, _threshold_lines,
                 _spont_lines, _broad_lines, _technical_lines):
        lines += part(results)
    figures = [p for p in FIGURES if (DOCS_PSY / p).exists()]
    if figures:
        lines += ["## Картинки", ""] + [f"![{FIGURES[p]}]({p})\n\n*{FIGURES[p]}*\n" for p in figures]
    return "\n".join(lines).rstrip() + "\n"


# ---------- картинки ----------

FIGURES = {
    "runaway.png": "Поиск дозы разгона: сколько нейронов активно и гаснет ли активность после стимула",
    "responders.png": "Откликнувшиеся нейроны (правило Б2) в зависимости от дозы",
    "readouts.png": "Частота «исполнителей» в зависимости от дозы",
    "ignition.png": "Доля повторов с незатухающей активностью и сходство ответов на разные стимулы",
    "thresholds.png": "«Исполнитель» против частоты стимула при разных дозах",
    "diversity.png": "Эффективное число нейронов (разнообразие паттерна) в зависимости от дозы",
    "modalities.png": "Откликнувшиеся по ведущей модальности входа",
    "timecourse.png": "Суммарная активность во времени: стимул 0–1000 мс, затем без стимула",
    "spontaneous.png": "Спонтанная активность с шумом мембраны в зависимости от дозы",
}


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
    return f"{value:g}".replace(".", ",")


def _figure(plt, n_panels: int, width: float = 3.4, height: float = 3.0):
    fig, axes = plt.subplots(1, n_panels, figsize=(width * n_panels, height), facecolor=SURFACE, squeeze=False)
    return fig, axes[0]


def _finish(fig, plt, path: Path, title: str, note: str | None = None) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    for ax in fig.axes:
        if not getattr(ax, "categorical_x", False):
            ax.xaxis.set_major_formatter(FuncFormatter(_comma))
        if getattr(ax, "categorical_y", False):
            pass
        elif ax.get_yscale() == "linear":
            ax.yaxis.set_major_formatter(FuncFormatter(_comma))
        else:  # логарифмическая шкала: только 1, 10, 100… обычными числами
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: _n(v) if v >= 1 else _comma(v)))
            ax.yaxis.set_minor_formatter(NullFormatter())
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    if note:
        fig.text(0.01, 0.01, note, color=INK_2, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 0.93))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def _mark_runaway(ax, dose) -> None:
    if dose is not None:
        ax.axvline(dose, color=MUTED, linewidth=1, linestyle="--", zorder=1)


def figure_runaway(results: Results, path: Path) -> bool:
    import matplotlib.pyplot as plt

    config = results.setup.config
    crits = [c for c in (config.criterion, config.sensitivity_criterion) if results.find(part="runaway", criterion=c)]
    if not crits:
        return False
    fig, (ax1, ax2) = _figure(plt, 2, 4.4, 3.4)
    for color, crit in zip(CATEGORICAL, crits):
        specs = sorted(results.find(part="runaway", criterion=crit), key=lambda s: s.dose)
        doses = [s.dose for s in specs]
        label = CRITERIA_RU[crit].split(":")[0]
        ax1.plot(doses, [max(results.metrics[s].active, 0.8) for s in specs], color=color, linewidth=2, marker="o",
                 markersize=6, markeredgecolor=SURFACE, label=label)
        ax2.plot(doses, [100 * results.metrics[s].sustained_trials / s.trials for s in specs], color=color,
                 linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, label=label)
    ax1.set_yscale("log")
    ax1.set_title("активных нейронов (≥ 1 Гц) во время стимула", fontsize=9, color=INK, loc="left")
    ax2.set_ylim(-4, 104)
    ax2.set_title("повторов, где активность после стимула не гаснет, %", fontsize=9, color=INK, loc="left")
    for ax in (ax1, ax2):
        _style(ax)
        ax.set_xlabel("доза d", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    _finish(fig, plt, path, f"Поиск дозы разгона: {config.condition(config.runaway_condition).name_ru}, "
            f"{config.runaway_trials} повторов",
            "«не гаснет»: через 300–500 мс после стимула спайков не меньше 10 % от конца стимула (D-056)")
    return True


def _per_condition(results: Results, path: Path, title: str, value, ylabel: str, note: str | None = None,
                   log: bool = False, error=None) -> bool:
    import matplotlib.pyplot as plt

    config = results.setup.config
    specs = _main_specs(results, config.criterion)
    if not specs:
        return False
    fig, axes = _figure(plt, len(config.conditions), 3.3, 3.1)
    for ax, color, c in zip(axes, CATEGORICAL, config.conditions):
        doses = sorted(d for key, d in specs if key == c.key)
        ys = [value(c, results.metrics[specs[(c.key, d)]]) for d in doses]
        ax.plot(doses, ys, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, zorder=3)
        if error is not None:
            errs = [error(c, results.metrics[specs[(c.key, d)]]) for d in doses]
            ax.errorbar(doses, ys, yerr=[2 * e for e in errs], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2.5)
        _mark_runaway(ax, config.runaway_dose)
        if log:
            ax.set_yscale("log")
        _style(ax)
        ax.set_title(c.name_ru, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("доза d", color=INK_2, fontsize=9)
    axes[0].set_ylabel(ylabel, color=INK_2, fontsize=9)
    _finish(fig, plt, path, title, note)
    return True


def _stimulus_word(name_ru: str) -> str:
    """«тень слева 100 Гц ⚗» → «тень слева ⚗»: название стимула без частоты (на графике частота — по оси)."""
    words = name_ru.split(" (")[0].split()
    return " ".join(w for w in words if not any(ch.isdigit() for ch in w) and w != "Гц")


def figure_ignition(results: Results, path: Path) -> bool:
    """Доля повторов, где активность после стимула не гаснет, и средняя корреляция ответов разных стимулов."""
    import matplotlib.pyplot as plt

    config = results.setup.config
    specs = _main_specs(results, config.criterion)
    if not specs:
        return False
    fig, (ax1, ax2) = _figure(plt, 2, 4.6, 3.4)
    for color, c in zip(CATEGORICAL, config.conditions):
        doses = sorted(d for key, d in specs if key == c.key)
        share = [100 * results.metrics[specs[(c.key, d)]].sustained_trials / specs[(c.key, d)].trials for d in doses]
        ax1.plot(doses, share, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                 label=c.name_ru, zorder=3)
    ax1.set_ylim(-4, 104)
    ax1.set_title("повторов, где активность после стимула не гаснет, %", fontsize=9, color=INK, loc="left")
    ax1.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left")
    table = similarity_by_dose(results)
    doses = list(table)
    ax2.plot(doses, [100 * float(np.mean([v[3] for v in table[d]])) for d in doses], color=INK_2, linewidth=2,
             marker="o", markersize=6, markeredgecolor=SURFACE, zorder=3)
    ax2.set_ylim(-4, 104)
    ax2.set_title("общие активные нейроны у пар стимулов, % (среднее)", fontsize=9, color=INK, loc="left")
    for ax in (ax1, ax2):
        _mark_runaway(ax, config.runaway_dose)
        _style(ax)
        ax.set_xlabel("доза d", color=INK_2, fontsize=9)
    _finish(fig, plt, path, "Разгон: самоподдерживающаяся активность и одинаковые ответы на разные стимулы",
            "пунктир — доза разгона из поиска (сахар 40 Гц); справа — мера, добавленная после того, как увидели "
            "результат (раздел 5а)")
    return True


def figure_thresholds(results: Results, path: Path) -> bool:
    import matplotlib.pyplot as plt

    setup, config = results.setup, results.setup.config
    if not results.find(part="thresholds"):
        return False
    fig, axes = _figure(plt, len(config.conditions), 3.3, 3.2)
    colors = SEQUENTIAL[1::max(1, len(SEQUENTIAL) // max(1, len(config.doses)))][:len(config.doses)]
    if len(colors) < len(config.doses):
        colors = SEQUENTIAL[-len(config.doses):]
    for ax, c in zip(axes, config.conditions):
        k = setup.key_index(c.readout_root_id)
        for color, d in zip(colors, config.doses):
            points = _threshold_points(results, c, d, k)
            ax.plot([f for f, _, _ in points], [r for _, r, _ in points], color=color, linewidth=2, marker="o",
                    markersize=5, markeredgecolor=SURFACE, label=f"d = {_dose(d)}")
        ax.axhline(ACTIVE_HZ, color=MUTED, linewidth=1)
        _style(ax)
        ax.set_title(f"{_stimulus_word(c.name_ru)} → {setup.keys[k].label}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("частота стимула, Гц", color=INK_2, fontsize=9)
    axes[0].set_ylabel("частота исполнителя, Гц", color=INK_2, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    _finish(fig, plt, path, "«Исполнители» против частоты стимула при разных дозах",
            "серая линия — порог «активного» 1 Гц (D-018); темнее — больше доза")
    return True


def figure_modalities(results: Results, path: Path) -> bool:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    config = results.setup.config
    specs = _main_specs(results, config.criterion)
    if not specs:
        return False
    fig, axes = _figure(plt, len(config.conditions), 3.3, 3.4)
    for ax, c in zip(axes, config.conditions):
        doses = sorted(d for key, d in specs if key == c.key)
        x = np.arange(len(doses))
        bottom = np.zeros(len(doses))
        for mod in (*MODALITIES, "none"):
            counts = np.array([results.metrics[specs[(c.key, d)]].modalities[mod] for d in doses], dtype=float)
            totals = np.array([max(results.metrics[specs[(c.key, d)]].responders, 1) for d in doses], dtype=float)
            share = 100 * counts / totals
            ax.bar(x, share, bottom=bottom, width=0.7, color=MODALITY_COLORS.get(mod, NONE_COLOR),
                   edgecolor=SURFACE, linewidth=1, zorder=2)
            bottom += share
        ax.set_xticks(x, [_dose(d) for d in doses], fontsize=8, color=INK_2)
        ax.categorical_x = True
        _style(ax)
        ax.set_ylim(0, 100)
        ax.set_title(c.name_ru, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("доза d", color=INK_2, fontsize=9)
    axes[0].set_ylabel("% откликнувшихся", color=INK_2, fontsize=9)
    handles = [Patch(color=MODALITY_COLORS.get(m, NONE_COLOR), label=MODALITIES_RU[m]) for m in (*MODALITIES, "none")]
    fig.legend(handles=handles, loc="lower right", ncols=6, frameon=False, fontsize=8, labelcolor=INK_2,
               bbox_to_anchor=(0.995, 0.0))
    _finish(fig, plt, path, "Откликнувшиеся нейроны по ведущей модальности входа",
            "[ЧЕСТНОСТЬ] модальность у не сенсорных нейронов — наш расчёт по схеме связей")
    return True


def figure_timecourse(results: Results, path: Path) -> bool:
    import matplotlib.pyplot as plt

    config = results.setup.config
    specs = _main_specs(results, config.criterion)
    if not specs:
        return False
    fig, axes = _figure(plt, len(config.conditions), 3.3, 3.2)
    for ax, c in zip(axes, config.conditions):
        doses = sorted(d for key, d in specs if key == c.key)
        colors = SEQUENTIAL[-len(doses):] if len(doses) <= len(SEQUENTIAL) else SEQUENTIAL
        for color, d in zip(colors, doses):
            spec = specs[(c.key, d)]
            run = results.runs[spec]
            edges = run.edges_ms
            spikes = np.diff(run.cumulative.sum(1)) / spec.trials
            widths = np.diff(edges) / 1000
            rate = np.divide(spikes, widths, out=np.zeros_like(spikes, dtype=float), where=widths > 0)
            ax.plot(edges[1:], np.maximum(rate, 1), color=color, linewidth=1.5, label=f"d = {_dose(d)}")
        ax.axvline(config.stimulus_ms, color=MUTED, linewidth=1, linestyle="--")
        ax.set_yscale("log")
        _style(ax)
        ax.set_title(c.name_ru, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("время, мс", color=INK_2, fontsize=9)
    axes[0].set_ylabel("спайков/с во всём мозге", color=INK_2, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncols=len(labels), frameon=False, fontsize=8, labelcolor=INK_2,
               bbox_to_anchor=(0.995, 0.0))
    _finish(fig, plt, path, "Суммарная активность не стимулируемых нейронов во времени (среднее по повторам)",
            "пунктир — стимул выключен; темнее — больше доза; ниже 1 спайка/с показано как 1")
    return True


def figure_spontaneous(results: Results, path: Path) -> bool:
    import matplotlib.pyplot as plt

    config = results.setup.config
    specs = results.find(part="spont")
    if not specs:
        return False
    fig, (ax1, ax2) = _figure(plt, 2, 4.2, 3.2)
    sigmas = sorted({s.sigma for s in specs})
    colors = SEQUENTIAL[1::2][:len(sigmas)] if len(sigmas) <= 3 else SEQUENTIAL[-len(sigmas):]
    for color, sigma in zip(colors, sigmas):
        row = sorted((s for s in specs if s.sigma == sigma), key=lambda s: s.dose)
        doses = [s.dose for s in row]
        ax1.plot(doses, [max(results.metrics[s].mean_rate, 1e-4) for s in row], color=color, linewidth=2, marker="o",
                 markersize=6, markeredgecolor=SURFACE, label=f"σ = {number(sigma, 0)} мВ")
        ax2.plot(doses, [max(results.metrics[s].active, 0.8) for s in row], color=color, linewidth=2, marker="o",
                 markersize=6, markeredgecolor=SURFACE, label=f"σ = {number(sigma, 0)} мВ")
    ax1.set_title("средняя частота по мозгу, Гц", fontsize=9, color=INK, loc="left")
    ax2.set_title("активных нейронов (≥ 1 Гц)", fontsize=9, color=INK, loc="left")
    for ax in (ax1, ax2):
        ax.set_yscale("log")
        _mark_runaway(ax, config.runaway_dose)
        _style(ax)
        ax.set_xlabel("доза d", color=INK_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    _finish(fig, plt, path, "Спонтанная активность без стимула (шум мембраны σ)",
            "пунктир — доза разгона из поиска со стимулом; точки у нижнего края — ноль (на логарифмической шкале "
            "его не нарисовать)")
    return True


def write_figures(results: Results) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    DOCS_PSY.mkdir(parents=True, exist_ok=True)
    config = results.setup.config
    k_of = lambda c: results.setup.key_index(c.readout_root_id)  # noqa: E731
    makers = {
        "runaway.png": lambda p: figure_runaway(results, p),
        "responders.png": lambda p: _per_condition(
            results, p, "Откликнувшиеся нейроны (> 0 Гц, вместе со стимулируемыми, правило Б2)",
            lambda c, m: m.responders_b2, "нейронов", "пунктир — доза разгона; шкала логарифмическая", log=True),
        "readouts.png": lambda p: _per_condition(
            results, p, "Частота «исполнителя»: MN9 (сахар, вода), aBN1 (JO-F), гигантское волокно (тень)",
            lambda c, m: m.key_rates[k_of(c)][0], "Гц", "чёрточки — ± две погрешности; пунктир — доза разгона",
            error=lambda c, m: m.key_rates[k_of(c)][1]),
        "ignition.png": lambda p: figure_ignition(results, p),
        "thresholds.png": lambda p: figure_thresholds(results, p),
        "diversity.png": lambda p: _per_condition(
            results, p, "Разнообразие паттерна: эффективное число нейронов exp(H)", lambda c, m: m.n_eff,
            "нейронов", "H — энтропия Шеннона распределения спайков не стимулируемых нейронов; пунктир — доза разгона",
            log=True),
        "modalities.png": lambda p: figure_modalities(results, p),
        "timecourse.png": lambda p: figure_timecourse(results, p),
        "spontaneous.png": lambda p: figure_spontaneous(results, p),
    }
    written = []
    for name, make in makers.items():
        path = DOCS_PSY / name
        if make(path):
            written.append(path)
        elif path.exists():
            path.unlink()
    del config
    return written


def summary(results: Results) -> dict:
    config = results.setup.config
    out = {"criterion": config.criterion, "serotonin_neurons": {c: int(results.setup.sets.masks[c].sum())
                                                                for c in CRITERIA},
           "runaway_dose": {c: runaway_dose(results, c) for c in (config.criterion, config.sensitivity_criterion)},
           "main": {}}
    for (cond, dose), spec in sorted(_main_specs(results, config.criterion).items()):
        m = results.metrics[spec]
        k = results.setup.key_index(config.condition(cond).readout_root_id)
        out["main"].setdefault(cond, {})[_dose(dose)] = {
            "responders_b2": m.responders_b2, "active": m.active, "n_eff": round(m.n_eff, 1),
            "readout_hz": round(m.key_rates[k][0], 2), "visual_system": m.visual_system,
            "runaway": m.runaway}
    return out


def write_report(results: Results) -> list[Path]:
    figures = write_figures(results)
    RESULTS_FILE.write_text(render_markdown(results), encoding="utf-8", newline="\n")
    SUMMARY_FILE.write_text(json.dumps(summary(results), ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                            newline="\n")
    return [RESULTS_FILE, SUMMARY_FILE, *figures]
