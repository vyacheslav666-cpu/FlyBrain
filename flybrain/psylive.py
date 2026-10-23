"""Панель «Психоделик (эксперимент)» в живом режиме (P2, D-064…D-066).

[ЧЕСТНОСТЬ] Вычислительная гипотеза P1/P1b, а не ощущения мухи: доза снижает порог у мишеней серотониновых
нейронов (D-053…D-055), в режиме «клетки Кеньона тормозятся» у KC порог повышается (D-060); шум мембраны σ
(D-055). Здесь — только перенос этих правил в живой режим и индикатор «сеть зажглась» (D-064).

- `PsyLive` — данные: серотониновые синапсы каждого нейрона (строгий критерий, D-054), клетки Кеньона,
  возбуждающие локальные нейроны усиковой доли (петля «зажигания», D-058), ΔV и s_half из `config/psy.yaml`.
- `IgnitionMeter` — индикатор: «горит», если за последние ≥ 100 мс модели не стимулируемые нейроны дали больше
  100 тыс. спайков в секунду (тот же порог, что в гистерезисе P1b, D-061). Отдельно — сколько возбуждающих
  локальных нейронов усиковой доли стреляли в этом окне (для объяснения, на решение не влияет).
"""

import json
from collections import deque
from dataclasses import dataclass

import numpy as np
import yaml

from flybrain import processed
from flybrain.paths import CONFIG_DIR, PROCESSED_DIR
from flybrain.serotonin import SerotoninError, read_annotations, saturation, serotonin_sets, serotonin_synapses, \
    threshold_shift

PSY_CONFIG = CONFIG_DIR / "psy.yaml"
DOSE_MAX = 6.0  # ползунок 0…6: выше разгона P1 (5) и P1b (6), дальше — уже не интересно (D-065)
DOSE_STEP = 0.25
SIGMA_MAX = 3.0
SIGMA_STEP = 0.25
SIGMA_IGNITES = 1.75  # с этого шума сеть зажигается и без дозы (P1b, D-058)
MODES = ("p1", "kc_inhibit")
IGNITED_HZ = 100_000.0  # «горит» — больше 100 тыс. спайков в секунду во всём мозге (P1b, D-061)
WINDOW_MS = 100.0  # окно индикатора — последние 100 мс модели, как в P1b
GAP_MV = 7.0  # зазор «покой — порог» в модели Shiu (D-005)
KC_CLASS = "Kenyon_Cell"


@dataclass
class PsyLive:
    available: bool
    reason: str = ""
    synapses: np.ndarray | None = None  # s_i — серотониновые синапсы нейрона i (строгий критерий)
    serotonin: np.ndarray | None = None  # номера серотониновых нейронов
    kc: np.ndarray | None = None  # булева маска клеток Кеньона
    alln_exc: np.ndarray | None = None  # номера возбуждающих локальных нейронов усиковой доли
    delta_v_mv: float = 1.0
    s_half: float = 5.0

    @classmethod
    def load(cls, processed_dir=PROCESSED_DIR, config_path=PSY_CONFIG) -> "PsyLive":
        """Всё, что нужно панели. Нет таблицы аннотаций — панель недоступна, но сервер работает."""
        try:
            with open(config_path, encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            neurons = processed.load_neurons(processed_dir)
            csr = processed.load_connectivity(processed_dir)
            with open(processed_dir / processed.BUILD_INFO_FILE, encoding="utf-8") as handle:
                expected = json.load(handle)["inputs_sha256"].get("Supplemental_file1_neuron_annotations.tsv")
            sets = serotonin_sets(neurons, read_annotations(expected_sha256=expected))
        except (OSError, KeyError, SerotoninError) as exc:
            return cls(False, f"панель недоступна: {str(exc).splitlines()[0]}")
        strict = sets.masks[config.get("criterion", "strict")]
        alln = (neurons["cell_class"] == "ALLN").to_numpy() & (neurons["sign"] > 0).to_numpy()
        return cls(True, "", serotonin_synapses(csr["indptr"], csr["indices"], csr["weight"], strict),
                   np.flatnonzero(strict), (neurons["cell_class"] == KC_CLASS).to_numpy(), np.flatnonzero(alln),
                   float(config["delta_v_mv"]), float(config["s_half"]))

    def shift(self, dose: float, mode: str) -> np.ndarray | None:
        """Снижение порога каждого нейрона, мВ; None — доза 0 (движок идёт прежним путём)."""
        if dose <= 0:
            return None
        return threshold_shift(self.synapses, dose, self.delta_v_mv, self.s_half,
                               self.kc if mode == "kc_inhibit" else None)

    def mv_per_dose(self) -> dict[str, float]:
        """На сколько мВ снижает порог единица дозы у типичной (медианной) и у самой «нагруженной» мишени."""
        targets = self.synapses[self.synapses > 0]
        median, peak = float(np.median(targets)), float(targets.max())
        return {"median_s": median, "max_s": peak,
                "median_mv": self.delta_v_mv * float(saturation(np.array([median]), self.s_half)[0]),
                "max_mv": self.delta_v_mv * float(saturation(np.array([peak]), self.s_half)[0])}

    def info(self) -> dict:
        """Ответ /api/psy: настройки панели и данные слоя «серотониновые нейроны и их мишени»."""
        if not self.available:
            return {"available": False, "reason": self.reason}
        targets = np.flatnonzero(self.synapses > 0)
        return {
            "available": True, "dose_max": DOSE_MAX, "dose_step": DOSE_STEP, "sigma_max": SIGMA_MAX,
            "sigma_step": SIGMA_STEP, "sigma_ignites": SIGMA_IGNITES, "gap_mv": GAP_MV, "ignited_hz": IGNITED_HZ,
            "window_ms": WINDOW_MS, "s_half": self.s_half, **self.mv_per_dose(),
            "modes": [{"key": "p1", "name_ru": "все мишени возбуждаются (P1)"},
                      {"key": "kc_inhibit", "name_ru": "клетки Кеньона тормозятся (P1b)"}],
            "al_total": int(len(self.alln_exc)),
            "serotonin": self.serotonin.tolist(), "targets": targets.tolist(),
            "target_synapses": self.synapses[targets].tolist(),
        }


class IgnitionMeter:
    """Индикатор «сеть зажглась» по кадрам живого режима (D-064)."""

    def __init__(self, al_total: int, window_ms: float = WINDOW_MS, threshold_hz: float = IGNITED_HZ):
        self.al_total = al_total
        self.window_ms = window_ms
        self.threshold_hz = threshold_hz
        self._frames: deque = deque()  # (начало, конец, спайки не стимулируемых, маска стрелявших ALLN)

    def reset(self) -> None:
        self._frames.clear()

    def add(self, start_ms: float, end_ms: float, spikes: int, al_fired: np.ndarray) -> None:
        if self._frames and abs(self._frames[-1][1] - start_ms) > 1e-6:
            self._frames.clear()  # разрыв во времени (сброс): окно — заново
        self._frames.append((start_ms, end_ms, int(spikes), np.asarray(al_fired, dtype=bool)))
        while len(self._frames) > 1 and self._frames[-1][1] - self._frames[1][0] >= self.window_ms:
            self._frames.popleft()

    @property
    def span_ms(self) -> float:
        return self._frames[-1][1] - self._frames[0][0] if self._frames else 0.0

    @property
    def rate_hz(self) -> float:
        """Спайков в секунду во всём мозге (без стимулируемых) за окно."""
        span = self.span_ms
        return sum(f[2] for f in self._frames) * 1000.0 / span if span > 0 else 0.0

    @property
    def al_active(self) -> int:
        if not self._frames or self.al_total == 0:
            return 0
        return int(np.any(np.stack([f[3] for f in self._frames]), axis=0).sum())

    @property
    def ignited(self) -> bool:
        return self.span_ms >= self.window_ms - 1e-6 and self.rate_hz > self.threshold_hz

    def reading(self) -> dict:
        return {"type": "psy", "rate_hz": round(self.rate_hz, 1), "span_ms": round(self.span_ms, 2),
                "al_active": self.al_active, "al_total": self.al_total, "ignited": self.ignited}
