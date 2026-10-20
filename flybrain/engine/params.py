"""Параметры модели Shiu — ровно как `default_params` в `model.py` (D-005). Менять нельзя (D-018)."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    # Единицы: милливольты и миллисекунды (Brian2 считает в вольтах и секундах — числа те же).
    v_0: float = -52.0  # потенциал покоя, мВ
    v_rst: float = -52.0  # сброс после спайка, мВ
    v_th: float = -45.0  # порог, мВ (спайк, если v > v_th)
    t_mbr: float = 20.0  # постоянная мембраны, мс
    tau: float = 5.0  # постоянная синапса, мс
    t_rfc: float = 2.2  # рефрактерный период («отдых»), мс; у стимулируемых нейронов 0
    t_dly: float = 1.8  # задержка синапса, мс
    w_syn: float = 0.275  # вес одного синапса, мВ
    f_poi: float = 250.0  # во сколько раз импульс стимула тяжелее синапса
    dt: float = 0.1  # шаг расчёта, мс (по умолчанию в Brian2)

    @property
    def delay_steps(self) -> int:
        """Задержка в шагах: Brian2 округляет её до целого числа шагов."""
        return int(round(self.t_dly / self.dt))

    @property
    def refractory_steps(self) -> int:
        """Отдых в шагах — как `timestep(rfc, dt)` в Brian2: int(t/dt + 0,001)."""
        return int(self.t_rfc / self.dt + 1e-3)

    @property
    def poisson_weight(self) -> float:
        """Один импульс стимула, мВ: 250 × 0,275 = 68,75 — больше разницы порога и покоя."""
        return self.w_syn * self.f_poi

    def decay_factors(self) -> tuple[float, float, float]:
        """Точное решение уравнений за один шаг (метод 'linear' в Brian2), в двойной точности.

        g ← g·E;  v ← v_0 + (v − v_0)·A + g·B, где g — значение до шага.
        """
        a = math.exp(-self.dt / self.t_mbr)
        e = math.exp(-self.dt / self.tau)
        b = self.tau * (a - e) / (self.t_mbr - self.tau)
        return a, b, e


SHIU = ModelParams()
