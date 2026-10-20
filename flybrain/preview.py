"""Картинка data/processed/preview.png: облако нейронов в трёх проекциях (коммит C3).

Оси таблицы нейронов (D-011, docs/DATA.md раздел 5): x растёт к левой стороне мухи (X уже отражён),
y — сверху вниз, z — спереди назад. Проекции нарисованы так, как их видит человек:
- фронтальная — со спины мухи: левая сторона мухи слева, верх сверху;
- сбоку — с левой стороны мухи: перед слева;
- сверху — перед вверху, левая сторона мухи слева.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

BACKGROUND = "#0b0e14"
CLOUD_COLOR = "#8b93a7"
TEXT_COLOR = "#e6e9ef"
MUTED_TEXT = "#9aa3b5"
MARGIN_UM = 20.0
INCH_PER_UM = 0.0105


@dataclass
class Highlight:
    label: str  # подпись в легенде
    short: str  # подпись на фронтальной проекции
    indices: np.ndarray
    color: str
    size: float
    offset_um: tuple[float, float]  # куда отнести подпись на фронтальной проекции: (вправо, вверх)


def _limits(values: np.ndarray) -> tuple[float, float]:
    return float(np.nanmin(values)) - MARGIN_UM, float(np.nanmax(values)) + MARGIN_UM


def draw_preview(neurons: pd.DataFrame, highlights: list[Highlight], path: Path) -> None:
    placed = neurons[neurons["x_um"].notna()]
    x, y, z = (placed[c].to_numpy() for c in ("x_um", "y_um", "z_um"))
    x_lim, y_lim, z_lim = _limits(x), _limits(y), _limits(z)
    width_x, height_y, depth_z = (hi - lo for lo, hi in (x_lim, y_lim, z_lim))

    # Раскладка в дюймах: один масштаб на всех проекциях, чтобы форма не искажалась.
    pad_left, pad_right, pad_bottom, pad_top, gap = 0.55, 0.35, 0.55, 1.35, 0.75
    w_front, h_front = width_x * INCH_PER_UM, height_y * INCH_PER_UM
    w_side, h_top = depth_z * INCH_PER_UM, depth_z * INCH_PER_UM
    fig_w = pad_left + w_front + gap + w_side + pad_right
    fig_h = pad_bottom + h_top + gap + h_front + pad_top
    fig = Figure(figsize=(fig_w, fig_h), dpi=150, facecolor=BACKGROUND)

    def axes(left: float, bottom: float, width: float, height: float):
        ax = fig.add_axes((left / fig_w, bottom / fig_h, width / fig_w, height / fig_h))
        ax.set_facecolor(BACKGROUND)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2a3040")
        return ax

    row_top = pad_bottom + h_top + gap
    col_right = pad_left + w_front + gap
    front = axes(pad_left, row_top, w_front, h_front)
    side = axes(col_right, row_top, w_side, h_front)
    top = axes(pad_left, pad_bottom, w_front, h_top)

    # Пары столбцов для каждой проекции и пределы осей (перевёрнутые там, где так видит человек).
    views = [
        (front, x, y, (x_lim[1], x_lim[0]), (y_lim[1], y_lim[0]), "x_um", "y_um"),
        (side, z, y, z_lim, (y_lim[1], y_lim[0]), "z_um", "y_um"),
        (top, x, z, (x_lim[1], x_lim[0]), (z_lim[1], z_lim[0]), "x_um", "z_um"),
    ]
    for ax, horizontal, vertical, xlim, ylim, col_h, col_v in views:
        ax.scatter(horizontal, vertical, s=0.3, c=CLOUD_COLOR, alpha=0.3, linewidths=0, rasterized=True)
        for item in highlights:
            members = neurons.iloc[item.indices]
            members = members[members["x_um"].notna()]
            ax.scatter(members[col_h], members[col_v], s=item.size, c=item.color,
                       edgecolors="#000000", linewidths=0.4, zorder=3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    # Подписи групп на фронтальной проекции. Оси перевёрнуты, поэтому «вправо/вверх» — это минус в данных.
    for item in highlights:
        members = neurons.iloc[item.indices]
        members = members[members["x_um"].notna()]
        if members.empty:
            continue
        cx, cy = members["x_um"].mean(), members["y_um"].mean()
        label_xy = (cx - item.offset_um[0], cy - item.offset_um[1])
        targets = members[["x_um", "y_um"]].to_numpy() if len(members) <= 4 else np.array([[cx, cy]])
        for i, target in enumerate(targets):
            front.annotate(
                item.short if i == 0 else "", xy=tuple(target), xytext=label_xy,
                color=item.color, fontsize=10, fontweight="bold", ha="center", va="center", zorder=4,
                arrowprops={"arrowstyle": "-", "color": item.color, "lw": 0.8, "shrinkA": 6, "shrinkB": 3},
            )

    def text(x_in: float, y_in: float, s: str, **kwargs) -> None:
        kwargs.setdefault("color", MUTED_TEXT)
        kwargs.setdefault("fontsize", 9)
        fig.text(x_in / fig_w, y_in / fig_h, s, **kwargs)

    title_kw = {"color": TEXT_COLOR, "fontsize": 11, "va": "bottom"}
    text(pad_left, row_top + h_front + 0.08, "Фронтальная проекция — вид со спины мухи", **title_kw)
    text(col_right, row_top + h_front + 0.08, "Сбоку — с левой стороны мухи", **title_kw)
    text(pad_left, pad_bottom + h_top + 0.08, "Сверху", **title_kw)
    # Направления.
    text(pad_left, row_top - 0.2, "← левая сторона мухи", va="center")
    text(pad_left + w_front, row_top - 0.2, "правая сторона мухи →", va="center", ha="right")
    text(pad_left - 0.12, row_top + h_front, "верх", va="top", ha="right")
    text(pad_left - 0.12, row_top, "низ", va="bottom", ha="right")
    text(col_right, row_top - 0.2, "← перед", va="center")
    text(col_right + w_side, row_top - 0.2, "зад →", va="center", ha="right")
    text(pad_left, pad_bottom - 0.2, "← левая сторона мухи", va="center")
    text(pad_left + w_front, pad_bottom - 0.2, "правая сторона мухи →", va="center", ha="right")
    text(pad_left - 0.12, pad_bottom + h_top, "перед", va="top", ha="right")
    text(pad_left - 0.12, pad_bottom, "зад", va="bottom", ha="right")

    n_placed = len(placed)
    n_total = len(neurons)
    text(pad_left, fig_h - 0.42,
         f"FlyBrain · {n_placed:,} нейронов FlyWire 783 в настоящих координатах".replace(",", " "),
         color=TEXT_COLOR, fontsize=14, fontweight="bold", va="bottom")
    text(pad_left, fig_h - 0.68,
         "Точка — тело клетки, а если оно вне мозга — опорная точка на отростке (D-016). "
         "Стороны — со стороны мухи, по таблице аннотаций (D-011).", va="bottom")
    if n_total > n_placed:
        text(pad_left, fig_h - 0.88,
             f"У {n_total - n_placed} нейронов модели нет аннотации и координат — они не нарисованы.", va="bottom")

    # Легенда и масштаб — справа внизу.
    legend_y = pad_bottom + h_top - 0.1
    for item in highlights:
        text(col_right, legend_y, "●", color=item.color, fontsize=13, va="top")
        text(col_right + 0.25, legend_y - 0.02, item.label, color=TEXT_COLOR, va="top", linespacing=1.3)
        legend_y -= 0.25 + 0.17 * item.label.count("\n")
    text(col_right, legend_y, "●", color=CLOUD_COLOR, fontsize=13, va="top")
    text(col_right + 0.25, legend_y - 0.02, "остальные нейроны", color=TEXT_COLOR, va="top")
    bar_y = pad_bottom + 0.1
    scale = fig.add_axes((col_right / fig_w, bar_y / fig_h, 100 * INCH_PER_UM / fig_w, 0.02 / fig_h))
    scale.axis("off")
    scale.axhline(0.5, color=TEXT_COLOR, lw=2)
    text(col_right, bar_y + 0.08, "100 мкм", va="bottom")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=BACKGROUND)
