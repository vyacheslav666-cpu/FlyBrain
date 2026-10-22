"""Правые двойники для стимулов, у которых в статье список только для левой стороны мухи (D-047).

Двойники ищутся по таблице аннотаций FlyWire 783: тип клетки + сторона. Для каждого типа клеток левого
списка двойники однозначны, только если список берёт весь тип на левой стороне, а на правой стороне
нейронов этого типа ровно столько же. Тогда двойники — все правые нейроны типа: кто чей двойник внутри
типа, таблица не говорит, но стимул включает их все одинаково, и выбирать ничего не приходится.
Во всех остальных случаях двойников нет, и левые нейроны этого типа справа не представлены.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Почему у левых нейронов нет двойников (пусто — двойники нашлись).
REASONS_RU = {
    "part_of_type": "в списке статьи только часть левых нейронов этого типа — какие из правых им соответствуют, "
                    "по аннотациям сказать нельзя",
    "count_differs": "в списке весь тип, но справа нейронов этого типа другое число — у кого-то пары нет, "
                     "а у кого, неизвестно",
    "no_type": "у нейрона нет типа клетки в таблице аннотаций",
    "not_left": "нейрон по таблице аннотаций не на левой стороне",
}


@dataclass
class TypeMatch:
    """Один тип клеток левого списка и его двойники на правой стороне."""
    cell_type: str | None
    listed: list[int]  # внутренние номера нейронов левого списка этого типа
    left_total: int  # всего нейронов этого типа на левой стороне по таблице
    right_total: int  # на правой
    twins: list[int] = field(default_factory=list)  # внутренние номера правых двойников
    reason: str = ""  # ключ REASONS_RU; пусто — двойники нашлись

    def to_json(self) -> dict:
        return {"cell_type": self.cell_type, "listed": self.listed, "left_total": self.left_total,
                "right_total": self.right_total, "twins": self.twins, "reason": self.reason}


def match_twins(left_indices, neurons: pd.DataFrame) -> list[TypeMatch]:
    """Двойники по правилу D-047 для каждого типа клеток левого списка (порядок — по типу)."""
    cell_types = neurons["cell_type"].to_numpy(dtype=object)
    sides = neurons["side"].to_numpy(dtype=object)
    by_type: dict[str, list[int]] = {}
    matches = []
    for index in sorted(int(i) for i in left_indices):
        cell_type, side = cell_types[index], sides[index]
        if pd.isna(cell_type):
            matches.append(TypeMatch(None, [index], 0, 0, reason="no_type"))
        elif side != "left":
            matches.append(TypeMatch(str(cell_type), [index], 0, 0, reason="not_left"))
        else:
            by_type.setdefault(str(cell_type), []).append(index)
    for cell_type in sorted(by_type):
        listed = by_type[cell_type]
        of_type = cell_types == cell_type
        left = np.flatnonzero(of_type & (sides == "left"))
        right = np.flatnonzero(of_type & (sides == "right"))
        match = TypeMatch(cell_type, listed, len(left), len(right))
        if len(listed) < len(left):
            match.reason = "part_of_type"
        elif len(left) != len(right):
            match.reason = "count_differs"
        else:
            match.twins = right.tolist()
        matches.append(match)
    return matches


def twins_of(matches: list[TypeMatch]) -> np.ndarray:
    """Все найденные двойники, по возрастанию внутреннего номера."""
    return np.array(sorted(i for match in matches for i in match.twins), dtype=np.int64)

