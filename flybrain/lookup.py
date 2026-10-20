"""Справки про отдельные нейроны для подсказки, связей и поиска на странице (коммит C8)."""

import numpy as np
import pandas as pd

from flybrain.webdata import NO_POSITION, class_codes

LINKS_PER_SIDE = 20  # сколько самых сильных входов и выходов показываем (PLAN_M1, C8)
MAX_SEARCH_POINTS = 3000  # больше точек страница не отмечает кольцами; остальное только считаем
MIN_QUERY_LENGTH = 2
MIN_ROOT_ID_LENGTH = 10  # номер FlyWire — длинное число; короткие числа ищем как обычный текст


def _text(value) -> str | None:
    return None if pd.isna(value) else str(value)


class NeuronLookup:
    """Всё, что можно узнать про нейрон по его внутреннему номеру (номеру строки таблицы)."""

    def __init__(self, neurons: pd.DataFrame, connectivity: dict, names: dict, legend: list[dict]):
        self.neurons = neurons
        self.indptr = connectivity["indptr"]
        self.indices = connectivity["indices"]
        self.weight = connectivity["weight"]
        self.names = names
        self.shown = class_codes(neurons, legend) != NO_POSITION
        self.cell_names = names["cell_types"]
        self._prepare_search()

    # ---------- подпись нейрона ----------

    def label(self, cell_type: str | None) -> tuple[str, str | None]:
        """(подпись, пояснение): для известных типов «MN9» + «мотонейрон хоботка…», для прочих просто тип."""
        if cell_type is None:
            return "без типа", None
        entry = self.cell_names.get(cell_type)
        if entry is None:
            return cell_type, None
        return entry["short"], entry["name"]

    def info(self, index: int) -> dict:
        row = self.neurons.iloc[index]
        cell_type = _text(row["cell_type"])
        label, about = self.label(cell_type)
        super_class = _text(row["super_class"]) or "missing"
        side = _text(row["side"]) or "missing"
        transmitter = _text(row["nt"]) or "missing"
        sign = int(row["sign"])
        return {
            "index": index,
            "flywire_id": str(int(row["root_id"])),  # строкой: в JavaScript большие числа теряют точность
            "label": label,
            "about": about,
            "cell_type": cell_type,
            "class_key": super_class,
            "class_name": self.names["super_classes"].get(super_class, super_class),
            "side_name": self.names["sides"].get(side, side),
            "transmitter_name": self.names["neurotransmitters"].get(transmitter, transmitter),
            "sign": sign,
            "sign_name": self.names["signs"][sign],
            "position_source": row["pos_source"],
        }

    # ---------- связи ----------

    def links(self, index: int, top: int = LINKS_PER_SIDE) -> dict:
        """Самые сильные входы и выходы. Сила — число синапсов; знак — знак отправителя в модели Shiu."""
        start, stop = self.indptr[index], self.indptr[index + 1]
        inputs = self._strongest(self.indices[start:stop], self.weight[start:stop], top)

        positions = np.flatnonzero(self.indices == index)  # где нейрон значится отправителем
        receivers = np.searchsorted(self.indptr, positions, side="right") - 1
        outputs = self._strongest(receivers, self.weight[positions], top)
        return {"index": index, "inputs": inputs, "outputs": outputs,
                "inputs_total": int(stop - start), "outputs_total": int(len(positions))}

    def _strongest(self, partners: np.ndarray, weights: np.ndarray, top: int) -> list[dict]:
        order = np.argsort(-np.abs(weights.astype(np.int32)), kind="stable")[:top]
        return [
            {
                "index": int(partners[i]),
                "label": self.label(_text(self.neurons["cell_type"].iat[partners[i]]))[0],
                "synapses": int(abs(weights[i])),
                "sign": int(np.sign(weights[i])),
                "visible": bool(self.shown[partners[i]]),  # у 14 нейронов нет координат (D-025)
            }
            for i in order
        ]

    # ---------- поиск ----------

    def _prepare_search(self) -> None:
        table = self.neurons
        lower = lambda column: table[column].fillna("").str.lower()  # noqa: E731
        short = table["cell_type"].map({k: v["short"] for k, v in self.cell_names.items()})
        about = table["cell_type"].map({k: v["name"] for k, v in self.cell_names.items()})
        self._names = {
            "cell_type": lower("cell_type"),
            "hemibrain_type": lower("hemibrain_type"),
            "short": short.fillna("").str.lower(),
        }
        self._text = {"synonyms": lower("synonyms"), "about": about.fillna("").str.lower()}
        self._root_ids = table["root_id"].astype(str)

    def search(self, query: str) -> dict:
        """Нейроны по имени, типу или номеру FlyWire. Сначала точное совпадение, потом любое вхождение."""
        query = query.strip().lower()
        empty = {"query": query, "total": 0, "shown": 0, "label": None, "indices": [], "types": [],
                 "truncated": False, "exact": False}
        if len(query) < MIN_QUERY_LENGTH:
            return empty

        if query.isdigit() and len(query) >= MIN_ROOT_ID_LENGTH:
            mask, exact = (self._root_ids == query).to_numpy(), True
        else:
            mask = np.zeros(len(self.neurons), dtype=bool)
            for column in self._names.values():
                mask |= (column == query).to_numpy()
            exact = bool(mask.any())
            if not exact:
                for column in (*self._names.values(), *self._text.values()):
                    mask |= column.str.contains(query, regex=False).to_numpy()
        total = int(mask.sum())
        if total == 0:
            return empty

        visible = np.flatnonzero(mask & self.shown)
        types = self.neurons.loc[mask, "cell_type"].fillna("без типа").value_counts().head(6)
        type_rows = [{"label": self.label(None if name == "без типа" else name)[0], "count": int(count)}
                     for name, count in types.items()]
        label = type_rows[0]["label"] if len(types) == 1 else f"{total} нейронов"
        return {
            "query": query, "total": total, "shown": int(len(visible)), "label": label,
            "indices": visible[:MAX_SEARCH_POINTS].tolist(), "types": type_rows,
            "truncated": len(visible) > MAX_SEARCH_POINTS, "exact": exact,
        }
