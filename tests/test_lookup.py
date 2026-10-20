"""Тесты справок про нейроны (коммит C8): подсказка, самые сильные связи, поиск."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from flybrain import processed
from flybrain.lookup import NeuronLookup
from flybrain.paths import PROCESSED_DIR
from flybrain.server import create_app

LEGEND = [{"key": "optic", "color": "#4f8dff"}, {"key": "motor", "color": "#ffe14d"}]
NAMES = {
    "cell_types": {"CB0701": {"short": "MN9", "name": "мотонейрон хоботка"}},
    "super_classes": {"optic": "зрительная доля", "motor": "моторный", "missing": "без аннотации"},
    "sides": {"left": "левая", "right": "правая", "missing": "без аннотации"},
    "neurotransmitters": {"acetylcholine": "ацетилхолин", "gaba": "ГАМК", "missing": "не предсказан"},
    "signs": {1: "возбуждающий", -1: "тормозящий", 0: "нет выходных связей"},
}


def _lookup() -> NeuronLookup:
    neurons = pd.DataFrame(
        {
            "root_id": [720575940000000001, 720575940000000002, 720575940000000003, 720575940000000004],
            "x_um": [-1.0, -2.0, -3.0, np.nan],
            "y_um": [1.0, 2.0, 3.0, np.nan],
            "z_um": [0.0, 0.0, 0.0, np.nan],
            "pos_source": ["soma", "soma", "anchor", "none"],
            "super_class": ["motor", "motor", "optic", None],
            "cell_type": ["CB0701", "CB0701", "LPLC2", None],
            "hemibrain_type": [None, None, "LC22", None],
            "synonyms": [None, None, "Wu 2016: LPLC2", None],
            "side": ["left", "right", "left", None],
            "nt": ["acetylcholine", "acetylcholine", "gaba", None],
            "sign": [1, 1, -1, 0],
        }
    )
    # Строка — получатель, в `indices` — отправители. Нейрон 0 получает от 1 (+5) и 2 (−3); 1 получает от 0 (+2); 3 — от 0 (+7).
    connectivity = {
        "indptr": np.array([0, 2, 3, 3, 4]),
        "indices": np.array([1, 2, 0, 0], dtype=np.int32),
        "weight": np.array([5, -3, 2, 7], dtype=np.int16),
    }
    return NeuronLookup(neurons, connectivity, NAMES, LEGEND)


def test_info_has_everything_for_the_tooltip():
    info = _lookup().info(0)
    assert info["label"] == "MN9" and info["about"] == "мотонейрон хоботка"
    assert (info["class_name"], info["side_name"]) == ("моторный", "левая")
    assert (info["transmitter_name"], info["sign_name"]) == ("ацетилхолин", "возбуждающий")
    assert info["flywire_id"] == "720575940000000001"  # строкой, чтобы JavaScript не округлил


def test_info_for_neuron_without_annotation_does_not_crash():
    info = _lookup().info(3)
    assert info["class_name"] == "без аннотации" and info["label"] == "без типа"
    assert info["sign_name"] == "нет выходных связей"


def test_links_are_sorted_by_strength_and_know_direction_and_sign():
    links = _lookup().links(0)
    assert [(p["index"], p["synapses"], p["sign"]) for p in links["inputs"]] == [(1, 5, 1), (2, 3, -1)]
    assert [(p["index"], p["synapses"]) for p in links["outputs"]] == [(3, 7), (1, 2)]
    assert (links["inputs_total"], links["outputs_total"]) == (2, 2)
    assert links["outputs"][0]["visible"] is False  # у нейрона 3 нет координат


def test_links_respect_the_limit():
    assert len(_lookup().links(0, top=1)["inputs"]) == 1


def test_search_by_short_name_type_and_synonym():
    lookup = _lookup()
    assert lookup.search("MN9")["indices"] == [0, 1]
    assert lookup.search("mn9")["exact"] is True
    assert lookup.search("LPLC2")["indices"] == [2]  # и по типу, и по синониму — один нейрон
    assert lookup.search("LC22")["indices"] == [2]  # тип из мозга-образца (hemibrain)
    by_text = lookup.search("хоботка")  # по русскому описанию
    assert by_text["indices"] == [0, 1] and by_text["exact"] is False


def test_search_by_flywire_id_and_edge_cases():
    lookup = _lookup()
    assert lookup.search("720575940000000002")["indices"] == [1]
    assert lookup.search("m")["total"] == 0  # слишком короткий запрос
    assert lookup.search("несуществующее")["total"] == 0


def test_search_does_not_return_neurons_without_coordinates():
    lookup = _lookup()
    result = lookup.search("720575940000000004")
    assert (result["total"], result["shown"], result["indices"]) == (1, 0, [])


needs_data = pytest.mark.skipif(
    not (PROCESSED_DIR / processed.BUILD_INFO_FILE).exists(), reason="нет data/processed — запустите build-data"
)


@needs_data
def test_server_answers_about_neurons_and_search():
    with TestClient(create_app()) as client:
        search = client.get("/api/search", params={"q": "MN9"}).json()
        assert search["total"] == 2 and search["shown"] == 2 and search["exact"] is True
        left, right = search["indices"]

        info = client.get(f"/api/neuron/{left}").json()
        assert info["label"] == "MN9" and info["class_key"] == "motor"
        assert {info["side_name"], client.get(f"/api/neuron/{right}").json()["side_name"]} == {"левая", "правая"}

        links = client.get(f"/api/neuron/{left}/links").json()
        assert len(links["inputs"]) == 20 and len(links["outputs"]) == 20
        synapses = [p["synapses"] for p in links["inputs"]]
        assert synapses == sorted(synapses, reverse=True)

        assert client.get("/api/neuron/138639").status_code == 404
        assert client.get("/api/neuron/-1").status_code == 404
        assert client.get("/api/search", params={"q": "LPLC2"}).json()["total"] == 210
