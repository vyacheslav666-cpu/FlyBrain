import numpy as np
import pandas as pd
import pytest

from flybrain.build import BuildError, build_connectivity_csr, build_neuron_table, model_signs, \
    sign_nt_mismatches


def _annotations() -> pd.DataFrame:
    # Нейрон 10 — с телом клетки, 20 — только с опорной точкой, 30 — без аннотации.
    return pd.DataFrame({
        "root_id": [10, 20],
        "pos_x": [1000.0, 2000.0], "pos_y": [100.0, 200.0], "pos_z": [10.0, 20.0],
        "soma_x": [5000.0, np.nan], "soma_y": [500.0, np.nan], "soma_z": [50.0, np.nan],
        "super_class": ["central", "sensory"], "cell_class": [np.nan, "gustatory"],
        "cell_sub_class": [np.nan, "sugar"], "cell_type": ["CB0701", "LB3c"],
        "hemibrain_type": [np.nan, np.nan], "synonyms": [np.nan, np.nan],
        "side": ["left", "right"], "flow": ["intrinsic", "afferent"], "top_nt": ["glutamate", "acetylcholine"],
    })


def test_neuron_table_coordinates_side_and_sign():
    root_ids = np.array([10, 20, 30], dtype=np.int64)
    sign = model_signs(np.array([0, 1, 1]), np.array([1, -1, -1], dtype=np.int8), 3)
    table = build_neuron_table(root_ids, _annotations(), sign)

    assert table["index"].tolist() == [0, 1, 2]
    assert table["root_id"].tolist() == [10, 20, 30]
    # Тело клетки важнее опорной точки (D-016); X отражён (D-011); воксель 4 × 4 × 40 нм (D-010).
    assert table.loc[0, ["x_um", "y_um", "z_um"]].tolist() == pytest.approx([-20.0, 2.0, 2.0])
    assert table.loc[1, ["x_um", "y_um", "z_um"]].tolist() == pytest.approx([-8.0, 0.8, 0.8])
    assert table["pos_source"].tolist() == ["soma", "anchor", "none"]
    assert np.isnan(table.loc[2, "x_um"])
    assert table["side"].tolist()[:2] == ["left", "right"] and pd.isna(table.loc[2, "side"])
    # Знак — из файла связей модели, 0 у нейрона без выходных связей.
    assert table["sign"].tolist() == [1, -1, 0]
    # Нейрон 10: глутамат по аннотациям, но в модели возбуждающий; нейрон 20: ацетилхолин, но тормозящий.
    assert sign_nt_mismatches(table) == 2


def test_model_signs_reject_mixed_sign():
    with pytest.raises(BuildError, match="разного знака"):
        model_signs(np.array([0, 0]), np.array([1, -1], dtype=np.int8), 1)


def test_connectivity_csr_matches_dense_matrix():
    rng = np.random.default_rng(0)
    n = 7
    pairs = rng.choice(n * n, size=20, replace=False)
    pre, post = pairs // n, pairs % n
    weight = rng.integers(1, 50, size=20) * rng.choice([-1, 1], size=20)
    csr = build_connectivity_csr(post, pre, weight, n)

    dense = np.zeros((n, n), dtype=np.int64)  # строка — получатель, столбец — отправитель
    for row in range(n):
        start, stop = csr["indptr"][row], csr["indptr"][row + 1]
        dense[row, csr["indices"][start:stop]] = csr["weight"][start:stop]
    expected = np.zeros((n, n), dtype=np.int64)
    expected[post, pre] = weight
    np.testing.assert_array_equal(dense, expected)
    # Вход получателей от выстреливших отправителей 2 и 5.
    spikes = np.zeros(n, dtype=np.int64)
    spikes[[2, 5]] = 1
    np.testing.assert_array_equal(dense @ spikes, expected[:, 2] + expected[:, 5])


def test_connectivity_csr_rejects_duplicate_pairs():
    with pytest.raises(BuildError, match="повторяющиеся"):
        build_connectivity_csr(np.array([1, 1]), np.array([0, 0]), np.array([3, 4]), 2)
