"""Откуда приходит вход нейрона: доли пяти органов чувств (для эксперимента P1, D-056).

[ЧЕСТНОСТЬ] У сенсорных нейронов модальность записана в таблице аннотаций (`cell_class`), а у остальных её нет.
Её считаем мы: идём от нейрона назад по входным синапсам, каждый раз выбирая вход с вероятностью, пропорциональной
числу синапсов (по модулю), пока не придём в сенсорный нейрон. Доля модальности m у нейрона — вероятность, что
такой обратный путь закончится в сенсорном нейроне модальности m (вероятность поглощения). Ведущая модальность —
та, у которой доля больше всех. Это свойство схемы связей, а не активности, и не то же самое, что «нейрон зрительный».

Сенсорные нейроны (затравка) — по `cell_class`: visual (фоторецепторы глаза и оцеллий), mechanosensory,
olfactory, gustatory (в том числе `sensory_ascending`), thermosensory и hygrosensory. Восходящие нейроны из
брюшного тяжа (`ascending`, `sensory_ascending` / AN) затравкой не считаются: их модальность в таблице не указана.
"""

import warnings

import numpy as np
import pandas as pd
import torch

MODALITIES = ("visual", "mechano", "olfactory", "gustatory", "thermo_hygro")
MODALITIES_RU = {
    "visual": "зрение",
    "mechano": "механорецепция",
    "olfactory": "обоняние",
    "gustatory": "вкус",
    "thermo_hygro": "температура и влажность",
    "none": "нет сенсорного входа",
}
SEED_CLASSES = {
    "visual": ("visual",),
    "mechano": ("mechanosensory",),
    "olfactory": ("olfactory",),
    "gustatory": ("gustatory",),
    "thermo_hygro": ("thermosensory", "hygrosensory"),
}
SENSORY_SUPER_CLASSES = ("sensory", "sensory_ascending")
MAX_STEPS = 400
TOLERANCE = 1e-7


def seed_matrix(neurons: pd.DataFrame) -> np.ndarray:
    """(N, 5): 1 там, где нейрон — сенсорный этой модальности."""
    super_class = neurons["super_class"].fillna("")
    cell_class = neurons["cell_class"].fillna("")
    sensory = super_class.isin(SENSORY_SUPER_CLASSES)
    return np.stack([(sensory & cell_class.isin(SEED_CLASSES[m])).to_numpy(np.float64) for m in MODALITIES], 1)


def input_shares(neurons: pd.DataFrame, indptr: np.ndarray, indices: np.ndarray,
                 weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Доли модальностей (N, 5; в сумме 1 или 0) и ведущая модальность (номер, −1 — путь не дошёл ни до кого)."""
    n = len(indptr) - 1
    seeds = seed_matrix(neurons)
    is_seed = torch.as_tensor(seeds.sum(1) > 0)
    w = np.abs(np.asarray(weight, dtype=np.float64))
    total = np.bincount(np.repeat(np.arange(n), np.diff(indptr)), weights=w, minlength=n)
    values = w / np.repeat(np.where(total > 0, total, 1.0), np.diff(indptr))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
        step = torch.sparse_csr_tensor(torch.as_tensor(np.asarray(indptr, dtype=np.int64)),
                                       torch.as_tensor(np.asarray(indices, dtype=np.int64)),
                                       torch.as_tensor(values), size=(n, n), check_invariants=True)
    fixed = torch.as_tensor(seeds)
    x = fixed.clone()
    for _ in range(MAX_STEPS):
        new = step @ x
        new[is_seed] = fixed[is_seed]  # сенсорный нейрон — конец пути
        change = float((new - x).abs().max())
        x = new
        if change < TOLERANCE:
            break
    x = x.numpy()
    mass = x.sum(1)
    shares = np.divide(x, mass[:, None], out=np.zeros_like(x), where=mass[:, None] > 0)
    dominant = np.where(mass > 0, shares.argmax(1), -1)
    return shares, dominant
