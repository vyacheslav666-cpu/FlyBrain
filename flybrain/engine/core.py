"""Модель Shiu на PyTorch: видеокарта или процессор, несколько повторов сразу (C5, D-007, D-008).

Порядок операций за один шаг — как в Brian2 для `model.py` Shiu (расписание по умолчанию:
groups → thresholds → synapses → resets; проверено по исходникам Brian2 2.x):

1. **groups** — у нейронов, которые не «отдыхают», точное решение уравнений за шаг (метод 'linear'):
   g ← g·E, v ← v_0 + (v − v_0)·A + g·B. «Отдыхает» — если с последнего спайка прошло меньше
   `refractory_steps` шагов (`timestep(t − lastspike) < timestep(rfc)`); тогда v и g заморожены.
2. **thresholds** — спайк, если v > v_th и нейрон не отдыхает; запоминается шаг спайка.
3. **synapses** — сначала связи: g += w от спайков, случившихся ровно `delay_steps` шагов назад
   (у Brian2 это очередь спайков, а здесь — блок, см. ниже); затем стимулы: v += 68,75 мВ, если пришёл
   пуассоновский импульс (вероятность hz·dt за шаг, как `PoissonInput` с N = 1).
   Важно: v и g помечены `(unless refractory)`, и Brian2 тогда разрешает связям и стимулам писать в них
   только у нейронов, которые сейчас не отдыхают («conditional write»). Отдыхающий нейрон и нейрон,
   выстреливший в этом шаге, входы **теряют**, а не копят.
4. **resets** — у выстреливших в этом шаге: v ← v_rst, g ← 0.

Блок из 18 шагов (D-008). Задержка ровно 18 шагов, поэтому всё, что пришло за блок, выстрелено
за предыдущий блок. В начале блока одно разреженное умножение «связи × спайки прошлого блока» даёт
входы на все 18 шагов сразу — в точности то же, что 18 отдельных рассылок. На видеокарте блок
записывается в CUDA Graph один раз и дальше только повторяется.

Повторы — второй размер всех массивов: состояние (N, повторы), блок спайков (N, 18, повторы).
"""

import warnings

import numpy as np
import torch

from flybrain.engine.params import SHIU, ModelParams

NEVER_SPIKED = -(2**30)  # «последний спайк» до начала расчёта: давно, нейрон не отдыхает


class EngineError(Exception):
    """Движок нельзя построить с такими данными."""


def pick_device(name: str = "auto") -> torch.device:
    """`auto` — видеокарта, если PyTorch её видит, иначе процессор."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise EngineError("видеокарта недоступна для PyTorch; запустите с --device cpu")
    return torch.device(name)


class Engine:
    """Сеть из N нейронов; связи — CSR (строка — получатель, `indices` — отправители, `weight` — синапсы со знаком).

    Стимулы и выключение нейронов можно менять между вызовами `run_steps` — то есть на ходу.
    """

    def __init__(self, indptr: np.ndarray, indices: np.ndarray, weight: np.ndarray, n_trials: int = 1,
                 device: str | torch.device = "auto", params: ModelParams = SHIU, seed: int | None = None,
                 use_graph: bool = True, record_spikes: bool = False, dtype: torch.dtype = torch.float32):
        self.params = params
        self.dtype = dtype  # float32 — для скорости; float64 — как Brian2, для сверки в тестах
        self.device = pick_device(device) if isinstance(device, str) else device
        self.n_neurons = len(indptr) - 1
        self.n_trials = int(n_trials)
        if self.n_trials < 1:
            raise EngineError("число повторов должно быть не меньше 1")
        self.block_steps = params.delay_steps
        if self.block_steps < 1:
            raise EngineError("приём «блок шагов» требует задержки синапса хотя бы в один шаг")
        self.use_graph = bool(use_graph) and self.device.type == "cuda"
        self.record_spikes = record_spikes

        n, t, d = self.n_neurons, self.n_trials, self.block_steps
        dev = self.device
        # Вес в мВ = число синапсов со знаком × 0,275 мВ (D-005, D-025). Индексы 32-битные: экономят память.
        values = torch.as_tensor(np.asarray(weight, dtype=np.float64) * params.w_syn, dtype=dtype)
        index_dtype = torch.int32 if len(indices) < 2**31 else torch.int64
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta")
            self.weights = torch.sparse_csr_tensor(
                torch.as_tensor(np.asarray(indptr), dtype=index_dtype),
                torch.as_tensor(np.asarray(indices), dtype=index_dtype),
                values, size=(n, n), check_invariants=True).to(dev)  # кривые данные — ошибка, а не сбой

        a, b, e = params.decay_factors()
        self._a, self._b, self._e = a, b, e
        self._kick = params.poisson_weight
        self._p_per_hz = params.dt / 1000.0  # вероятность импульса за шаг = hz · dt (dt в секундах)

        self.v = torch.empty((n, t), dtype=dtype, device=dev)
        self.g = torch.empty((n, t), dtype=dtype, device=dev)
        self.last_spike = torch.empty((n, t), dtype=torch.int32, device=dev)
        self.counts = torch.zeros((n, t), dtype=torch.int32, device=dev)
        # Спайки текущего блока: для рассылки по связям (выключенные нейроны дают 0) и для записи.
        self._outbox = torch.zeros((n, d, t), dtype=dtype, device=dev)
        self._spiked = torch.zeros((n, d, t), dtype=torch.bool, device=dev)
        self._inbox = torch.zeros((n, d, t), dtype=dtype, device=dev)
        self._live = torch.ones((n, 1), dtype=dtype, device=dev)  # 0 — нейрон выключен (его связи не действуют)
        self._ref_steps = torch.full((n, 1), params.refractory_steps, dtype=torch.int32, device=dev)
        self._block_start = torch.zeros((), dtype=torch.int32, device=dev)  # номер шага начала блока

        self._generator = torch.Generator(device=dev)
        if seed is None:
            self._generator.seed()  # случайное зерно
        else:
            self._generator.manual_seed(seed)
        self._stimuli: dict[str, tuple[np.ndarray, float]] = {}
        self._stim_index = torch.zeros(0, dtype=torch.int64, device=dev)
        self._stim_p = torch.zeros((0, 1), device=dev)
        self._graph: torch.cuda.CUDAGraph | None = None

        self.spike_log: list[np.ndarray] = []  # при record_spikes: массивы (шаг, нейрон, повтор)
        self.reset()

    # ---------- управление ----------

    @property
    def step(self) -> int:
        """Сколько шагов посчитано с последнего `reset`."""
        return self._step

    @property
    def time_ms(self) -> float:
        return self._step * self.params.dt

    def reset(self) -> None:
        """Все нейроны в покое, счётчики спайков на нуле, время 0. Стимулы и выключения остаются."""
        self.v.fill_(self.params.v_0)
        self.g.zero_()
        self.last_spike.fill_(NEVER_SPIKED)
        self.counts.zero_()
        self._outbox.zero_()
        self._spiked.zero_()
        self._inbox.zero_()
        self._block_start.zero_()
        self._step = 0
        self.spike_log = []

    def set_stimulus(self, name: str, indices, hz: float) -> None:
        """Пуассоновский стимул `name` на нейроны `indices` с частотой `hz` (как `PoissonInput` у Shiu).

        Пока стимул существует, у его нейронов нет «отдыха» (`rfc = 0` в `poi()` Shiu), даже при 0 Гц.
        Повторный вызов с тем же именем меняет частоту или состав на ходу.
        """
        indices = np.asarray(indices, dtype=np.int64).ravel()
        if len(indices) and (indices.min() < 0 or indices.max() >= self.n_neurons):
            raise EngineError(f"стимул «{name}»: номер нейрона вне сети")
        if hz < 0 or hz * self._p_per_hz > 1:
            raise EngineError(f"стимул «{name}»: частота должна быть от 0 до {1 / self._p_per_hz:g} Гц")
        self._stimuli[name] = (indices, float(hz))
        self._update_stimuli()

    def remove_stimulus(self, name: str) -> None:
        self._stimuli.pop(name, None)
        self._update_stimuli()

    def set_silenced(self, indices, silenced: bool = True) -> None:
        """Выключение как `silence()` у Shiu: вес всех выходных связей нейрона = 0. Сам нейрон считается дальше."""
        index = torch.as_tensor(np.asarray(indices, dtype=np.int64).ravel(), device=self.device)
        self._live[index] = 0.0 if silenced else 1.0

    def _update_stimuli(self) -> None:
        parts = [(indices, hz) for indices, hz in self._stimuli.values()]
        index = np.concatenate([indices for indices, _ in parts]) if parts else np.zeros(0, dtype=np.int64)
        p = np.concatenate([np.full(len(indices), hz * self._p_per_hz) for indices, hz in parts]) if parts \
            else np.zeros(0)
        self._ref_steps.fill_(self.params.refractory_steps)
        if len(index):
            self._ref_steps[torch.as_tensor(index, device=self.device)] = 0
        if len(index) == len(self._stim_index):
            # Та же форма — пишем в те же массивы: записанный CUDA Graph читает их и увидит новое.
            self._stim_index.copy_(torch.as_tensor(index, device=self.device))
            self._stim_p.copy_(torch.as_tensor(p, dtype=torch.float32, device=self.device)[:, None])
        else:
            self._stim_index = torch.as_tensor(index, device=self.device)
            self._stim_p = torch.as_tensor(p, dtype=torch.float32, device=self.device)[:, None]
            self._graph = None  # другая форма — блок записывается заново

    # ---------- расчёт ----------

    def prepare(self) -> None:
        """Заранее записать блок в CUDA Graph (иначе это случится при первом расчёте)."""
        if self.use_graph and self._graph is None:
            self._graph = self._capture()

    def synchronize(self) -> None:
        """Дождаться, пока видеокарта досчитает всё отправленное (для замеров времени и прогресса)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run_ms(self, duration_ms: float) -> None:
        """Посчитать `duration_ms` миллисекунд (целое число шагов, как `net.run` в Brian2)."""
        self.run_steps(int(round(duration_ms / self.params.dt)))

    def run_steps(self, n_steps: int) -> None:
        while n_steps > 0:
            position = self._step % self.block_steps
            if position == 0 and n_steps >= self.block_steps:
                self._run_block()
                taken = self.block_steps
            else:
                # Хвост, не кратный блоку: те же операции, по одному шагу (входы блока уже посчитаны).
                taken = min(n_steps, self.block_steps - position)
                for j in range(position, position + taken):
                    if j == 0:
                        self._deliver()
                    self._one_step(j, self._random_kicks(1)[0])
                if position + taken == self.block_steps:
                    self._block_start.add_(self.block_steps)
            if self.record_spikes:
                self._log_spikes(position, taken)
            self._step += taken
            n_steps -= taken

    def _run_block(self) -> None:
        if not self.use_graph:
            self._block_body()
            return
        self.prepare()
        self._graph.replay()

    def _block_body(self) -> None:
        self._deliver()
        kicks = self._random_kicks(self.block_steps)
        for j in range(self.block_steps):
            self._one_step(j, kicks[j])
        self._block_start.add_(self.block_steps)

    def _capture(self) -> "torch.cuda.CUDAGraph":
        # Запись меняет состояние, поэтому сохраняем его и возвращаем после (прогрев + запись = 2 блока).
        saved = [tensor.clone() for tensor in self._state_tensors()]
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            self._block_body()  # прогрев: cuSPARSE и распределитель памяти готовят всё заранее
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._block_body()
        for tensor, value in zip(self._state_tensors(), saved):
            tensor.copy_(value)
        return graph

    def _state_tensors(self) -> list[torch.Tensor]:
        return [self.v, self.g, self.last_spike, self.counts, self._outbox, self._spiked, self._inbox,
                self._block_start]

    def _deliver(self) -> None:
        """Входы на весь блок: связи × спайки прошлого блока (у выключенных нейронов спайки = 0)."""
        n, d, t = self._outbox.shape
        self._inbox.copy_(torch.sparse.mm(self.weights, self._outbox.view(n, d * t)).view(n, d, t))

    def _random_kicks(self, steps: int) -> torch.Tensor:
        """Импульсы стимулов, мВ: (шаги, стимулируемые нейроны, повторы)."""
        m = len(self._stim_index)
        noise = torch.rand((steps, m, self.n_trials), generator=self._generator, device=self.device)
        return (noise < self._stim_p).to(self.dtype) * self._kick

    def _one_step(self, j: int, kicks: torch.Tensor) -> None:
        p = self.params
        v, g = self.v, self.g
        now = self._block_start + j
        # 1. groups: точное решение за шаг, кроме «отдыхающих».
        active = (now - self.last_spike) >= self._ref_steps
        v_new = (v - p.v_0) * self._a + g * self._b + p.v_0
        torch.where(active, v_new, v, out=v)
        torch.where(active, g * self._e, g, out=g)
        # 2. thresholds.
        spike = (v > p.v_th) & active
        torch.where(spike, now, self.last_spike, out=self.last_spike)
        self._spiked[:, j, :] = spike
        self._outbox[:, j, :] = spike * self._live
        self.counts.add_(spike)
        # 3. synapses: сначала связи (задержка 18 шагов), затем стимулы; пишутся только у неотдыхающих.
        writable = active & ~spike
        g.add_(self._inbox[:, j, :] * writable)
        if len(self._stim_index):
            v.index_add_(0, self._stim_index, kicks * writable[self._stim_index])
        # 4. resets.
        v.masked_fill_(spike, p.v_rst)
        g.masked_fill_(spike, 0.0)

    def _log_spikes(self, start: int, steps: int) -> None:
        """Записать спайки шагов [start, start + steps) текущего блока (для тестов и потока в браузер)."""
        neuron, offset, trial = torch.nonzero(self._spiked[:, start:start + steps, :], as_tuple=True)
        events = torch.stack([offset + self._step, neuron, trial], dim=1)
        self.spike_log.append(events.cpu().numpy().astype(np.int64))

    # ---------- результаты ----------

    def spikes(self) -> np.ndarray:
        """Все записанные спайки (нужен `record_spikes=True`): строки (шаг, нейрон, повтор)."""
        if not self.spike_log:
            return np.zeros((0, 3), dtype=np.int64)
        events = np.concatenate(self.spike_log)
        return events[np.lexsort((events[:, 1], events[:, 0], events[:, 2]))]

    def rates_hz(self) -> np.ndarray:
        """Частота каждого нейрона в каждом повторе, Гц: массив (повторы, нейроны)."""
        seconds = self.time_ms / 1000.0
        if seconds == 0:
            return np.zeros((self.n_trials, self.n_neurons))
        return self.counts.T.cpu().numpy() / seconds
