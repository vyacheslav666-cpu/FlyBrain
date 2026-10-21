// Связь с «живым» мозгом на сервере (C9): кадры спайков по WebSocket, кнопки и текущая частота нейронов.
// Формат кадра задан в flybrain/live.py (FRAME_HEADER); числа должны совпадать.

const FRAME_FORMAT = 1;
const HEADER_BYTES = 32; // uint32 версия, uint32 эпоха, float64 начало и конец (мс модели), float32 скорость, uint32 число
const RATE_WINDOW_MS = 1000; // «текущая частота» — спайки за последнюю 1 с модельного времени
const RECONNECT_MS = 2000;

/**
 * `count` — число нейронов. `onFrame(neurons, spikes)` — кто выстрелил в кадре; `onStatus(status)` — состояние
 * сервера; `onReset()` — модель сброшена, всё гасим.
 */
export function connectLive({ count, onFrame, onStatus, onReset, onConnection }) {
  const windowCounts = new Uint16Array(count); // спайки каждого нейрона в окне частоты
  const frames = []; // кадры внутри окна: {start, end, neurons, spikes}
  let windowStart = 0; // начало окна, мс модели
  let latestEnd = 0;
  let epoch = null;
  let socket = null;
  let status = null;

  function clearWindow() {
    windowCounts.fill(0);
    frames.length = 0;
    windowStart = 0;
    latestEnd = 0;
  }

  function changeEpoch(next) {
    if (epoch !== null && next !== epoch) {
      clearWindow();
      onReset();
    }
    epoch = next;
  }

  function acceptFrame(buffer) {
    const view = new DataView(buffer);
    if (view.getUint32(0, true) !== FRAME_FORMAT) return;
    changeEpoch(view.getUint32(4, true));
    const start = view.getFloat64(8, true);
    const end = view.getFloat64(16, true);
    const speed = view.getFloat32(24, true);
    const n = view.getUint32(28, true);
    const neurons = new Uint32Array(buffer, HEADER_BYTES, n);
    const spikes = new Uint16Array(buffer, HEADER_BYTES + 4 * n, n);

    // Кадр пропал (страница не успевала их забирать): окно частоты начинаем заново, иначе частота занижалась бы.
    if (frames.length > 0 && Math.abs(start - latestEnd) > 1e-6) clearWindow();
    if (frames.length === 0) windowStart = start;
    for (let k = 0; k < n; k++) windowCounts[neurons[k]] += spikes[k];
    frames.push({ start, end, neurons, spikes });
    latestEnd = end;
    while (frames.length > 1 && frames[0].end <= latestEnd - RATE_WINDOW_MS) {
      const old = frames.shift();
      for (let k = 0; k < old.neurons.length; k++) windowCounts[old.neurons[k]] -= old.spikes[k];
      windowStart = frames[0].start;
    }
    if (status) {
      status.time_ms = end;
      status.speed = speed;
    }
    onFrame(neurons, spikes);
  }

  function open() {
    const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/live`;
    socket = new WebSocket(url);
    socket.binaryType = "arraybuffer";
    socket.addEventListener("open", () => onConnection(true));
    socket.addEventListener("message", (event) => {
      if (typeof event.data !== "string") {
        acceptFrame(event.data);
        return;
      }
      const message = JSON.parse(event.data);
      if (message.type !== "status") return;
      changeEpoch(message.epoch);
      status = message;
      onStatus(message);
    });
    socket.addEventListener("close", () => {
      onConnection(false);
      setTimeout(open, RECONNECT_MS); // сервер перезапускают — подключаемся снова сами
    });
  }
  open();

  return {
    send(command) {
      if (socket && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify(command));
    },
    /** Текущая частота нейрона, Гц, за последнюю 1 с модели (или меньше, если модель считает меньше секунды). */
    rateOf(index) {
      const span = latestEnd - windowStart;
      if (frames.length === 0 || span <= 0) return null;
      return { hz: (windowCounts[index] * 1000) / span, spanMs: span };
    },
  };
}
