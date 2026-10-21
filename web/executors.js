// Панель «исполнителей» (C11, D-035): моторные и нисходящие нейроны по текущей частоте и блок «ожидается по статье».
// Список и ожидания приходят с сервера (/api/executors), частоты считаются по потоку спайков (live.countsIn).

const BAR_WINDOW_MS = 200; // полоски и порядок: спайки за последние 0,2 с модели (один спайк = 5 Гц), чтобы они быстро падали
const CHECK_WINDOW_MS = 1000; // галочка ✓: частота за последнюю 1 с модели, как в проверке D-018
const BAR_FULL_HZ = 100; // полоска во всю длину — от 100 Гц и выше
const PAGE_ROWS = 20;
const FILTER_ALL = "all";

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const formatHz = (hz) => `${hz.toLocaleString("ru-RU", { maximumFractionDigits: hz < 10 ? 1 : 0 })} Гц`;

/**
 * `data` — ответ /api/executors, `stimulusNames` — ключ кнопки → имя («Сахар»), `source.countsIn` — из connectLive,
 * `onSelect(index)` — щелчок по нейрону (камера подлетает и нейрон закрепляется).
 */
export function createExecutorPanel(data, stimulusNames, source, onSelect) {
  const root = document.getElementById("executors");
  const expectedBox = document.getElementById("exec-expected");
  const filters = document.getElementById("exec-filters");
  const summary = document.getElementById("exec-summary");
  const list = document.getElementById("exec-list");
  const moreButton = document.getElementById("exec-more");
  const indices = data.neurons.map((neuron) => neuron.i);
  let status = null;
  let filter = FILTER_ALL;
  let shown = PAGE_ROWS;
  let pinned = -1;
  const rows = [];

  // ---------- фильтр по классу ----------
  const filterButtons = new Map();
  for (const [key, name, count] of [[FILTER_ALL, "Все", indices.length],
    ...data.classes.map((item) => [item.key, item.name[0].toUpperCase() + item.name.slice(1), item.count])]) {
    const button = element("button", "exec-filter", `${name} (${count.toLocaleString("ru-RU")})`);
    button.type = "button";
    button.addEventListener("click", () => {
      filter = key;
      shown = PAGE_ROWS;
      for (const [other, node] of filterButtons) node.setAttribute("aria-pressed", String(other === key));
      update();
    });
    button.setAttribute("aria-pressed", String(key === filter));
    filterButtons.set(key, button);
    filters.append(button);
  }
  moreButton.addEventListener("click", () => {
    shown = moreButton.dataset.mode === "less" ? PAGE_ROWS : shown + PAGE_ROWS;
    update();
  });

  // ---------- список ----------
  function buildRow() {
    const node = element("button", "exec-row");
    node.type = "button";
    const name = element("span", "exec-name");
    const meta = element("span", "exec-meta");
    const hz = element("span", "exec-hz");
    const bar = element("span", "exec-bar");
    const fill = element("i");
    bar.append(fill);
    node.append(name, hz, meta, bar);
    node.addEventListener("click", () => onSelect(Number(node.dataset.index)));
    return { node, name, meta, hz, fill, index: -1, text: "", width: -1 };
  }

  function bindRow(row, neuron, hz) {
    if (row.index !== neuron.i) {
      row.index = neuron.i;
      row.node.dataset.index = String(neuron.i);
      row.name.textContent = neuron.name;
      const parts = [data.classes.find((item) => item.key === neuron.class).name, sideName(neuron.side)];
      if (neuron.type) parts.push(neuron.type);
      row.meta.textContent = parts.join(" · ");
      row.node.title = neuron.about ?? `${neuron.name}: у этого типа нет русского имени в проекте, показан тип FlyWire`;
    }
    const text = hz > 0 ? formatHz(hz) : "0 Гц";
    if (row.text !== text) {
      row.text = text;
      row.hz.textContent = text;
      row.node.classList.toggle("silent", hz <= 0);
    }
    const width = Math.round(Math.min(hz / BAR_FULL_HZ, 1) * 100);
    if (row.width !== width) {
      row.width = width;
      row.fill.style.width = `${width}%`;
    }
    row.node.classList.toggle("pinned", neuron.i === pinned);
  }

  function sideName(side) {
    return side === "left" ? "левый" : side === "right" ? "правый" : side === "center" ? "по центру" : "сторона не указана";
  }

  function updateList() {
    const { counts, spanMs } = source.countsIn(indices, BAR_WINDOW_MS);
    const hz = new Float32Array(counts.length);
    let active = 0;
    if (spanMs > 0) {
      for (let k = 0; k < counts.length; k++) {
        hz[k] = (counts[k] * 1000) / spanMs;
        if (counts[k] > 0) active++;
      }
    }
    const order = [];
    for (let k = 0; k < indices.length; k++) if (filter === FILTER_ALL || data.neurons[k].class === filter) order.push(k);
    order.sort((a, b) => hz[b] - hz[a]); // порядок при равных частотах не меняется (сортировка устойчивая)

    const visible = Math.min(shown, order.length);
    while (rows.length < visible) {
      const row = buildRow();
      rows.push(row);
      list.append(row.node);
    }
    for (let k = 0; k < rows.length; k++) {
      rows[k].node.hidden = k >= visible;
      if (k < visible) bindRow(rows[k], data.neurons[order[k]], hz[order[k]]);
    }
    summary.textContent = spanMs > 0
      ? `Сейчас активны: ${active.toLocaleString("ru-RU")} из ${order.length.toLocaleString("ru-RU")}` +
        (active === 0 ? ". В покое мозг молчит — это свойство модели (D-019)." : "")
      : "Модель ещё не считала.";
    moreButton.hidden = order.length <= PAGE_ROWS;
    const rest = order.length - visible;
    moreButton.dataset.mode = rest > 0 ? "more" : "less";
    moreButton.textContent = rest > 0 ? `Показать ещё ${Math.min(PAGE_ROWS, rest)} (осталось ${rest.toLocaleString("ru-RU")})` : "Свернуть";
  }

  // ---------- «ожидается по статье» ----------
  function isOn(key) {
    return Boolean(status?.stimuli?.[key]?.on);
  }

  /** ✓ — частота за последнюю 1 с модели не меньше порога D-018; «…» — модель ещё не набрала секунду и спайков не было. */
  function verdict(count, spanMs) {
    const hz = spanMs > 0 ? (count * 1000) / Math.min(spanMs, CHECK_WINDOW_MS) : 0;
    if (hz >= data.active_hz) return { mark: "✓", state: "yes", hz };
    return spanMs < CHECK_WINDOW_MS * 0.95 ? { mark: "…", state: "wait", hz } : { mark: "✗", state: "no", hz };
  }

  function updateExpected() {
    const onKeys = Object.keys(status?.stimuli ?? {}).filter(isOn);
    const matching = data.expected.entries.filter((entry) => entry.needs.every(isOn));
    // Запись с большим набором стимулов вытесняет запись про те же нейроны с меньшим (сахар + горечь, а не один сахар).
    const entries = matching.filter((entry) => !matching.some((other) => other.name === entry.name &&
      other.needs.length > entry.needs.length && entry.needs.every((key) => other.needs.includes(key))));
    const caveats = data.expected.caveats.filter(
      (item) => isOn(item.stimulus) && (!item.side || status.stimuli[item.stimulus].side === item.side) &&
        (!item.alone || onKeys.length === 1));
    expectedBox.replaceChildren();
    if (entries.length === 0 && caveats.length === 0) {
      expectedBox.append(element("p", "exec-hint", "Включите стимул — здесь появится, какой «исполнитель» по статье должен сработать."));
      return;
    }
    const all = entries.flatMap((entry) => entry.neurons.map((neuron) => neuron.i));
    const { counts, spanMs } = source.countsIn(all, CHECK_WINDOW_MS);
    let next = 0;
    for (const entry of entries) {
      const box = element("div", "exp-entry");
      const head = element("div", "exp-head");
      const title = entry.needs.map((key) => stimulusNames[key] ?? key).join(" + ");
      head.append(element("span", "exp-title", `${title} → ${entry.name}`));
      if (entry.source === "own") {
        const badge = element("span", "exp-badge", "не из статьи, эксперимент");
        badge.title = entry.honesty || "Не из статьи Shiu: нейрон выбран нами.";
        head.append(badge);
      }
      box.append(head);
      const cells = element("div", "exp-cells");
      for (const neuron of entry.neurons) {
        const count = counts[next++];
        const result = entry.test === "active" ? verdict(count, spanMs) : { mark: "·", state: "info", hz: spanMs > 0 ? (count * 1000) / Math.min(spanMs, CHECK_WINDOW_MS) : 0 };
        const cell = element("button", `exp-cell ${result.state}`);
        cell.type = "button";
        cell.title = "Подлететь к нейрону";
        cell.append(element("span", "exp-mark", result.mark), element("span", "exp-side", neuron.side_name),
          element("span", "exp-hz", result.state === "wait" ? "считаю…" : formatHz(result.hz)));
        cell.addEventListener("click", () => onSelect(neuron.i));
        cells.append(cell);
      }
      box.append(cells);
      const why = entry.test === "weaker"
        ? `${entry.why}. Галочку не ставим: нужен ответ на один сахар для сравнения.`
        : `${entry.why}. ✓ — не меньше ${data.active_hz.toLocaleString("ru-RU")} Гц за последнюю 1 с модели (порог D-018).`;
      box.append(element("p", "exp-why", why));
      if (entry.source === "own" && entry.honesty) box.append(element("p", "exp-why honesty", `[ЧЕСТНОСТЬ] ${entry.honesty}`));
      expectedBox.append(box);
    }
    for (const caveat of caveats) expectedBox.append(element("p", "exp-why", caveat.text));
  }

  function update() {
    updateList();
    updateExpected();
  }

  root.hidden = false;
  update();

  return {
    setStatus(next) { status = next; updateExpected(); },
    setPinned(index) { pinned = index; for (const row of rows) row.node.classList.toggle("pinned", row.index === pinned); },
    update,
  };
}
