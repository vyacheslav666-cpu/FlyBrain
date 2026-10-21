// Панель стимулов (C10, D-034): кнопки «Сахар», «Горечь», «Касание усика», «Надвигающаяся тень».
// У каждой кнопки — частота 0–200 Гц, сторона мухи и «ⓘ» с описанием: какие нейроны, сколько, откуда список.
// Описание кнопок приходит с сервера (/api/stimuli), состояние — в сообщениях «status» (поле stimuli).

const SEND_EVERY_MS = 100; // ползунок частоты шлёт команды не чаще 10 раз в секунду
const HZ_STEP = 5;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** `catalog` — ответ /api/stimuli; `onCommand(command)` отправляет команду на сервер. */
export function createStimulusPanel(catalog, onCommand) {
  const root = document.getElementById("stimuli");
  const list = document.getElementById("stimuli-list");
  const rows = new Map();
  let status = null;
  let connected = false;

  for (const button of catalog.buttons) {
    const row = buildRow(button);
    rows.set(button.key, row);
    list.append(row.node);
  }
  root.hidden = false;

  function buildRow(button) {
    const node = element("div", "stim");
    const toggle = element("button", "stim-toggle");
    toggle.type = "button";
    toggle.setAttribute("aria-pressed", "false");
    const name = element("span", "stim-name", button.name_ru);
    toggle.append(name);
    if (button.experiment) {
      const badge = element("span", "stim-badge", "⚗ эксперимент");
      badge.title = "Опыт, которого нет в статье Shiu: список нейронов выбрали мы";
      toggle.append(badge);
    }
    const infoButton = element("button", "stim-info-button", "ⓘ");
    infoButton.type = "button";
    infoButton.title = "Что именно включается";
    infoButton.setAttribute("aria-expanded", "false");
    const head = element("div", "stim-head");
    head.append(toggle, infoButton);

    const slider = element("input", "stim-hz");
    slider.type = "range";
    slider.min = "0";
    slider.max = String(button.max_hz);
    slider.step = String(HZ_STEP);
    slider.value = String(button.default_hz);
    slider.setAttribute("aria-label", `${button.name_ru}: частота, Гц`);
    const hzText = element("span", "stim-hz-text");
    const sides = element("div", "stim-sides");
    sides.setAttribute("role", "group");
    sides.setAttribute("aria-label", `${button.name_ru}: сторона мухи`);
    const sideButtons = new Map();
    for (const side of button.sides) {
      const sideButton = element("button", "stim-side", side.side === "left" ? "левая" : "правая");
      sideButton.type = "button";
      sideButton.title = `${side.label}: ${side.count} нейронов`;
      sideButton.addEventListener("click", () => {
        if (!button.side_locked) onCommand({ cmd: "stimulus", key: button.key, side: side.side });
      });
      sideButtons.set(side.side, sideButton);
      sides.append(sideButton);
    }
    if (button.side_locked) {
      sides.classList.add("locked");
      sides.title = button.side_note_ru;
    }
    const controls = element("div", "stim-controls");
    controls.append(slider, hzText, sides);

    const lockNote = element("p", "stim-note");
    if (button.side_locked) {
      const only = button.sides[0].side === "left" ? "левой" : "правой";
      lockNote.textContent = `Сторона не выбирается: в статье список только для ${only} стороны мухи.`;
    }
    const info = element("div", "stim-info");
    info.hidden = true;

    toggle.addEventListener("click", () => {
      onCommand({ cmd: "stimulus", key: button.key, on: !status?.stimuli?.[button.key]?.on });
    });
    infoButton.addEventListener("click", () => {
      info.hidden = !info.hidden;
      infoButton.setAttribute("aria-expanded", String(!info.hidden));
      if (!info.hidden) renderInfo(row);
    });

    // Частоту шлём не чаще раза в 100 мс, но последнее значение не теряем.
    let pending = null;
    let lastSent = 0;
    let timer = null;
    function flush() {
      timer = null;
      if (pending === null) return;
      onCommand({ cmd: "stimulus", key: button.key, hz: pending });
      pending = null;
      lastSent = performance.now();
    }
    slider.addEventListener("input", () => {
      pending = Number(slider.value);
      hzText.textContent = `${slider.value} Гц`;
      const wait = SEND_EVERY_MS - (performance.now() - lastSent);
      if (wait <= 0) flush();
      else if (timer === null) timer = setTimeout(flush, wait);
    });

    node.append(head, controls, lockNote, info);
    const row = { node, button, toggle, slider, hzText, sideButtons, info, infoButton };
    hzText.textContent = `${slider.value} Гц`;
    return row;
  }

  function renderInfo(row) {
    const side = status?.stimuli?.[row.button.key]?.side ?? row.button.default_side;
    const entry = row.button.sides.find((item) => item.side === side) ?? row.button.sides[0];
    row.info.replaceChildren(...entry.info.map((text) => element("p", text.startsWith("[ЧЕСТНОСТЬ]") ? "honesty" : "", text)));
    if (row.button.side_note_ru) {
      const prefix = row.button.side_locked ? "" : "Сторона: ";
      row.info.append(element("p", "side-note", `${prefix}${row.button.side_note_ru}`));
    }
  }

  function refresh() {
    const ready = connected && status && (status.state === "running" || status.state === "paused");
    for (const [key, row] of rows) {
      const state = status?.stimuli?.[key];
      row.toggle.disabled = row.slider.disabled = !ready;
      row.toggle.setAttribute("aria-pressed", String(Boolean(state?.on)));
      row.node.classList.toggle("on", Boolean(state?.on));
      if (state) {
        // Пока ползунок тянут, не двигаем его назад значением, которое сервер ещё не получил.
        if (document.activeElement !== row.slider) row.slider.value = String(state.hz);
        row.hzText.textContent = `${row.slider.value} Гц`;
        for (const [side, button] of row.sideButtons) {
          button.setAttribute("aria-pressed", String(side === state.side));
          button.disabled = !ready || row.button.side_locked;
        }
        if (!row.info.hidden) renderInfo(row);
      }
    }
  }

  return {
    setStatus(next) { status = next; refresh(); },
    setConnected(value) { connected = value; refresh(); },
    /** Включён ли хоть один стимул (тогда плашка «в покое мозг молчит» не нужна). */
    get anyOn() { return Object.values(status?.stimuli ?? {}).some((state) => state.on); },
  };
}
