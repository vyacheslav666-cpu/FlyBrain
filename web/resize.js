// Размер панелей меняется мышью за уголок — как у карточки нейрона (D-038, D-042): ширина и высота сразу.
// Уголок стоит на углу, обращённом к середине экрана и к соседней панели в той же колонке:
// у стимулов — справа внизу, у «Вида» — справа вверху, у исполнителей — слева внизу, у живого мозга — слева вверху.
// Двойной щелчок по уголку возвращает обычный размер. Кнопка «−» в углу сворачивает панель до заголовка.
// Всё запоминается в браузере (localStorage).

const STORAGE_KEY = "flybrain.panels";
const MIN_HEIGHT_PX = 70;
const MIN_WIDTH_PX = 220;
const SIDE_GAP_PX = 30; // панель не шире половины окна минус этот зазор: колонки не наезжают друг на друга
const PANELS = [
  { id: "stimuli", corner: "br" },
  { id: "executors", corner: "bl" },
  { id: "view", corner: "tr" },
  { id: "live", corner: "tl" },
];

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? {};
    // До D-042 хранилась только высота числом.
    for (const [id, value] of Object.entries(saved)) if (typeof value === "number") saved[id] = { height: value };
    return saved;
  } catch {
    return {};
  }
}

function saveState(state) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // нет места или запрещено — не страшно
  }
}

/** Содержимое панели — во внутреннюю прокручиваемую обёртку: сама панель не прокручивается, и уголок не уезжает. */
function wrapContent(panel) {
  const body = document.createElement("div");
  body.className = "panel-body";
  body.append(...panel.childNodes);
  panel.append(body);
}

export function createPanelResizers() {
  const state = loadState();

  for (const { id, corner } of PANELS) {
    const panel = document.getElementById(id);
    const saved = (state[id] ??= {});
    wrapContent(panel);
    panel.classList.add("resizable");

    const title = document.createElement("div");
    title.className = "panel-collapsed-title";
    title.textContent = panel.querySelector("h2")?.textContent ?? "";
    const collapse = document.createElement("button");
    collapse.type = "button";
    collapse.className = "panel-collapse";
    const grip = document.createElement("div");
    grip.className = `corner-grip grip-${corner}`;
    grip.title = "Потяните, чтобы изменить размер панели. Двойной щелчок — обычный размер";
    panel.prepend(title);
    panel.append(collapse, grip);

    const growsDown = corner[0] === "b";
    const growsRight = corner[1] === "r";
    // Наибольшая высота задана в style.css (max-height): панель не залезает на соседку и на подсказку внизу.
    // Если окно стало меньше, панель ужимается только на экране, а запомненный размер не меняется.
    const maxWidth = () => window.innerWidth / 2 - SIDE_GAP_PX;
    const clampHeight = (height) => {
      const max = parseFloat(getComputedStyle(panel).maxHeight);
      return Math.max(MIN_HEIGHT_PX, Number.isFinite(max) ? Math.min(height, max) : height);
    };
    const clampWidth = (width) => Math.max(MIN_WIDTH_PX, Math.min(width, maxWidth()));

    function place() {
      if (Number.isFinite(saved.height)) panel.style.height = `${Math.round(saved.height)}px`;
      else panel.style.removeProperty("height");
      if (Number.isFinite(saved.width)) panel.style.width = `${Math.round(clampWidth(saved.width))}px`;
      else panel.style.removeProperty("width");
      panel.classList.toggle("sized", Number.isFinite(saved.height));
      panel.classList.toggle("collapsed", Boolean(saved.collapsed));
      collapse.textContent = saved.collapsed ? "+" : "−";
      collapse.title = saved.collapsed ? "развернуть панель" : "свернуть панель (останется только заголовок)";
    }
    place();
    window.addEventListener("resize", place);
    if (id === "executors") {
      // Карточка нейрона по умолчанию стоит левее панели исполнителей — сдвигаем её вслед за шириной панели.
      new ResizeObserver(() => {
        if (panel.offsetWidth > 0) document.documentElement.style.setProperty("--executors-width", `${panel.offsetWidth}px`);
      }).observe(panel);
    }

    collapse.addEventListener("click", () => {
      saved.collapsed = !saved.collapsed;
      saveState(state);
      place();
    });

    let drag = null;
    grip.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      grip.setPointerCapture(event.pointerId);
      drag = { x: event.clientX, y: event.clientY, width: panel.offsetWidth, height: panel.offsetHeight };
      document.body.classList.add(growsDown === growsRight ? "sizing-nwse" : "sizing-nesw");
      grip.classList.add("active");
    });
    grip.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      saved.width = clampWidth(drag.width + (growsRight ? dx : -dx));
      saved.height = clampHeight(drag.height + (growsDown ? dy : -dy));
      place();
    });
    const finish = () => {
      if (!drag) return;
      drag = null;
      document.body.classList.remove("sizing-nwse", "sizing-nesw");
      grip.classList.remove("active");
      saveState(state);
    };
    grip.addEventListener("pointerup", finish);
    grip.addEventListener("pointercancel", finish);
    grip.addEventListener("dblclick", () => {
      delete saved.width;
      delete saved.height;
      saveState(state);
      place();
    });
  }
}
