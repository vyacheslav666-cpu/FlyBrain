// Высоту панелей можно менять мышью, как у окна Windows (D-038): тянуть за край, обращённый к соседней панели.
// Верхние панели (стимулы, исполнители) — за нижний край, нижние («Вид», живой мозг) — за верхний.
// Двойной щелчок по краю возвращает обычную высоту. Высоты запоминаются в браузере (localStorage).

const STORAGE_KEY = "flybrain.panels";
const MIN_HEIGHT_PX = 70;
const PANELS = [
  { id: "stimuli", edge: "bottom" },
  { id: "executors", edge: "bottom" },
  { id: "view", edge: "top" },
  { id: "live", edge: "top" },
];

function loadHeights() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? {};
  } catch {
    return {};
  }
}

function saveHeights(heights) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(heights));
  } catch {
    // нет места или запрещено — не страшно
  }
}

/** Содержимое панели — во внутреннюю прокручиваемую обёртку: сама панель не прокручивается, и ручка на краю не уезжает. */
function wrapContent(panel) {
  const body = document.createElement("div");
  body.className = "panel-body";
  body.append(...panel.childNodes);
  panel.append(body);
}

export function createPanelResizers() {
  const heights = loadHeights();

  for (const { id, edge } of PANELS) {
    const panel = document.getElementById(id);
    wrapContent(panel);
    panel.classList.add("resizable");
    const handle = document.createElement("div");
    handle.className = `resize-handle resize-${edge}`;
    handle.title = "Потяните, чтобы изменить высоту панели. Двойной щелчок — обычная высота";
    panel.append(handle);

    // Наибольшая высота задана в style.css (max-height): панель не залезает на соседку и на подсказку внизу.
    const clamp = (height) => {
      const max = parseFloat(getComputedStyle(panel).maxHeight);
      return Math.max(MIN_HEIGHT_PX, Number.isFinite(max) ? Math.min(height, max) : height);
    };
    const setHeight = (height) => {
      if (height === null) panel.style.removeProperty("height");
      else panel.style.height = `${Math.round(height)}px`;
      panel.classList.toggle("sized", height !== null);
    };
    if (Number.isFinite(heights[id])) setHeight(heights[id]);

    let drag = null;
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      drag = { y: event.clientY, height: panel.offsetHeight };
      document.body.classList.add("resizing");
      handle.classList.add("active");
    });
    handle.addEventListener("pointermove", (event) => {
      if (!drag) return;
      const dy = event.clientY - drag.y;
      setHeight(clamp(drag.height + (edge === "bottom" ? dy : -dy)));
    });
    const finish = () => {
      if (!drag) return;
      drag = null;
      document.body.classList.remove("resizing");
      handle.classList.remove("active");
      heights[id] = panel.offsetHeight;
      saveHeights(heights);
    };
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
    handle.addEventListener("dblclick", () => {
      setHeight(null);
      delete heights[id];
      saveHeights(heights);
    });
  }
}
