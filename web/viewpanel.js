// Панель «Вид» (C12): ракурсы, «только активные», формы нейронов (C12b), контур мозга, снимок, яркость и шлейф.
// Настройки запоминаются в браузере (localStorage); если он недоступен — страница работает без запоминания.

const STORAGE_KEY = "flybrain.view";
const BRIGHTNESS = { min: 0.4, max: 2.5, start: 50 }; // ползунок 0…100 по логарифму; 50 — «как раньше» (1×)
const TRAIL = { min: 0.08, max: 1.5, start: 45 }; // секунды показа; 45 ≈ 0,3 с — как в C9

const positionToValue = (range, position) => range.min * (range.max / range.min) ** (position / 100);

function formatSeconds(seconds) {
  return `${seconds.toLocaleString("ru-RU", { maximumFractionDigits: seconds < 1 ? 2 : 1 })} с`;
}

function loadPrefs() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? {};
  } catch {
    return {};
  }
}

function savePrefs(prefs) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
  } catch {
    // нет места или запрещено — не страшно
  }
}

const SHAPES_TITLE = "По щелчку на нейрон показывает его настоящую форму — скелет из FlyWire (Schlegel et al. 2024, " +
  "CC BY 4.0), упрощённый. Только показ: модель не меняется";
const SHAPES_MISSING = "Формы нейронов не скачаны. Это необязательная загрузка 5,4 ГБ: команда " +
  "«uv run flybrain fetch-skeletons» в PowerShell в папке проекта, затем перезапустите run.bat";

/**
 * `handlers`: onView(name), onActiveOnly(bool), onShell(bool), onShapes(bool), onShot(), onBrightness(value),
 * onTrail(seconds), onShapesMissing(text). `shapes` — {available, reason} из /api/meta (D-037).
 * Сохранённые значения применяются сразу, через те же обработчики.
 */
export function createViewPanel(handlers, shapes) {
  const $ = (id) => document.getElementById(id);
  const prefs = {
    activeOnly: false, shell: true, shapes: true, brightness: BRIGHTNESS.start, trail: TRAIL.start, ...loadPrefs(),
  };
  const viewButtons = [...document.querySelectorAll("#view [data-view]")];
  const activeButton = $("view-active");
  const shellButton = $("view-shell");
  const shapesButton = $("view-shapes");
  const brightness = $("view-brightness");
  const trail = $("view-trail");

  function remember(patch) {
    Object.assign(prefs, patch);
    savePrefs(prefs);
  }

  for (const button of viewButtons) button.addEventListener("click", () => handlers.onView(button.dataset.view));

  function setToggle(button, value) {
    button.setAttribute("aria-pressed", String(value));
  }
  activeButton.addEventListener("click", () => {
    const value = activeButton.getAttribute("aria-pressed") !== "true";
    setToggle(activeButton, value);
    remember({ activeOnly: value });
    handlers.onActiveOnly(value);
  });
  shellButton.addEventListener("click", () => {
    const value = shellButton.getAttribute("aria-pressed") !== "true";
    setToggle(shellButton, value);
    remember({ shell: value });
    handlers.onShell(value);
  });
  // Без скачанных скелетов флажок бледный и не включается; щелчок объясняет, как их получить.
  shapesButton.title = shapes.available ? SHAPES_TITLE : `${SHAPES_MISSING} (сейчас: ${shapes.reason}).`;
  shapesButton.setAttribute("aria-disabled", String(!shapes.available));
  shapesButton.addEventListener("click", () => {
    if (!shapes.available) {
      handlers.onShapesMissing(SHAPES_MISSING);
      return;
    }
    const value = shapesButton.getAttribute("aria-pressed") !== "true";
    setToggle(shapesButton, value);
    remember({ shapes: value });
    handlers.onShapes(value);
  });
  $("view-shot").addEventListener("click", () => handlers.onShot());

  function applyBrightness() {
    const value = positionToValue(BRIGHTNESS, Number(brightness.value));
    $("view-brightness-text").textContent = `${value.toLocaleString("ru-RU", { maximumFractionDigits: 1, minimumFractionDigits: 1 })}×`;
    handlers.onBrightness(value);
  }
  function applyTrail() {
    const value = positionToValue(TRAIL, Number(trail.value));
    $("view-trail-text").textContent = formatSeconds(value);
    handlers.onTrail(value);
  }
  brightness.addEventListener("input", () => {
    remember({ brightness: Number(brightness.value) });
    applyBrightness();
  });
  trail.addEventListener("input", () => {
    remember({ trail: Number(trail.value) });
    applyTrail();
  });

  // Сохранённое состояние — в разметку и в обработчики.
  brightness.value = String(prefs.brightness);
  trail.value = String(prefs.trail);
  setToggle(activeButton, prefs.activeOnly);
  setToggle(shellButton, prefs.shell);
  setToggle(shapesButton, shapes.available && prefs.shapes);
  applyBrightness();
  applyTrail();
  handlers.onActiveOnly(prefs.activeOnly);
  handlers.onShell(prefs.shell);
  handlers.onShapes(shapes.available && prefs.shapes);
  $("view").hidden = false;

  return {
    /** Подсветить кнопку ракурса, на который сейчас смотрит камера (или ни одну). */
    setCurrentView(name) {
      for (const button of viewButtons) button.setAttribute("aria-pressed", String(button.dataset.view === name));
    },
  };
}

const NARROW_PX = 760; // окно уже этого: панели закрывали бы облако, поэтому сначала скрыты

/** Кнопка и клавиша H: скрыть или показать панели (остаются облако, контур и компас внизу посередине). */
export function createPanelsToggle() {
  const button = document.getElementById("panels-toggle");
  const set = (hidden) => {
    document.body.classList.toggle("panels-hidden", hidden);
    button.textContent = hidden ? "Показать панели" : "Скрыть панели";
    button.setAttribute("aria-pressed", String(hidden));
  };
  button.addEventListener("click", () => set(!document.body.classList.contains("panels-hidden")));
  window.addEventListener("keydown", (event) => {
    if (event.code !== "KeyH" || event.ctrlKey || event.metaKey || event.altKey) return;
    const typing = document.activeElement?.tagName === "INPUT" && document.activeElement.type !== "range";
    if (!typing) button.click();
  });
  set(window.innerWidth < NARROW_PX);
  button.hidden = false;
}
