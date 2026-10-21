// Панель «Вид» (C12): ракурсы, «только активные», контур мозга, снимок, яркость и шлейф.
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

/**
 * `handlers`: onView(name), onActiveOnly(bool), onShell(bool), onShot(), onBrightness(value), onTrail(seconds).
 * Сохранённые значения применяются сразу, через те же обработчики.
 */
export function createViewPanel(handlers) {
  const $ = (id) => document.getElementById(id);
  const prefs = { activeOnly: false, shell: true, brightness: BRIGHTNESS.start, trail: TRAIL.start, ...loadPrefs() };
  const viewButtons = [...document.querySelectorAll("#view [data-view]")];
  const activeButton = $("view-active");
  const shellButton = $("view-shell");
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
  applyBrightness();
  applyTrail();
  handlers.onActiveOnly(prefs.activeOnly);
  handlers.onShell(prefs.shell);
  $("view").hidden = false;

  return {
    /** Подсветить кнопку ракурса, на который сейчас смотрит камера (или ни одну). */
    setCurrentView(name) {
      for (const button of viewButtons) button.setAttribute("aria-pressed", String(button.dataset.view === name));
    },
  };
}

const NARROW_PX = 760; // окно уже этого: панели закрывали бы облако, поэтому сначала скрыты

/** Кнопка и клавиша H: скрыть или показать панели (остаётся только облако, контур и компас). */
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
