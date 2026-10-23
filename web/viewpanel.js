// Панель «Вид» (C12): вид по умолчанию, «только активные», контур мозга,
// яркость активных и неактивных нейронов (D-048) и шлейф.
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
 * `handlers`: onView(name), onActiveOnly(bool), onShell(bool), onActiveBrightness(value), onIdleBrightness(value),
 * onTrail(seconds).
 * Сохранённые значения применяются сразу, через те же обработчики.
 */
export function createViewPanel(handlers) {
  const $ = (id) => document.getElementById(id);
  const prefs = {
    activeOnly: false, shell: true, trail: TRAIL.start, ...loadPrefs(),
  };
  // До D-048 яркость была одна — она становится начальной для обеих.
  prefs.brightActive ??= prefs.brightness ?? BRIGHTNESS.start;
  prefs.brightIdle ??= prefs.brightness ?? BRIGHTNESS.start;
  delete prefs.brightness;
  delete prefs.shapes; // флажок «Формы нейронов» убран (D-067)
  const viewButtons = [...document.querySelectorAll("#view [data-view]")];
  const activeButton = $("view-active");
  const shellButton = $("view-shell");
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

  /** Ползунок яркости: `id` — его поле, `key` — имя в сохранённых настройках, `handler` — куда отдать значение. */
  function brightnessSlider(id, key, handler) {
    const input = $(id);
    const apply = () => {
      const value = positionToValue(BRIGHTNESS, Number(input.value));
      $(`${id}-text`).textContent = `${value.toLocaleString("ru-RU", { maximumFractionDigits: 1, minimumFractionDigits: 1 })}×`;
      handler(value);
    };
    input.addEventListener("input", () => {
      remember({ [key]: Number(input.value) });
      apply();
    });
    input.value = String(prefs[key]);
    apply();
  }
  function applyTrail() {
    const value = positionToValue(TRAIL, Number(trail.value));
    $("view-trail-text").textContent = formatSeconds(value);
    handlers.onTrail(value);
  }
  trail.addEventListener("input", () => {
    remember({ trail: Number(trail.value) });
    applyTrail();
  });

  // Сохранённое состояние — в разметку и в обработчики.
  trail.value = String(prefs.trail);
  setToggle(activeButton, prefs.activeOnly);
  setToggle(shellButton, prefs.shell);
  brightnessSlider("view-bright-active", "brightActive", handlers.onActiveBrightness);
  brightnessSlider("view-bright-idle", "brightIdle", handlers.onIdleBrightness);
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
