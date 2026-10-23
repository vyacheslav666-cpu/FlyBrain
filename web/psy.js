// Панель «Психоделик (эксперимент)» (P2, D-064…D-066): доза, режим, шум мембраны, индикатор «сеть зажглась»,
// слой «серотониновые нейроны и их мишени». [ЧЕСТНОСТЬ] Вычислительная гипотеза P1/P1b, не ощущения мухи.
// Настройки и данные слоя — /api/psy; состояние — в сообщениях «status» (поле psy), показания индикатора —
// в сообщениях «psy» (flybrain/psylive.py, IgnitionMeter).

const SEND_EVERY_MS = 100; // ползунки шлют команды не чаще 10 раз в секунду, последнее значение не теряется
const IGNITED_TEXT = "Сеть в самоподдерживающемся состоянии. Сама не погаснет, даже при дозе 0 — нажмите «Сброс». " +
  "В живой мухе так не бывает: в модели нет адаптации нейронов (P1b).";

function format(number, digits) {
  return number.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/** Отправка значения ползунка не чаще SEND_EVERY_MS, но последнее — всегда. */
function throttled(send) {
  let pending = null;
  let lastSent = 0;
  let timer = null;
  function flush() {
    timer = null;
    if (pending === null) return;
    send(pending);
    pending = null;
    lastSent = performance.now();
  }
  return (value) => {
    pending = value;
    const wait = SEND_EVERY_MS - (performance.now() - lastSent);
    if (wait <= 0) flush();
    else if (timer === null) timer = setTimeout(flush, wait);
  };
}

/**
 * `info` — ответ /api/psy; `onCommand(command)` — команда серверу; `layer` — слой в 3D (psylayer.js).
 * Возвращает setStatus, setConnected, setReading, `noiseOn` и `ignited` (тогда плашка «в покое мозг молчит» не нужна).
 */
export function createPsyPanel(info, onCommand, layer) {
  const $ = (id) => document.getElementById(id);
  const root = $("psy");
  const dose = $("psy-dose");
  const sigma = $("psy-sigma");
  const modeButtons = [...root.querySelectorAll("[data-psy-mode]")];
  const layerButton = $("psy-layer");
  const resetButton = $("live-reset");
  let status = null;
  let connected = false;
  let reading = null;

  if (!info.available) {
    $("psy-unavailable").textContent = `${info.reason}. Нужны данные P1: uv run flybrain fetch-data и build-data.`;
    $("psy-controls").hidden = true;
    root.hidden = false;
    return { setStatus() {}, setConnected() {}, setReading() {}, get noiseOn() { return false; }, get ignited() { return false; } };
  }

  dose.max = String(info.dose_max);
  dose.step = String(info.dose_step);
  sigma.max = String(info.sigma_max);
  sigma.step = String(info.sigma_step);
  $("psy-ignited").textContent = IGNITED_TEXT;
  $("psy-sigma-hint").textContent =`При σ ≥ ${format(info.sigma_ignites, 2)} мВ сеть зажигается и без дозы (P1b).`;
  $("psy-layer-legend").textContent = `${info.serotonin.length} серотониновых нейронов (розовые) и ` +
    `${info.targets.length.toLocaleString("ru-RU")} их мишеней (жёлтые: ярче — больше серотониновых синапсов).`;

  const sendDose = throttled((value) => onCommand({ cmd: "psy", dose: value }));
  const sendSigma = throttled((value) => onCommand({ cmd: "psy", sigma: value }));
  dose.addEventListener("input", () => { showDose(Number(dose.value)); sendDose(Number(dose.value)); });
  sigma.addEventListener("input", () => { showSigma(Number(sigma.value)); sendSigma(Number(sigma.value)); });
  for (const button of modeButtons) {
    button.addEventListener("click", () => onCommand({ cmd: "psy", mode: button.dataset.psyMode }));
  }
  layerButton.addEventListener("click", () => {
    const on = layerButton.getAttribute("aria-pressed") !== "true";
    layerButton.setAttribute("aria-pressed", String(on));
    layer.setVisible(on);
    $("psy-layer-legend").hidden = !on;
  });

  function showDose(value) {
    const median = value * info.median_mv;
    const top = value * info.max_mv;
    $("psy-dose-text").textContent = `доза ${format(value, 2)}`;
    $("psy-dose-mv").textContent = value > 0
      ? `Порог ниже на ${format(median, 2)} мВ у типичной мишени и на ${format(top, 2)} мВ у самой «нагруженной» ` +
        `(зазор «покой — порог» — ${format(info.gap_mv, 0)} мВ).`
      : "Доза 0: модель Shiu без изменений.";
  }

  function showSigma(value) {
    $("psy-sigma-text").textContent = `σ = ${format(value, 2)} мВ`;
  }

  function refresh() {
    const psy = status?.psy;
    const ready = connected && status && (status.state === "running" || status.state === "paused") && psy?.available;
    dose.disabled = sigma.disabled = !ready;
    for (const button of modeButtons) {
      button.disabled = !ready;
      button.setAttribute("aria-pressed", String(psy?.mode === button.dataset.psyMode));
    }
    if (psy) {
      // Пока ползунок тянут, не двигаем его назад значением, которое сервер ещё не получил.
      if (document.activeElement !== dose) dose.value = String(psy.dose);
      if (document.activeElement !== sigma) sigma.value = String(psy.sigma);
    }
    showDose(Number(dose.value));
    showSigma(Number(sigma.value));
    const ignited = Boolean(psy?.ignited);
    root.classList.toggle("ignited", ignited);
    resetButton.classList.toggle("attention", ignited);
    $("psy-lamp").textContent = ignited ? "🔥 сеть зажглась" : "сеть не горит";
    $("psy-ignited").hidden = !ignited;
    const collapsedTitle = root.querySelector(".panel-collapsed-title");
    if (collapsedTitle) collapsedTitle.textContent = `Психоделик (эксперимент)${ignited ? " — 🔥 зажглась" : ""}`;
    if (reading && ready) {
      $("psy-meter").textContent = `Во всём мозге ${Math.round(reading.rate_hz / 1000).toLocaleString("ru-RU")} тыс. ` +
        `спайков/с за последние ${format(Math.min(reading.span_ms, info.window_ms), 0)} мс (зажглась — больше ` +
        `${format(info.ignited_hz / 1000, 0)} тыс.). Петля усиковой доли: ${reading.al_active} из ${reading.al_total} ` +
        "возбуждающих локальных нейронов стреляли.";
    } else {
      $("psy-meter").textContent = "";
    }
  }

  root.hidden = false;
  refresh();
  return {
    setStatus(next) { status = next; refresh(); },
    setConnected(value) { connected = value; refresh(); },
    setReading(next) { reading = next; refresh(); },
    get noiseOn() { return (status?.psy?.sigma ?? 0) > 0; },
    get ignited() { return Boolean(status?.psy?.ignited); },
  };
}
