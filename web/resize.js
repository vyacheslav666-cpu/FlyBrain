// Панели — плавающие окошки, как карточка нейрона (D-038, D-042, D-043, D-051): за границу — менять размер,
// за рамку у края или за заголовок — двигать, кнопка «−» сворачивает до заголовка. Пока панель не трогали, она стоит в своём углу экрана (style.css).
// Всё запоминается в браузере (localStorage).

import { makeFloating, storedObject } from "./floating.js";

const PANEL_IDS = ["stimuli", "executors", "view", "live"];

/** Содержимое панели — во внутреннюю прокручиваемую обёртку: сама панель не прокручивается, и уголок не уезжает. */
function wrapContent(panel) {
  const body = document.createElement("div");
  body.className = "panel-body";
  body.append(...panel.childNodes);
  panel.append(body);
}

export function createPanelResizers() {
  const stored = storedObject("flybrain.panels");
  // До D-042 хранилась только высота числом.
  for (const [id, value] of Object.entries(stored.value)) if (typeof value === "number") stored.value[id] = { height: value };

  for (const id of PANEL_IDS) {
    const panel = document.getElementById(id);
    wrapContent(panel);
    panel.classList.add("resizable");

    const heading = panel.querySelector("h2"); // у «Вида» первый h2 — «Вид», а не заголовок легенды
    const title = document.createElement("div");
    title.className = "panel-collapsed-title";
    title.textContent = heading.textContent;
    const collapse = document.createElement("button");
    collapse.type = "button";
    collapse.className = "panel-collapse";
    heading.title = "Потяните, чтобы передвинуть панель. Двойной щелчок — обычное место и размер";
    title.title = heading.title;
    panel.prepend(title);
    panel.append(collapse);

    makeFloating(panel, {
      handles: [heading, title], collapseButton: collapse,
      state: (stored.value[id] ??= {}), save: stored.save,
      noun: "панель", minWidth: 220, minHeight: 70,
    });

    if (id === "executors") {
      // Карточка нейрона по умолчанию стоит левее панели исполнителей — сдвигаем её вслед за шириной панели.
      new ResizeObserver(() => {
        if (panel.offsetWidth > 0) document.documentElement.style.setProperty("--executors-width", `${panel.offsetWidth}px`);
      }).observe(panel);
    }
  }
}
