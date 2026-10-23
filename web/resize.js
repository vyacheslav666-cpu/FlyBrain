// Панели — плавающие окошки, как карточка нейрона (D-038, D-042, D-043, D-051): за границу — менять размер,
// за рамку у края или за заголовок — двигать, кнопка «−» сворачивает до заголовка. Пока панель не трогали, она стоит в своём углу экрана (style.css).
// Место и размер не запоминаются: после перезагрузки панели снова в своих углах (D-052).

import { forgetSavedLayout, makeFloating } from "./floating.js";

const PANEL_IDS = ["stimuli", "executors", "view", "live", "psy"];
const COLLAPSED_AT_START = new Set(["psy"]); // эксперимент: при запуске свёрнут до заголовка, чтобы не закрывать мозг

/** Содержимое панели — во внутреннюю прокручиваемую обёртку: сама панель не прокручивается, и уголок не уезжает. */
function wrapContent(panel) {
  const body = document.createElement("div");
  body.className = "panel-body";
  body.append(...panel.childNodes);
  panel.append(body);
}

export function createPanelResizers() {
  forgetSavedLayout("flybrain.panels");

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
      handles: [heading, title], collapseButton: collapse, noun: "панель", minWidth: 220, minHeight: 70,
      collapsed: COLLAPSED_AT_START.has(id),
    });

    if (id === "executors") {
      // Карточка нейрона по умолчанию стоит левее панели исполнителей — сдвигаем её вслед за шириной панели.
      new ResizeObserver(() => {
        if (panel.offsetWidth > 0) document.documentElement.style.setProperty("--executors-width", `${panel.offsetWidth}px`);
      }).observe(panel);
    }
  }
}
