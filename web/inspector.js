// Подсказка, карточка закреплённого нейрона и строка поиска (коммит C8). Данные берутся у сервера: /api/neuron, /api/search.
// Строку «Частота сейчас» заполняет main.js по потоку живого мозга (C9).
// Карточку можно двигать, сворачивать и менять её размер (floating.js, D-041).

import { makeFloating, storedObject } from "./floating.js";

const infoCache = new Map();
const linksCache = new Map();

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`сервер ответил ${response.status} на ${url}`);
  return response.json();
}

export async function getInfo(index) {
  if (!infoCache.has(index)) infoCache.set(index, await getJson(`/api/neuron/${index}`));
  return infoCache.get(index);
}

export async function getLinks(index) {
  if (!linksCache.has(index)) linksCache.set(index, await getJson(`/api/neuron/${index}/links`));
  return linksCache.get(index);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function plural(count, one, few, many) {
  const lastTwo = count % 100;
  const last = count % 10;
  if (lastTwo >= 11 && lastTwo <= 14) return many;
  if (last === 1) return one;
  if (last >= 2 && last <= 4) return few;
  return many;
}

const formatInt = (number) => number.toLocaleString("ru-RU");

export const RATE_HINT = "спайки этого нейрона за последнюю 1 с модельного времени (живой режим)";

/** Строки одного нейрона: имя, класс, сторона, медиатор, знак в модели, номер FlyWire, текущая частота. */
function describe(info) {
  const box = document.createDocumentFragment();
  const head = element("div", "neuron-head");
  head.append(element("strong", "neuron-label", info.label));
  if (info.about) head.append(element("span", "neuron-about", info.about));
  box.append(head);

  const rows = element("dl", "neuron-rows");
  const add = (name, value, className) => {
    rows.append(element("dt", "", name), element("dd", className, value));
  };
  if (info.cell_type && info.cell_type !== info.label) add("Тип FlyWire", info.cell_type, "mono");
  add("Класс", info.class_name);
  add("Сторона у мухи", info.side_name);
  add("Медиатор (по таблице)", info.transmitter_name);
  add("Знак в модели", info.sign_name, info.sign > 0 ? "sign-plus" : info.sign < 0 ? "sign-minus" : "");
  add("Номер FlyWire", info.flywire_id, "mono");
  add("Частота сейчас", "—", "live-rate");
  rows.lastElementChild.previousElementSibling.title = RATE_HINT;
  box.append(rows);
  return box;
}

export function createTooltip(root) {
  return {
    show(info, x, y) {
      root.replaceChildren(describe(info));
      root.hidden = false;
      this.moveTo(x, y);
    },
    moveTo(x, y) {
      const margin = 14;
      const width = root.offsetWidth;
      const height = root.offsetHeight;
      const left = x + 18 + width + margin > window.innerWidth ? x - 18 - width : x + 18;
      const top = Math.min(y + 18, window.innerHeight - height - margin);
      root.style.left = `${Math.max(margin, left)}px`;
      root.style.top = `${Math.max(margin, top)}px`;
    },
    hide() {
      root.hidden = true;
    },
  };
}

/** Карточка закреплённого нейрона: описание, цвета линий и самые сильные соседи. `onSelect(index)` — щелчок по соседу. */
export function createCard(root, { onClose, onSelect }) {
  function partnerList(title, className, partners, total) {
    const section = element("section", `partners ${className}`);
    section.append(element("h3", "", `${title} (всего ${formatInt(total)})`));
    const list = element("ol");
    for (const partner of partners.slice(0, 5)) {
      const item = element("li");
      const label = element(partner.visible ? "button" : "span", "partner-label", partner.label);
      if (partner.visible) label.addEventListener("click", () => onSelect(partner.index));
      else label.title = "у этого нейрона нет координат, его нельзя показать";
      const strength = element("span", "partner-strength", `${partner.sign < 0 ? "−" : "+"}${formatInt(partner.synapses)}`);
      strength.title = `${formatInt(partner.synapses)} синапсов; ${partner.sign < 0 ? "тормозящая" : "возбуждающая"} связь в модели`;
      item.append(label, strength);
      list.append(item);
    }
    section.append(list);
    return section;
  }

  // Заголовок (за него карточку двигают), прокручиваемое содержимое и уголок для размера.
  const bar = element("div", "card-bar");
  bar.title = "Потяните, чтобы передвинуть карточку. Двойной щелчок — обычное место и размер";
  const title = element("strong", "card-title");
  const collapse = element("button", "card-button card-collapse");
  const close = element("button", "card-button card-close", "×");
  close.title = "снять выбор (Esc)";
  close.addEventListener("click", onClose);
  bar.append(title, collapse, close);
  const body = element("div", "card-body");
  root.replaceChildren(bar, body);
  const stored = storedObject("flybrain.card");
  const floating = makeFloating(root, {
    handles: [bar], collapseButton: collapse, state: stored.value, save: stored.save,
    noun: "карточку", minWidth: 260, minHeight: 140,
  });

  return {
    showInfo(info) {
      title.textContent = info.label;
      body.replaceChildren(describe(info), element("p", "card-loading", "Загружаю связи…"));
      root.hidden = false;
      floating.fit();
      this.info = info;
    },
    showLinks(links) {
      root.querySelector(".card-loading")?.remove();
      const hint = element("p", "card-hint",
        "На рисунке: синие линии — самые сильные входы (кто передаёт этому нейрону), " +
        "оранжевые — самые сильные выходы (кому он передаёт). Не больше 20 с каждой стороны.");
      body.append(
        hint,
        partnerList("Получает от", "partners-in", links.inputs, links.inputs_total),
        partnerList("Передаёт", "partners-out", links.outputs, links.outputs_total),
        element("p", "card-hint",
          "Число — синапсы; «+» возбуждающая связь, «−» тормозящая, как в модели Shiu. " +
          "Знак берётся из файла модели, а медиатор — из нынешней таблицы FlyWire, поэтому они иногда расходятся (D-025)."),
      );
    },
    showError(message) {
      root.querySelector(".card-loading")?.remove();
      body.append(element("p", "card-hint", `Не получилось загрузить связи: ${message}`));
    },
    /** Строка о форме нейрона (C12b) под описанием; `text` пустой — строки нет. */
    showShape(text, title = "") {
      let line = root.querySelector(".card-shape");
      if (!text) {
        line?.remove();
        return;
      }
      if (!line) {
        line = element("p", "card-shape");
        root.querySelector(".neuron-rows")?.after(line);
      }
      line.textContent = text;
      line.title = title;
    },
    hide() {
      root.hidden = true;
    },
  };
}

/** Строка поиска: ищет через 0,25 с после последней буквы или сразу по Enter. */
export function createSearch({ input, status, clearButton, onResult, onClear }) {
  let timer = null;
  let ticket = 0;

  function describeResult(result) {
    if (result.total === 0) return "Ничего не найдено. Попробуйте «MN9», «LPLC2», «GF» или номер FlyWire.";
    const nouns = plural(result.total, "нейрон", "нейрона", "нейронов");
    let text = result.types.length === 1
      ? `${result.label}: ${formatInt(result.total)} ${nouns}`
      : `Найдено ${formatInt(result.total)} ${nouns}: ` +
        result.types.map((type) => `${type.label} (${formatInt(type.count)})`).join(", ");
    if (result.shown < result.total) text += `; на рисунке ${formatInt(result.shown)} (у остальных нет координат)`;
    if (result.truncated) text += `; кольцами отмечены первые ${formatInt(result.indices.length)}`;
    return text;
  }

  async function run() {
    const query = input.value.trim();
    const mine = ++ticket;
    if (query.length < 2) {
      status.textContent = query ? "Введите хотя бы два символа." : "";
      clearButton.hidden = query.length === 0;
      onClear();
      return;
    }
    clearButton.hidden = false;
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error(`сервер ответил ${response.status}`);
      const result = await response.json();
      if (mine !== ticket) return; // пока ждали, текст изменился
      status.textContent = describeResult(result);
      if (result.indices.length > 0) onResult(result);
      else onClear();
    } catch (error) {
      status.textContent = `Поиск не удался: ${error.message}`;
    }
  }

  input.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(run, 250);
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      clearTimeout(timer);
      run();
    } else if (event.key === "Escape") {
      input.blur();
    }
  });
  clearButton.addEventListener("click", () => {
    input.value = "";
    run();
  });
  return { run };
}
