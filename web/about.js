// Окно «О проекте» (C13, D-039): авторы данных, упрощения модели, источники и итоги проверки.
// Итоги берутся из docs/validation/summary.json, который пишет `flybrain validate`, — те же отметки, что в отчёте.

const PASS = "pass";
const INFO = "info";

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function markCell(summary, state) {
  const { mark, word } = summary.marks[state];
  return element("td", `about-mark about-${state}`, `${mark} ${word}`);
}

function renderValidation(target, summary) {
  target.replaceChildren();
  const { passed, total } = summary.a;
  target.append(element("p", "", passed === total
    ? `Наш движок против оригинального кода Shiu (Brian2), пункт А: все три порога выполнены в ${passed} условиях из ${total}.`
    : `Наш движок против оригинального кода Shiu (Brian2), пункт А: пороги выполнены в ${passed} условиях из ${total}; ` +
      "что не сошлось — в отчёте."));

  const table = element("table", "about-table");
  const head = table.createTHead().insertRow();
  for (const title of ["", "Вывод статьи", "Brian2", "Движок"]) head.append(element("th", "", title));
  const body = table.createTBody();
  for (const item of summary.items) {
    const row = body.insertRow();
    row.append(element("td", "about-key", item.key));
    const title = element("td", "", item.title);
    if (item.added_after_reference) title.append(element("span", "about-muted", " — добавлен после эталона"));
    row.append(title, markCell(summary, item.reference), markCell(summary, item.engine));
  }
  target.append(table);

  const agreed = summary.items.filter((item) => item.reference === PASS && item.engine === PASS);
  const other = summary.items.filter((item) => item.reference !== INFO && !(item.reference === PASS && item.engine === PASS));
  const info = summary.items.filter((item) => item.reference === INFO);
  const list = element("ul", "about-verdicts");
  if (agreed.length) {
    const line = element("li");
    line.append(element("b", "", "Сошлось: "), `${agreed.map((item) => item.key).join(", ")} — и в оригинале, и в движке.`);
    list.append(line);
  }
  for (const item of other) {
    const line = element("li");
    // «На границе» — не провал, но и не выполнено: так и называем.
    const doubtful = ![item.reference, item.engine].includes("fail");
    line.append(element("b", "", `${doubtful ? "Под вопросом" : "Не сошлось"} — ${item.key}: `), item.words);
    list.append(line);
  }
  for (const item of info) {
    const line = element("li");
    line.append(element("b", "", `${item.key}, эксперимент вне статьи: `), item.words);
    list.append(line);
  }
  target.append(list);
  target.append(element("p", "about-muted",
    "✅ да · ❌ нет · ⚠️ на границе: число ближе к порогу, чем две погрешности, и не засчитывается. " +
    "Пороги записаны до написания кода (D-018), модель не подгонялась."));
}

/** Окно `dialogId` открывается кнопкой `buttonId`; закрывается «×», Esc и щелчком по затемнению вокруг. */
export function setupDialog(dialogId, buttonId, onOpen) {
  const dialog = document.getElementById(dialogId);
  document.getElementById(buttonId).addEventListener("click", () => {
    onOpen?.();
    dialog.showModal();
  });
  dialog.querySelector(".about-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => { if (event.target === dialog) dialog.close(); });
  // Клавиши H, R и F, пока окно открыто, не должны прятать панели, двигать камеру и включать счётчик кадров позади него.
  dialog.addEventListener("keydown", (event) => { if (event.key !== "Escape") event.stopPropagation(); });
}

export function createAbout() {
  const target = document.getElementById("about-validation");
  let loaded = false;

  async function load() {
    try {
      const response = await fetch("/api/validation", { cache: "no-store" });
      if (!response.ok) {
        const answer = await response.json().catch(() => ({}));
        throw new Error(answer.detail || "сервер не отдал итоги проверки");
      }
      renderValidation(target, await response.json());
      loaded = true;
    } catch (error) {
      target.replaceChildren(element("p", "about-muted", `Итоги проверки не загрузились: ${error.message}.`));
    }
  }

  setupDialog("about", "about-open", () => { if (!loaded) load(); });
  // Окно «Управление» (D-049): всё, что раньше было мелким текстом внизу экрана.
  setupDialog("controls", "controls-open");
}
