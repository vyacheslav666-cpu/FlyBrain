// Карточку нейрона можно двигать за заголовок, сворачивать и менять её размер за уголок справа внизу (D-041).
// Двойной щелчок по заголовку возвращает обычное место и размер. Всё запоминается в браузере (localStorage).

const STORAGE_KEY = "flybrain.card";
const MARGIN_PX = 8;
const MIN_WIDTH_PX = 260;
const MIN_HEIGHT_PX = 140;

function load() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) ?? {};
  } catch {
    return {};
  }
}

function save(state) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {
    // нет места или запрещено — не страшно
  }
}

/** Окошко `root` с заголовком `bar`, уголком `grip` и кнопкой `collapseButton`. `fit()` — подогнать под окно браузера. */
export function makeFloating(root, { bar, grip, collapseButton }) {
  const state = load();
  const has = (key) => Number.isFinite(state[key]);

  // Если окно браузера стало меньше, карточка подвигается и сужается только на экране, а запомненное не меняется:
  // окно снова станет большим — карточка вернётся туда, где была.
  function place() {
    const width = has("width")
      ? Math.max(MIN_WIDTH_PX, Math.min(state.width, window.innerWidth - 2 * MARGIN_PX)) : null;
    if (width === null) root.style.removeProperty("width");
    else root.style.width = `${width}px`;
    if (has("left")) {
      // Заголовок всегда остаётся в окне браузера, чтобы карточку можно было вытащить обратно.
      const shownWidth = width ?? root.offsetWidth;
      const left = Math.max(MARGIN_PX, Math.min(state.left, window.innerWidth - shownWidth - MARGIN_PX));
      const top = Math.max(MARGIN_PX, Math.min(state.top, window.innerHeight - bar.offsetHeight - MARGIN_PX));
      root.style.left = `${left}px`;
      root.style.top = `${top}px`;
      root.style.right = "auto";
      root.style.maxHeight = `${window.innerHeight - top - MARGIN_PX}px`;
    } else {
      for (const name of ["left", "top", "right", "max-height"]) root.style.removeProperty(name);
    }
    if (has("height")) root.style.height = `${Math.max(MIN_HEIGHT_PX, state.height)}px`;
    else root.style.removeProperty("height");
    root.classList.toggle("collapsed", Boolean(state.collapsed));
    collapseButton.textContent = state.collapsed ? "+" : "−";
    collapseButton.title = state.collapsed ? "развернуть карточку" : "свернуть карточку (останется только заголовок)";
  }

  /** Перед тем как тянуть, закрепляем место левым верхним углом: тогда уголок справа внизу тянет вправо и вниз. */
  function anchor() {
    const rect = root.getBoundingClientRect();
    state.left = rect.left;
    state.top = rect.top;
  }

  function track(handle, onMove) {
    let start = null;
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest("button")) return;
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      anchor();
      const rect = root.getBoundingClientRect();
      start = { x: event.clientX, y: event.clientY, left: state.left, top: state.top, width: rect.width, height: rect.height };
      document.body.classList.add(handle === grip ? "card-sizing" : "card-moving");
    });
    handle.addEventListener("pointermove", (event) => {
      if (!start) return;
      onMove(event.clientX - start.x, event.clientY - start.y, start);
      place();
    });
    const finish = () => {
      if (!start) return;
      start = null;
      document.body.classList.remove("card-sizing", "card-moving");
      save(state);
    };
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
  }

  track(bar, (dx, dy, start) => {
    state.left = start.left + dx;
    state.top = start.top + dy;
  });
  track(grip, (dx, dy, start) => {
    state.width = Math.max(MIN_WIDTH_PX, start.width + dx);
    state.height = Math.max(MIN_HEIGHT_PX, Math.min(start.height + dy, window.innerHeight - start.top - MARGIN_PX));
  });

  bar.addEventListener("dblclick", (event) => {
    if (event.target.closest("button")) return;
    for (const key of Object.keys(state)) delete state[key];
    save(state);
    place();
  });
  collapseButton.addEventListener("click", () => {
    state.collapsed = !state.collapsed;
    save(state);
    place();
  });
  window.addEventListener("resize", () => place());
  place();

  return { fit: place };
}
