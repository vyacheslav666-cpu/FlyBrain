// Плавающие окошки: карточка нейрона (D-041) и все панели (D-043). Окошко двигают за полосу с точками
// у левого края (D-044) или за заголовок,
// размер меняют за уголок справа внизу, кнопка «−» сворачивает его до заголовка.
// Двойной щелчок по заголовку возвращает обычное место и размер, по уголку — обычный размер.
// Пока окошко не трогали, оно стоит там, где его ставит style.css; первое касание закрепляет его левым верхним углом.

const MARGIN_PX = 8;
const KEEP_VISIBLE_PX = 32; // столько от заголовка всегда остаётся в окне браузера, чтобы окошко можно было вытащить обратно

/** Запомненное в браузере (localStorage) по ключу; если хранилище недоступно, всё работает без запоминания. */
export function storedObject(key) {
  let value = {};
  try {
    value = JSON.parse(localStorage.getItem(key)) ?? {};
  } catch {
    // нет хранилища или в нём мусор — начинаем с пустого
  }
  return {
    value,
    save() {
      try {
        localStorage.setItem(key, JSON.stringify(value));
      } catch {
        // нет места или запрещено — не страшно
      }
    },
  };
}

let raisedRoot = null;
/** Окошко, которого коснулись последним, — поверх остальных. */
function raise(root) {
  raisedRoot?.classList.remove("raised");
  raisedRoot = root;
  root.classList.add("raised");
}

/**
 * Окошко `root`: заголовки `handles` (за них двигают), уголок `grip`, кнопка `collapseButton`.
 * `state` — объект, куда пишутся left/top/width/height/collapsed; `save()` сохраняет его. `noun` — «карточку» или «панель».
 * Возвращает `fit()` — расставить заново (например, когда окошко показали).
 */
export function makeFloating(root, { handles, grip, collapseButton, state, save, noun, minWidth, minHeight }) {
  const has = (key) => Number.isFinite(state[key]);
  // Полоса с точками вдоль левого края — за неё окошко двигают (как за заголовок).
  const strip = document.createElement("div");
  strip.className = "move-strip";
  strip.title = `Потяните, чтобы передвинуть ${noun}. Двойной щелчок — обычное место и размер`;
  root.append(strip);

  // Если окно браузера стало меньше, окошко подвигается и сужается только на экране, а запомненное не меняется:
  // окно снова станет большим — окошко вернётся туда, где было.
  function place() {
    const width = has("width") ? Math.max(minWidth, Math.min(state.width, window.innerWidth - 2 * MARGIN_PX)) : null;
    if (width === null) root.style.removeProperty("width");
    else root.style.width = `${width}px`;
    if (has("left")) {
      const shownWidth = width ?? root.offsetWidth;
      const left = Math.max(MARGIN_PX, Math.min(state.left, window.innerWidth - shownWidth - MARGIN_PX));
      const top = Math.max(MARGIN_PX, Math.min(state.top, window.innerHeight - KEEP_VISIBLE_PX - MARGIN_PX));
      root.style.left = `${left}px`;
      root.style.top = `${top}px`;
      root.style.right = "auto";
      root.style.bottom = "auto";
      root.style.maxHeight = `${window.innerHeight - top - MARGIN_PX}px`;
    } else {
      for (const name of ["left", "top", "right", "bottom", "max-height"]) root.style.removeProperty(name);
    }
    if (has("height")) root.style.height = `${Math.max(minHeight, state.height)}px`;
    else root.style.removeProperty("height");
    root.classList.toggle("sized", has("height"));
    root.classList.toggle("collapsed", Boolean(state.collapsed));
    collapseButton.textContent = state.collapsed ? "+" : "−";
    collapseButton.title = state.collapsed ? `развернуть ${noun}` : `свернуть ${noun} (останется только заголовок)`;
  }

  /** Перед тем как тянуть, закрепляем место левым верхним углом: тогда уголок справа внизу тянет вправо и вниз. */
  function anchor() {
    const rect = root.getBoundingClientRect();
    state.left = rect.left;
    state.top = rect.top;
  }

  function track(handle, cursorClass, onMove) {
    let start = null;
    handle.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest("button")) return;
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      anchor();
      const rect = root.getBoundingClientRect();
      start = { x: event.clientX, y: event.clientY, left: state.left, top: state.top, width: rect.width, height: rect.height };
      document.body.classList.add(cursorClass);
      handle.classList.add("active");
    });
    handle.addEventListener("pointermove", (event) => {
      if (!start) return;
      onMove(event.clientX - start.x, event.clientY - start.y, start);
      place();
    });
    const finish = () => {
      if (!start) return;
      start = null;
      document.body.classList.remove(cursorClass);
      handle.classList.remove("active");
      save();
    };
    handle.addEventListener("pointerup", finish);
    handle.addEventListener("pointercancel", finish);
  }

  for (const handle of [strip, ...handles]) {
    handle.classList.add("drag-handle");
    track(handle, "moving", (dx, dy, start) => {
      state.left = start.left + dx;
      state.top = start.top + dy;
    });
    handle.addEventListener("dblclick", (event) => {
      if (event.target.closest("button")) return;
      for (const key of ["left", "top", "width", "height"]) delete state[key];
      save();
      place();
    });
  }
  track(grip, "sizing", (dx, dy, start) => {
    state.width = Math.max(minWidth, start.width + dx);
    state.height = Math.max(minHeight, Math.min(start.height + dy, window.innerHeight - start.top - MARGIN_PX));
  });
  grip.addEventListener("dblclick", () => {
    delete state.width;
    delete state.height;
    save();
    place();
  });
  collapseButton.addEventListener("click", () => {
    state.collapsed = !state.collapsed;
    save();
    place();
  });
  root.addEventListener("pointerdown", () => raise(root), true);
  window.addEventListener("resize", () => place());
  place();

  return { fit: place };
}
