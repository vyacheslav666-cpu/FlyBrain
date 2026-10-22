// Плавающие окошки: карточка нейрона (D-041) и все панели (D-043). Управление — как у окна Windows (D-051):
// курсор на самой границе окошка — тянуть, чтобы изменить размер (любая сторона или угол);
// курсор внутри, у края (в рамке вокруг содержимого) — тянуть, чтобы передвинуть. За заголовок тоже можно двигать.
// Двойной щелчок в рамке — обычное место и размер, на границе — только обычный размер. Кнопка «−» сворачивает до заголовка.
// Пока окошко не трогали, оно стоит там, где его ставит style.css; первое касание закрепляет его левым верхним углом.

const MARGIN_PX = 8;
const KEEP_VISIBLE_PX = 32; // столько от заголовка всегда остаётся в окне браузера, чтобы окошко можно было вытащить обратно
const EDGE_PX = 6; // ширина границы, за которую меняют размер; дальше внутрь до содержимого — рамка для перетаскивания

const CURSORS = { n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize", ne: "nesw-resize", sw: "nesw-resize",
  nw: "nwse-resize", se: "nwse-resize", move: "move" };

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
 * Окошко `root`: заголовки `handles` (за них тоже двигают), кнопка `collapseButton`.
 * `state` — объект, куда пишутся left/top/width/height/collapsed; `save()` сохраняет его. `noun` — «карточку» или «панель».
 * Возвращает `fit()` — расставить заново (например, когда окошко показали).
 */
export function makeFloating(root, { handles, collapseButton, state, save, noun, minWidth, minHeight }) {
  const has = (key) => Number.isFinite(state[key]);
  root.classList.add("floating");
  for (const handle of handles) handle.classList.add("drag-handle");

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

  /**
   * Что сделает нажатие в этой точке: «n», «se» и т. п. — менять размер за эту сторону или угол; «move» — двигать;
   * null — ничего (курсор над содержимым, кнопкой или ползунком). Рамка — это само окошко вокруг содержимого.
   */
  function zoneAt(event) {
    const onHandle = handles.some((handle) => handle.contains(event.target)) && !event.target.closest("button");
    if (event.target !== root && !onHandle) return null;
    const rect = root.getBoundingClientRect();
    const vertical = state.collapsed ? "" // у свёрнутого окошка высота сама по себе
      : event.clientY - rect.top < EDGE_PX ? "n" : rect.bottom - event.clientY < EDGE_PX ? "s" : "";
    const horizontal = event.clientX - rect.left < EDGE_PX ? "w" : rect.right - event.clientX < EDGE_PX ? "e" : "";
    return vertical + horizontal || "move";
  }

  // Курсор и подсветка рамки подсказывают, что сейчас будет: стрелки — размер, крестик — перетаскивание.
  let drag = null;
  root.addEventListener("pointermove", (event) => {
    if (drag) return;
    const zone = zoneAt(event);
    root.style.cursor = zone ? CURSORS[zone] : "";
    root.classList.toggle("frame-hot", zone !== null && zone !== "move");
  });
  root.addEventListener("pointerleave", () => {
    if (drag) return;
    root.style.cursor = "";
    root.classList.remove("frame-hot");
  });

  root.addEventListener("pointerdown", (event) => {
    raise(root);
    if (event.button !== 0) return;
    const zone = zoneAt(event);
    if (!zone) return;
    event.preventDefault();
    root.setPointerCapture(event.pointerId);
    const rect = root.getBoundingClientRect();
    // Закрепляем место левым верхним углом: дальше и движение, и размер считаются от него.
    state.left = rect.left;
    state.top = rect.top;
    drag = { zone, x: event.clientX, y: event.clientY, left: rect.left, top: rect.top, width: rect.width, height: rect.height };
    document.body.style.cursor = CURSORS[zone];
    document.body.classList.add("dragging-window");
  });

  root.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    const { zone } = drag;
    if (zone === "move") {
      state.left = drag.left + dx;
      state.top = drag.top + dy;
    } else {
      if (zone.includes("e")) state.width = Math.max(minWidth, drag.width + dx);
      if (zone.includes("w")) {
        state.width = Math.max(minWidth, drag.width - dx);
        state.left = drag.left + drag.width - state.width; // правый край стоит на месте
      }
      if (zone.includes("s")) {
        state.height = Math.max(minHeight, Math.min(drag.height + dy, window.innerHeight - drag.top - MARGIN_PX));
      }
      if (zone.includes("n")) {
        const bottom = drag.top + drag.height; // нижний край стоит на месте
        state.height = Math.max(minHeight, Math.min(drag.height - dy, bottom - MARGIN_PX));
        state.top = bottom - state.height;
      }
    }
    place();
  });

  const finish = () => {
    if (!drag) return;
    drag = null;
    document.body.style.removeProperty("cursor");
    document.body.classList.remove("dragging-window");
    save();
  };
  root.addEventListener("pointerup", finish);
  root.addEventListener("pointercancel", finish);

  root.addEventListener("dblclick", (event) => {
    const zone = zoneAt(event);
    if (!zone) return;
    const keys = zone === "move" ? ["left", "top", "width", "height"] : ["width", "height"];
    for (const key of keys) delete state[key];
    save();
    place();
  });

  collapseButton.addEventListener("click", () => {
    state.collapsed = !state.collapsed;
    save();
    place();
  });
  window.addEventListener("resize", () => place());
  place();

  return { fit: place };
}
