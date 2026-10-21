// Куда отвести камеру, чтобы найденные нейроны были видны, а не прятались под панелями (после C13, D-040).
// Чистая математика без three.js: проверяется на Node (tests/js/framing_check.mjs).

const MARGIN_PX = 28; // отступ от края панели: кольцо поиска — 18 px
const MIN_FREE_PX = 60; // от центра до панели свободно меньше — панели закрыли почти всё; тогда считаем по всему экрану
const FULL_SHARE = 0.9;

/**
 * Свободная от панелей середина экрана в долях половины экрана (как координаты NDC): {x, y}.
 * Точка с |NDC x| ≤ x и |NDC y| ≤ y не закрыта панелями. Область симметрична относительно центра экрана:
 * камера вращается вокруг центра найденных нейронов, и он должен остаться в центре.
 * `rects` — прямоугольники видимых панелей {left, right, top, bottom} в пикселях окна.
 */
export function freeArea(rects, width, height, top = 0, bottom = height) {
  const cx = width / 2;
  const cy = height / 2;
  let halfX = cx;
  for (const rect of rects) {
    if (rect.right <= rect.left || rect.bottom <= rect.top) continue;
    // Панель стоит слева или справа от середины. Панели высокие, поэтому любая из них сужает середину по ширине.
    if ((rect.left + rect.right) / 2 < cx) halfX = Math.min(halfX, cx - rect.right);
    else halfX = Math.min(halfX, rect.left - cx);
  }
  const halfY = Math.min(cy - top, bottom - cy);
  const share = (half, centre) => (half - MARGIN_PX >= MIN_FREE_PX ? Math.min((half - MARGIN_PX) / centre, FULL_SHARE)
    : FULL_SHARE);
  return { x: share(halfX, cx), y: share(halfY, cy) };
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Расстояние от центра `centre` до камеры, при котором все точки `points` ([x, y, z], …) видны внутри `area`.
 * `right`, `up`, `back` — оси камеры (единичные; `back` — от центра к камере), `tanHalfFov` — тангенс половины
 * вертикального угла обзора, `aspect` — ширина / высота окна.
 */
export function fitDistance(points, centre, { right, up, back }, tanHalfFov, aspect, area) {
  let distance = 0;
  for (const point of points) {
    const rel = [point[0] - centre[0], point[1] - centre[1], point[2] - centre[2]];
    // Глубина точки от камеры = distance − rel·back; её проекция не должна выйти за свободную область.
    const need = Math.max(
      Math.abs(dot(rel, right)) / (tanHalfFov * aspect * area.x),
      Math.abs(dot(rel, up)) / (tanHalfFov * area.y),
    );
    distance = Math.max(distance, need + dot(rel, back));
  }
  return distance;
}
