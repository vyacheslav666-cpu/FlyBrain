// Силуэт мозга, построенный по самим точкам нейронов (C12, D-036). Внешних файлов нет.
// Без three.js: модуль считает только числа, поэтому его проверяет тест на Node (tests/js/shell_check.mjs).
//
// Как строится: точки раскладываются по кубикам-вокселям и размываются, так получается «плотность клеток».
// Где она выше порога — внутри. Пустоты внутри мозга (там нейропиль, тел клеток мало) заливаются:
// «снаружи» — всё, до чего можно дойти от края сетки, не заходя внутрь. Дальше остаётся самый большой
// кусок, граница слегка сглаживается и превращается в сетку из треугольников (surface nets).

export const SHELL = {
  voxelUm: 7, // сторона кубика, мкм
  blurRadius: 2, // размытие плотности: окно (2·r+1) кубиков, несколько проходов подряд
  blurPasses: 3,
  minDensity: 0.35, // «внутри», если в среднем есть хотя бы столько точек на кубик
  smoothRadius: 1, // сглаживание самой границы
  smoothPasses: 2,
};

/** Сколько раз хватит по краю сетки, чтобы размытие не упиралось в стенку. */
const padCells = (o) => o.blurRadius * o.blurPasses + 3;

function blurAxis(src, dst, dims, axis, radius) {
  const strides = [1, dims[0], dims[0] * dims[1]];
  const [a, b] = [0, 1, 2].filter((k) => k !== axis);
  const stride = strides[axis];
  const length = dims[axis];
  const norm = 1 / (2 * radius + 1);
  for (let j = 0; j < dims[b]; j++) {
    for (let i = 0; i < dims[a]; i++) {
      const base = i * strides[a] + j * strides[b];
      let sum = 0;
      for (let k = 0; k <= radius && k < length; k++) sum += src[base + k * stride];
      for (let k = 0; k < length; k++) {
        dst[base + k * stride] = sum * norm;
        const added = k + radius + 1;
        if (added < length) sum += src[base + added * stride];
        const removed = k - radius;
        if (removed >= 0) sum -= src[base + removed * stride];
      }
    }
  }
}

/** Размытие по трём осям `passes` раз; возвращает новый массив, исходный не меняется. */
function blur(field, dims, radius, passes) {
  let from = Float64Array.from(field);
  let to = new Float64Array(field.length);
  for (let pass = 0; pass < passes; pass++) {
    for (let axis = 0; axis < 3; axis++) {
      blurAxis(from, to, dims, axis, radius);
      [from, to] = [to, from];
    }
  }
  return from;
}

const NEIGHBOURS = (dims) => [1, -1, dims[0], -dims[0], dims[0] * dims[1], -dims[0] * dims[1]];

/** Ячейки, до которых можно дойти от края сетки, не заходя в `inside` (1 — внутри). */
function reachableOutside(inside, dims) {
  const [nx, ny, nz] = dims;
  const outside = new Uint8Array(inside.length);
  const queue = new Int32Array(inside.length);
  let head = 0;
  let tail = 0;
  const push = (cell) => {
    if (inside[cell] === 0 && outside[cell] === 0) {
      outside[cell] = 1;
      queue[tail++] = cell;
    }
  };
  push(0); // угол сетки всегда пустой: по краю лежит запас
  const offsets = NEIGHBOURS(dims);
  while (head < tail) {
    const cell = queue[head++];
    const x = cell % nx;
    const y = Math.floor(cell / nx) % ny;
    const z = Math.floor(cell / (nx * ny));
    const edges = [x < nx - 1, x > 0, y < ny - 1, y > 0, z < nz - 1, z > 0];
    for (let k = 0; k < 6; k++) if (edges[k]) push(cell + offsets[k]);
  }
  return outside;
}

/** Оставляет в `solid` только самый большой связный кусок; возвращает, сколько кусков было. */
function keepLargestPiece(solid, dims) {
  const label = new Int32Array(solid.length);
  const queue = new Int32Array(solid.length);
  const [nx, ny, nz] = dims;
  const offsets = NEIGHBOURS(dims);
  const sizes = [0];
  for (let start = 0; start < solid.length; start++) {
    if (solid[start] === 0 || label[start] !== 0) continue;
    const id = sizes.length;
    let head = 0;
    let tail = 0;
    label[start] = id;
    queue[tail++] = start;
    while (head < tail) {
      const cell = queue[head++];
      const x = cell % nx;
      const y = Math.floor(cell / nx) % ny;
      const z = Math.floor(cell / (nx * ny));
      const edges = [x < nx - 1, x > 0, y < ny - 1, y > 0, z < nz - 1, z > 0];
      for (let k = 0; k < 6; k++) {
        if (!edges[k]) continue;
        const next = cell + offsets[k];
        if (solid[next] !== 0 && label[next] === 0) {
          label[next] = id;
          queue[tail++] = next;
        }
      }
    }
    sizes.push(tail);
  }
  let best = 0;
  for (let id = 1; id < sizes.length; id++) if (sizes[id] > sizes[best]) best = id;
  for (let cell = 0; cell < solid.length; cell++) solid[cell] = label[cell] === best && best !== 0 ? 1 : 0;
  return sizes.length - 1;
}

// 12 рёбер куба: пары номеров углов (угол c — это биты x = c&1, y = c>>1&1, z = c>>2&1).
const CUBE_EDGES = [];
for (let corner = 0; corner < 8; corner++) {
  for (let axis = 0; axis < 3; axis++) if (!(corner & (1 << axis))) CUBE_EDGES.push([corner, corner | (1 << axis)]);
}

/** Поверхность уровня `level` поля `values` (surface nets). Координаты вершин — в кубиках сетки. */
function surfaceNets(values, dims, level) {
  const [nx, ny, nz] = dims;
  const strides = [1, nx, nx * ny];
  const cellVertex = new Int32Array(values.length).fill(-1);
  const cornerOffset = Array.from({ length: 8 }, (_, c) => (c & 1) * strides[0] + ((c >> 1) & 1) * strides[1] + ((c >> 2) & 1) * strides[2]);
  const cornerValue = new Float64Array(8);
  const vertices = [];
  const masks = new Uint8Array(values.length);

  for (let z = 0; z < nz - 1; z++) {
    for (let y = 0; y < ny - 1; y++) {
      for (let x = 0; x < nx - 1; x++) {
        const cell = x + y * nx + z * nx * ny;
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const value = values[cell + cornerOffset[c]] - level;
          cornerValue[c] = value;
          if (value > 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;
        masks[cell] = mask;
        let sx = 0;
        let sy = 0;
        let sz = 0;
        let crossings = 0;
        for (const [a, b] of CUBE_EDGES) {
          if (((mask >> a) & 1) === ((mask >> b) & 1)) continue;
          const t = cornerValue[a] / (cornerValue[a] - cornerValue[b]);
          sx += (a & 1) + ((b & 1) - (a & 1)) * t;
          sy += ((a >> 1) & 1) + (((b >> 1) & 1) - ((a >> 1) & 1)) * t;
          sz += ((a >> 2) & 1) + (((b >> 2) & 1) - ((a >> 2) & 1)) * t;
          crossings++;
        }
        cellVertex[cell] = vertices.length / 3;
        vertices.push(x + sx / crossings, y + sy / crossings, z + sz / crossings);
      }
    }
  }

  const indices = [];
  for (let z = 1; z < nz - 1; z++) {
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const cell = x + y * nx + z * nx * ny;
        const mask = masks[cell];
        if (mask === 0) continue;
        for (let axis = 0; axis < 3; axis++) {
          // ребро от угла 0 вдоль оси `axis` пересекает поверхность?
          if (((mask & 1) !== 0) === ((mask & (1 << (1 << axis))) !== 0)) continue;
          const du = strides[(axis + 1) % 3];
          const dv = strides[(axis + 2) % 3];
          const quad = (mask & 1)
            ? [cellVertex[cell], cellVertex[cell - du], cellVertex[cell - du - dv], cellVertex[cell - dv]]
            : [cellVertex[cell], cellVertex[cell - dv], cellVertex[cell - du - dv], cellVertex[cell - du]];
          if (quad.some((v) => v < 0)) continue;
          indices.push(quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]);
        }
      }
    }
  }
  return { vertices: Float32Array.from(vertices), indices: Uint32Array.from(indices) };
}

/**
 * `positions` — x, y, z нейронов подряд (мкм), `classes[i] === noPosition` — у нейрона нет координат.
 * Возвращает `{ vertices, indices, info }`: вершины в мкм (в тех же осях, что у точек) и треугольники.
 * `info.covered` — доля точек, лежащих внутри силуэта (проверка «силуэт охватывает облако»).
 */
export function buildShellMesh(positions, classes, noPosition, options = {}) {
  const o = { ...SHELL, ...options };
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  let points = 0;
  for (let i = 0; i < classes.length; i++) {
    if (classes[i] === noPosition) continue;
    points++;
    for (let axis = 0; axis < 3; axis++) {
      const value = positions[i * 3 + axis];
      if (value < min[axis]) min[axis] = value;
      if (value > max[axis]) max[axis] = value;
    }
  }
  const empty = { vertices: new Float32Array(0), indices: new Uint32Array(0), info: { points, vertices: 0, triangles: 0, covered: 0, pieces: 0 } };
  if (points === 0) return empty;

  const pad = padCells(o);
  const origin = min.map((value) => value - pad * o.voxelUm);
  const dims = min.map((value, axis) => Math.ceil((max[axis] - value) / o.voxelUm) + 1 + 2 * pad);
  const cells = dims[0] * dims[1] * dims[2];
  const cellOf = (i) => {
    const at = [0, 1, 2].map((axis) => Math.floor((positions[i * 3 + axis] - origin[axis]) / o.voxelUm));
    return at[0] + at[1] * dims[0] + at[2] * dims[0] * dims[1];
  };

  const counts = new Float64Array(cells);
  for (let i = 0; i < classes.length; i++) if (classes[i] !== noPosition) counts[cellOf(i)] += 1;
  const density = blur(counts, dims, o.blurRadius, o.blurPasses);

  const inside = new Uint8Array(cells);
  for (let cell = 0; cell < cells; cell++) inside[cell] = density[cell] > o.minDensity ? 1 : 0;
  const outside = reachableOutside(inside, dims);
  const solid = new Uint8Array(cells);
  for (let cell = 0; cell < cells; cell++) solid[cell] = outside[cell] === 0 ? 1 : 0;
  const pieces = keepLargestPiece(solid, dims);

  const smooth = blur(solid, dims, o.smoothRadius, o.smoothPasses);
  const mesh = surfaceNets(smooth, dims, 0.5);
  for (let k = 0; k < mesh.vertices.length; k++) {
    const axis = k % 3;
    mesh.vertices[k] = origin[axis] + mesh.vertices[k] * o.voxelUm;
  }

  let covered = 0;
  for (let i = 0; i < classes.length; i++) if (classes[i] !== noPosition && smooth[cellOf(i)] > 0.5) covered++;
  return {
    ...mesh,
    info: { points, vertices: mesh.vertices.length / 3, triangles: mesh.indices.length / 3, covered: covered / points, pieces },
  };
}
