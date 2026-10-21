// Проверка силуэта мозга (web/shellmesh.js) на Node. Запуск: node shell_check.mjs [--cloud файл.bin]
// Без аргументов проверяет искусственное облако (полая шаровая «кора» + одинокие точки); с --cloud — настоящее.
import { readFileSync } from "node:fs";
import { buildShellMesh } from "../../web/shellmesh.js";
import { HEADER_BYTES, NO_POSITION } from "../../web/constants.js";

function fail(message) {
  console.error(`ОШИБКА: ${message}`);
  process.exit(1);
}

function watertight(indices) {
  const edges = new Map();
  for (let t = 0; t < indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const a = indices[t + k];
      const b = indices[t + ((k + 1) % 3)];
      const key = a < b ? `${a}_${b}` : `${b}_${a}`;
      edges.set(key, (edges.get(key) ?? 0) + 1);
    }
  }
  let bad = 0;
  for (const count of edges.values()) if (count !== 2) bad++;
  return { edges: edges.size, bad };
}

function synthetic() {
  let seed = 12345;
  const random = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);
  const pts = [];
  while (pts.length < 20000) { // «кора»: радиус 100–120 мкм, внутри пусто
    const p = [random() * 280 - 140, random() * 280 - 140, random() * 280 - 140];
    const r = Math.hypot(...p);
    if (r >= 100 && r <= 120) pts.push(p);
  }
  pts.push([900, 0, 0], [0, 900, 0], [-900, -900, 0]); // одинокие точки далеко от мозга
  const positions = new Float32Array(pts.length * 3);
  pts.forEach((p, i) => positions.set(p, i * 3));
  const classes = new Uint8Array(pts.length);
  classes[pts.length - 1] = NO_POSITION; // и один нейрон без координат
  return { positions, classes };
}

function fromCloud(path) {
  const buffer = readFileSync(path);
  const bytes = new Uint8Array(buffer).buffer;
  const [count] = new Uint32Array(bytes, 0, 2);
  return {
    positions: new Float32Array(bytes, HEADER_BYTES, count * 3),
    classes: new Uint8Array(bytes, HEADER_BYTES + count * 12, count),
  };
}

const cloudAt = process.argv.indexOf("--cloud");
const { positions, classes } = cloudAt > 0 ? fromCloud(process.argv[cloudAt + 1]) : synthetic();
const started = performance.now();
const { vertices, indices, info } = buildShellMesh(positions, classes, NO_POSITION);
const tookMs = performance.now() - started;
const closed = watertight(indices);
console.log(JSON.stringify({ ...info, tookMs: Math.round(tookMs), edges: closed.edges, openEdges: closed.bad }));

if (info.triangles === 0) fail("силуэт пустой");
// У настоящего мозга на тонких перемычках surface nets может оставить 1–3 дырки размером в кубик: на вид они незаметны.
const allowedOpen = cloudAt > 0 ? 20 : 0;
if (closed.bad > allowedOpen) fail(`сетка не замкнута: рёбер без пары — ${closed.bad}`);
if (info.covered < 0.97) fail(`силуэт охватил только ${(info.covered * 100).toFixed(1)}% точек`);
if (info.pieces < 1) fail("не нашлось ни одного куска");
if (tookMs > 3000) fail(`построение заняло ${Math.round(tookMs)} мс: страница будет «зависать»`);

if (cloudAt < 0) {
  // Искусственное облако: пустая внутри «кора» должна стать цельным шаром без внутренней поверхности.
  if (info.pieces !== 1) fail(`одинокие точки не должны давать кусков: ${info.pieces}`);
  let rMin = Infinity;
  let rMax = 0;
  for (let i = 0; i < vertices.length; i += 3) {
    const r = Math.hypot(vertices[i], vertices[i + 1], vertices[i + 2]);
    rMin = Math.min(rMin, r);
    rMax = Math.max(rMax, r);
  }
  if (rMin < 100) fail(`внутри силуэта осталась пустота: вершина на радиусе ${rMin.toFixed(0)} мкм`);
  if (rMax > 150) fail(`силуэт «раздулся»: вершина на радиусе ${rMax.toFixed(0)} мкм`);
}
