// Проверка web/framing.js на Node: найденные поиском нейроны не уходят под панели (D-040). Запуск: node framing_check.mjs
import { fitDistance, freeArea } from "../../web/framing.js";

function check(condition, message) {
  if (!condition) {
    console.error(`ОШИБКА: ${message}`);
    process.exit(1);
  }
}
const near = (a, b) => Math.abs(a - b) < 1e-9;

// Окно 1600×900: стимулы и «Вид» слева (до 360 px), исполнители и живой мозг справа (от 1240 px),
// поиск сверху до 60 px, подсказка снизу от 830 px.
const panels = [
  { left: 20, right: 360, top: 70, bottom: 530 }, { left: 20, right: 360, top: 545, bottom: 825 },
  { left: 1240, right: 1580, top: 70, bottom: 540 }, { left: 1280, right: 1580, top: 555, bottom: 825 },
];
const area = freeArea(panels, 1600, 900, 60, 830);
check(near(area.x, (440 - 28) / 800) && near(area.y, (380 - 28) / 450), `свободная середина ${JSON.stringify(area)}`);
// Без панелей — почти весь экран; если панели закрыли почти всё — тоже весь экран (иначе камера улетела бы).
check(freeArea([], 1600, 900).x === 0.9, "без панелей");
check(freeArea([{ left: 0, right: 760, top: 0, bottom: 900 }], 1600, 900).x === 0.9, "панели закрыли почти всё");
// Окно 1024 px: середина узкая (до панели 152 px), но камера всё равно целится в неё, а не в весь экран.
const narrow = freeArea([{ left: 20, right: 360, top: 70, bottom: 700 }, { left: 684, right: 1004, top: 70, bottom: 700 }],
  1024, 768);
check(near(narrow.x, (152 - 28) / 512), `узкое окно ${JSON.stringify(narrow)}`);
// Пустая (скрытая) панель не считается.
check(freeArea([{ left: 0, right: 0, top: 0, bottom: 0 }], 1600, 900).x === 0.9, "скрытая панель");

// Две точки на ±100 мкм вдоль оси «вправо» камеры: при tg = 0,5, aspect = 2 и свободной половине (0,5)
// камера должна отойти на 100 / (0,5 · 2 · 0,5) = 200, и тогда точки встанут ровно на край свободной середины.
const axes = { right: [1, 0, 0], up: [0, 1, 0], back: [0, 0, 1] };
const distance = fitDistance([[100, 0, 0], [-100, 0, 0]], [0, 0, 0], axes, 0.5, 2, { x: 0.5, y: 0.5 });
check(near(distance, 200), `расстояние ${distance}`);
const ndc = 100 / (distance * 0.5 * 2);
check(near(ndc, 0.5), `точка на экране ${ndc}`);
// Точка ближе к камере (вдоль back) требует отойти дальше на столько же.
check(near(fitDistance([[0, 50, 30]], [0, 0, 0], axes, 0.5, 2, { x: 0.5, y: 0.5 }), 50 / 0.25 + 30), "глубина");
// Каждая точка после этого — внутри свободной середины.
const random = (() => { let seed = 7; return () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32); })();
const cloud = Array.from({ length: 200 }, () => [random() * 300 - 150, random() * 120 - 60, random() * 200 - 100]);
const d = fitDistance(cloud, [0, 0, 0], axes, 0.4, 1.6, area);
for (const [x, y, z] of cloud) {
  const depth = d - z;
  check(depth > 0 && Math.abs(x / (depth * 0.4 * 1.6)) <= area.x + 1e-9 && Math.abs(y / (depth * 0.4)) <= area.y + 1e-9,
    `точка ${x}, ${y}, ${z} вне свободной середины`);
}
console.log("OK: найденные нейроны помещаются между панелями");
