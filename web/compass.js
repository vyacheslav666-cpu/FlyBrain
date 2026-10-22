// Компас внизу посередине экрана (C12, D-044): три оси мухи — «перед», «лево», «верх» — поворачиваются вместе
// с мозгом, чтобы после вращения было видно, откуда смотришь. Рисуется на обычном 2D-холсте поверх сцены
// (а не в 3D-сцене): так он попадает в снимок экрана. Каждая ось — луч из центра, без продолжения за центр:
// иначе ось «верх» выглядела бы как линия в обе стороны, и казалось, что она двоится.
import * as THREE from "three";

// Оси данных (D-025): x — к левой стороне мухи, y — вниз, z — назад.
const AXES = [
  { label: "перед", direction: new THREE.Vector3(0, 0, -1), color: "#5ff0ff" },
  { label: "лево", direction: new THREE.Vector3(1, 0, 0), color: "#ffc247" },
  { label: "верх", direction: new THREE.Vector3(0, -1, 0), color: "#c4a8ff" },
];
const FONT = '12px "Segoe UI", system-ui, sans-serif';
const RADIUS = 24; // длина оси, px
const LABEL_GAP = 5;
const LINE_HEIGHT = 14; // подписи ближе этого по вертикали считаются налезшими
const FACING_LENGTH = 0.3; // ось короче этого на экране (смотрит почти на зрителя) — подпись уходит вверх-вправо

/** `canvas` — холст нужного размера в CSS-пикселях (width/height в разметке), `camera` — камера сцены. */
export function createCompass(canvas, camera) {
  const cssWidth = canvas.width;
  const cssHeight = canvas.height;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.width = `${cssWidth}px`;
  canvas.style.height = `${cssHeight}px`;
  canvas.width = Math.round(cssWidth * ratio);
  canvas.height = Math.round(cssHeight * ratio);
  const context = canvas.getContext("2d");
  const lastQuaternion = new THREE.Quaternion(NaN, NaN, NaN, NaN);
  const inverse = new THREE.Quaternion();
  const vector = new THREE.Vector3();

  function draw() {
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, cssWidth, cssHeight);
    context.font = FONT;
    context.textBaseline = "middle";
    context.lineCap = "round";
    const cx = cssWidth / 2;
    const cy = cssHeight / 2;
    inverse.copy(camera.quaternion).invert();
    // Рисуем от дальней оси к ближней: ближняя перекрывает.
    const axes = AXES.map((axis) => {
      vector.copy(axis.direction).applyQuaternion(inverse); // в осях камеры: x — вправо, y — вверх, z — к зрителю
      return { ...axis, x: vector.x, y: -vector.y, z: vector.z };
    }).sort((a, b) => a.z - b.z);

    context.strokeStyle = "rgba(255,255,255,0.18)";
    context.beginPath();
    context.arc(cx, cy, 2, 0, Math.PI * 2);
    context.stroke();

    const placed = [];
    for (const axis of axes) {
      const near = (axis.z + 1) / 2; // 0 — ось уходит от зрителя, 1 — смотрит на зрителя
      context.globalAlpha = 0.4 + 0.6 * near;
      context.strokeStyle = axis.color;
      context.fillStyle = axis.color;
      context.lineWidth = 1.5 + near;
      const tipX = cx + axis.x * RADIUS;
      const tipY = cy + axis.y * RADIUS;
      context.beginPath();
      context.moveTo(cx, cy);
      context.lineTo(tipX, tipY);
      context.stroke();
      context.beginPath();
      context.arc(tipX, tipY, 2.2 + 1.4 * near, 0, Math.PI * 2);
      context.fill();

      // Подпись — за кончиком оси. Её место меняется плавно: у оси, смотрящей вправо, подпись стоит справа от кончика,
      // у смотрящей вверх — по центру над ним, у смотрящей влево — слева (раньше подпись перескакивала с одной стороны
      // на другую, когда ось проходила через вертикаль, — D-045). Если ось смотрит почти на зрителя, подпись плавно
      // уходит вверх-вправо.
      const length = Math.hypot(axis.x, axis.y);
      const weight = Math.min(1, length / FACING_LENGTH);
      let dx = weight * (length > 0 ? axis.x / length : 0) + (1 - weight) * 0.7;
      let dy = weight * (length > 0 ? axis.y / length : 0) + (1 - weight) * -0.7;
      const norm = Math.hypot(dx, dy) || 1;
      dx /= norm;
      dy /= norm;
      const width = context.measureText(axis.label).width;
      let labelX = tipX + dx * LABEL_GAP - width * (1 - dx) / 2;
      let labelY = tipY + dy * (LABEL_GAP + 4);
      labelX = Math.min(Math.max(labelX, 1), cssWidth - width - 1);
      labelY = Math.min(Math.max(labelY, 7), cssHeight - 7);
      // Подписи не налезают друг на друга: ближняя к зрителю (рисуется позже) отодвигается по вертикали.
      for (const box of placed) {
        const overlapX = labelX < box.x + box.width + 2 && box.x < labelX + width + 2;
        if (overlapX && Math.abs(labelY - box.y) < LINE_HEIGHT) {
          labelY = labelY >= box.y ? box.y + LINE_HEIGHT : box.y - LINE_HEIGHT;
          labelY = Math.min(Math.max(labelY, 7), cssHeight - 7);
        }
      }
      placed.push({ x: labelX, y: labelY, width });
      context.fillText(axis.label, labelX, labelY);
    }
    context.globalAlpha = 1;
  }

  draw();
  return {
    /** Перерисовать, если камера повернулась. */
    update() {
      if (lastQuaternion.equals(camera.quaternion)) return;
      lastQuaternion.copy(camera.quaternion);
      draw();
    },
    /** Нарисовать компас на готовом снимке: `x`, `y` — левый верхний угол, `scale` — снимок/CSS-пиксели. */
    paintOn(target, x, y, scale) {
      target.drawImage(canvas, x, y, cssWidth * scale, cssHeight * scale);
    },
    cssWidth,
    cssHeight,
  };
}
