// 3D-облако нейронов мухи (коммит C7). Симуляции здесь пока нет: только положения и классы клеток.
import * as THREE from "three";
import { OrbitControls } from "/web/vendor/OrbitControls.js";

const HEADER_BYTES = 8; // uint32 число нейронов, uint32 версия формата
const NO_POSITION = 255;
const MAX_CLASSES = 16; // столько цветов в шейдере; совпадает с flybrain/webdata.py
const POINT_SIZE_UM = 3.2; // видимый размер точки в мкм: примерно как тело клетки
const MIN_POINT_PX = 1.3;
const MAX_POINT_PX = 7;
const POINT_ALPHA = 0.22; // ниже, чем хотелось бы глазу: иначе плотные зрительные доли «выгорают» в белое

const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = false;
  statusEl.textContent = "Не получилось показать нейроны";
}

function formatInt(number) {
  return number.toLocaleString("ru-RU");
}

async function loadData() {
  const [metaResponse, cloudResponse] = await Promise.all([fetch("/api/meta"), fetch("/api/cloud.bin")]);
  if (!metaResponse.ok || !cloudResponse.ok) throw new Error("сервер не отдал данные нейронов");
  const meta = await metaResponse.json();
  const buffer = await cloudResponse.arrayBuffer();
  const [count, version] = new Uint32Array(buffer, 0, 2);
  if (version !== meta.format || count !== meta.total) throw new Error("данные нейронов не подходят к странице");
  return {
    meta,
    positions: new Float32Array(buffer, HEADER_BYTES, count * 3),
    classes: new Uint8Array(buffer, HEADER_BYTES + count * 12, count),
  };
}

function buildPoints(meta, positions, classes) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("aClass", new THREE.BufferAttribute(classes, 1));
  // Границы известны заранее; без этого three.js считал бы их по всем точкам.
  const [cx, cy, cz] = meta.center;
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(cx, cy, cz), meta.radius * 1.05);

  // Цвета — прямо в цветах экрана, без пересчёта three.js (палитра задана в config/legend.yaml).
  const palette = Array.from({ length: MAX_CLASSES }, () => new THREE.Vector3(0, 0, 0));
  for (const entry of meta.classes) {
    const rgb = parseInt(entry.color.slice(1), 16);
    palette[entry.code].set((rgb >> 16) / 255, ((rgb >> 8) & 255) / 255, (rgb & 255) / 255);
  }

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uPalette: { value: palette },
      uSizeUm: { value: POINT_SIZE_UM },
      uMinPx: { value: MIN_POINT_PX },
      uMaxPx: { value: MAX_POINT_PX },
      uAlpha: { value: POINT_ALPHA },
      uViewportHeight: { value: 1 },
    },
    vertexShader: /* glsl */ `
      attribute float aClass;
      uniform vec3 uPalette[${MAX_CLASSES}];
      uniform float uSizeUm, uMinPx, uMaxPx, uViewportHeight;
      varying vec3 vColor;
      void main() {
        if (aClass > ${NO_POSITION - 0.5}) {   // нейрон без координат: не рисуем
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
          gl_PointSize = 0.0;
          vColor = vec3(0.0);
          return;
        }
        vColor = uPalette[int(aClass + 0.5)];
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewPosition;
        float pixels = uSizeUm * projectionMatrix[1][1] * uViewportHeight * 0.5 / -viewPosition.z;
        gl_PointSize = clamp(pixels, uMinPx, uMaxPx);
      }`,
    fragmentShader: /* glsl */ `
      uniform float uAlpha;
      varying vec3 vColor;
      void main() {
        vec2 offset = gl_PointCoord - 0.5;
        float d2 = dot(offset, offset);
        if (d2 > 0.25) discard;
        gl_FragColor = vec4(vColor, uAlpha * smoothstep(0.25, 0.02, d2));
      }`,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    blending: THREE.AdditiveBlending,
  });

  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  points.position.set(-cx, -cy, -cz); // мозг в начале координат: камера вращается вокруг его середины
  return { points, material };
}

function buildLegend(meta) {
  const list = document.getElementById("legend-list");
  for (const entry of meta.classes) {
    if (entry.count === 0) continue;
    const item = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = entry.color;
    const name = document.createElement("span");
    name.textContent = entry.name;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = formatInt(entry.count);
    item.append(swatch, name, count);
    list.append(item);
  }
  const hiddenNote = meta.hidden > 0 ? ` У ${formatInt(meta.hidden)} нейронов в таблице нет координат, их не видно.` : "";
  document.getElementById("legend-note").textContent =
    `Точка — тело клетки (${formatInt(meta.soma)}) или, если тела нет в данных, ` +
    `точка на отростке внутри мозга (${formatInt(meta.anchor)}).${hiddenNote}`;
  document.getElementById("legend").hidden = false;
}

function start({ meta, positions, classes }) {
  const canvas = document.getElementById("scene");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false });
  renderer.setClearColor(0x05070d, 1);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 1, 6000);
  // Оси данных: x — к левой стороне мухи, y — вниз, z — назад (D-025), поэтому «верх» у камеры — это −y.
  camera.up.set(0, -1, 0);
  // Вид спереди (перед мухи — малые z), чуть сверху и сбоку.
  const distance = meta.radius * 3.1;
  camera.position.set(distance * 0.28, -distance * 0.22, -distance * 0.93);

  const { points, material } = buildPoints(meta, positions, classes);
  scene.add(points);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.rotateSpeed = 0.7;
  controls.minDistance = 20;
  controls.maxDistance = meta.radius * 8;
  controls.autoRotate = true; // медленно крутится, пока вы не взяли мышь
  controls.autoRotateSpeed = 0.8;
  controls.addEventListener("start", () => { controls.autoRotate = false; });

  function resize() {
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    material.uniforms.uViewportHeight.value = window.innerHeight * pixelRatio;
    material.uniforms.uMinPx.value = MIN_POINT_PX * pixelRatio;
    material.uniforms.uMaxPx.value = MAX_POINT_PX * pixelRatio;
  }
  window.addEventListener("resize", resize);
  resize();

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });

  buildLegend(meta);
  statusEl.textContent = `Показано нейронов: ${formatInt(meta.shown)} из ${formatInt(meta.total)}`;
  window.flybrain = { camera, controls, renderer, scene }; // для отладки в консоли браузера
}

loadData().then(start).catch((error) => showError(`Ошибка: ${error.message}`));
