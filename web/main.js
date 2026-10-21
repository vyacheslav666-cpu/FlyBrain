// 3D-облако нейронов мухи (C7) с подсказкой, закреплением нейрона и поиском (C8) и «живым» мозгом (C9):
// сервер считает модель и шлёт, кто выстрелил; выстрелившие нейроны вспыхивают и гаснут.
import * as THREE from "three";
import { OrbitControls } from "/web/vendor/OrbitControls.js";
import { HEADER_BYTES, MAX_CLASSES, NO_POSITION, POINT_SIZE_UM } from "./constants.js";
import { createGlow } from "./glow.js";
import { createCard, createSearch, createTooltip, getInfo, getLinks, RATE_HINT } from "./inspector.js";
import { connectLive } from "./live.js";
import { createLinks, createRings } from "./markers.js";
import { createPicker } from "./picking.js";

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

function formatNumber(number, digits) {
  return number.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits });
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
  // Внутренний номер нейрона = номер точки; по нему сервер отвечает на вопросы про нейрон.
  geometry.setAttribute("aId", new THREE.BufferAttribute(Float32Array.from({ length: classes.length }, (_, i) => i), 1));
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
  const offset = new THREE.Vector3(-cx, -cy, -cz); // мозг в начале координат: камера вращается вокруг его середины
  points.position.copy(offset);
  return { points, material, geometry, offset, palette };
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


// Цвета колец и линий (в цветах экрана). Линии: синие — входы, оранжевые — выходы.
const COLOR = { hover: 0xffffff, pinned: 0x5ff0ff, search: 0xffc247, input: 0x4fc3ff, output: 0xff9a3d };
const CLICK_MAX_MOVE_PX = 5; // сдвиг мыши больше этого — это вращение, а не щелчок
const FLIGHT_MS = 900;
const LIVE_TEXT_EVERY_MS = 250; // как часто обновлять цифры живого режима (скорость, частоты)

/** Панель «Живой мозг»: кнопки, скорость (D-017), плашка «в покое мозг молчит» (D-019). */
function createLivePanel(onCommand) {
  const $ = (id) => document.getElementById(id);
  const pause = $("live-pause");
  const reset = $("live-reset");
  const test = $("live-test");
  let status = null;
  let connected = false;

  pause.addEventListener("click", () => onCommand({ cmd: status?.paused ? "resume" : "pause" }));
  reset.addEventListener("click", () => onCommand({ cmd: "reset" }));
  test.addEventListener("click", () => onCommand({ cmd: "test", on: !status?.test }));

  function stateText() {
    if (!connected) return "Нет связи с сервером: подключаюсь снова…";
    if (!status) return "Подключаюсь к серверу…";
    switch (status.state) {
      case "starting": return "Готовлю движок модели (несколько секунд)…";
      case "error": return `Ошибка: ${status.message}`;
      case "paused": return "Пауза: модель стоит, свечение замерло.";
      case "running": return `Считает ${status.device}.`;
      default: return "Жду…";
    }
  }

  function refresh() {
    const ready = connected && status && (status.state === "running" || status.state === "paused");
    pause.disabled = reset.disabled = test.disabled = !ready;
    pause.textContent = status?.paused ? "▶ Продолжить" : "⏸ Пауза";
    test.setAttribute("aria-pressed", String(Boolean(status?.test)));
    if (status) {
      test.textContent = `Тест: ${status.test_label.toLowerCase()} ${formatInt(status.test_hz)} Гц` +
        (status.test ? " — выключить" : "");
      $("live-test-note").textContent =
        `Временная кнопка: ${status.test_count} нейронов — ${status.test_about}. ` +
        "Прямое включение нейронов случайными импульсами, как в статье, а не настоящий сахар.";
    }
    $("live-quiet").hidden = Boolean(status?.test);
    $("live-state").textContent = stateText();
    const running = ready && status.state === "running";
    const speed = status?.speed ?? 0;
    $("live-speed").textContent = running && speed > 0 ? `${formatNumber(speed, 2)}× реального времени` : "—";
    $("live-speed-note").textContent = running && speed > 0
      ? (status.max_speed && speed > 0.98 * status.max_speed
        ? "Видеокарта успевает быстрее, но модель нарочно идёт не быстрее реального времени."
        : `1 с жизни мухи считается ${formatNumber(1 / speed, 1)} с; шаги по 0,1 мс не пропускаем.`)
      : "";
    $("live-time").textContent = status ? `${formatNumber(status.time_ms / 1000, 2)} с` : "—";
  }

  const root = document.getElementById("live");
  root.hidden = false;
  // Карточка нейрона стоит над панелью и не должна на неё наезжать.
  new ResizeObserver(() => {
    document.documentElement.style.setProperty("--live-height", `${root.offsetHeight}px`);
  }).observe(root);
  return {
    setStatus(next) { status = next; refresh(); },
    setConnected(value) { connected = value; refresh(); },
    refresh,
    get running() { return connected && status?.state === "running"; },
  };
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

  const { points, material, geometry, offset, palette } = buildPoints(meta, positions, classes);
  scene.add(points);
  const picker = createPicker(geometry, offset);
  const glow = createGlow(geometry, palette, offset, classes.length);

  // Кольца и линии лежат в тех же координатах, что и облако.
  const hoverRing = createRings({ color: COLOR.hover, sizePx: 22, alpha: 0.9 });
  const pinnedRing = createRings({ color: COLOR.pinned, sizePx: 32 });
  const searchRings = createRings({ color: COLOR.search, sizePx: 18, alpha: 0.9 });
  const inputRings = createRings({ color: COLOR.input, sizePx: 14, alpha: 0.85 });
  const outputRings = createRings({ color: COLOR.output, sizePx: 14, alpha: 0.85 });
  const links = createLinks();
  const rings = [hoverRing, pinnedRing, searchRings, inputRings, outputRings];
  for (const layer of [links, ...rings]) {
    layer.object.position.copy(offset);
    scene.add(layer.object);
  }

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.rotateSpeed = 0.7;
  controls.minDistance = 20;
  controls.maxDistance = meta.radius * 8;
  controls.autoRotateSpeed = 0.8;

  const tooltipElement = document.getElementById("tooltip");
  const tooltip = createTooltip(tooltipElement);
  const card = createCard(document.getElementById("card"), { onClose: () => unpin(), onSelect: (index) => pin(index) });

  let autoRotateWanted = true; // медленно крутится, пока вы не взяли мышь
  let hovered = -1;
  let pinned = -1;
  let searchActive = false;
  let flight = null;
  let pickDirty = false;
  let dragging = false;
  let pressedAt = null;
  const mouse = { x: 0, y: 0, inside: false };

  const pointOf = (index) => [positions[index * 3], positions[index * 3 + 1], positions[index * 3 + 2]];
  const flat = (indices) => indices.flatMap(pointOf);

  // ---------- наведение ----------
  function setHovered(index) {
    if (index === hovered) return;
    hovered = index;
    canvas.style.cursor = index >= 0 ? "pointer" : "";
    hoverRing.set(index >= 0 && index !== pinned ? pointOf(index) : []);
    if (index < 0) {
      tooltip.hide();
      return;
    }
    getInfo(index).then((info) => {
      if (hovered === index && !dragging && index !== pinned) tooltip.show(info, mouse.x, mouse.y);
    }).catch(() => {});
  }

  // ---------- закрепление ----------
  async function pin(index) {
    pinned = index;
    hoverRing.set([]);
    tooltip.hide();
    pinnedRing.set(pointOf(index));
    inputRings.set([]);
    outputRings.set([]);
    links.set([]);
    try {
      const info = await getInfo(index);
      if (pinned !== index) return;
      card.showInfo(info);
      const connections = await getLinks(index);
      if (pinned !== index) return;
      card.showLinks(connections);
      drawLinks(index, connections);
    } catch (error) {
      card.showError(error.message);
    }
  }

  function drawLinks(index, connections) {
    const strongest = Math.max(1, ...connections.inputs.map((p) => p.synapses), ...connections.outputs.map((p) => p.synapses));
    const centre = pointOf(index);
    const lines = [];
    const make = (partners, color, toCentre) => {
      const shown = partners.filter((partner) => partner.visible);
      for (const partner of shown) {
        const other = pointOf(partner.index);
        lines.push({
          from: toCentre ? other : centre,
          to: toCentre ? centre : other,
          color: new THREE.Color(color),
          strength: partner.synapses / strongest,
        });
      }
      return flat(shown.map((partner) => partner.index));
    };
    inputRings.set(make(connections.inputs, COLOR.input, true));
    outputRings.set(make(connections.outputs, COLOR.output, false));
    links.set(lines);
  }

  function unpin() {
    pinned = -1;
    pinnedRing.set([]);
    inputRings.set([]);
    outputRings.set([]);
    links.set([]);
    card.hide();
    pickDirty = true; // под курсором снова может оказаться нейрон
  }

  // ---------- поиск и полёт камеры ----------
  function flyToNeurons(indices) {
    const sum = [0, 0, 0];
    for (const index of indices) pointOf(index).forEach((value, axis) => { sum[axis] += value; });
    const centre = sum.map((value) => value / indices.length);
    let spread = 0;
    for (const index of indices) {
      const [x, y, z] = pointOf(index);
      spread = Math.max(spread, Math.hypot(x - centre[0], y - centre[1], z - centre[2]));
    }
    const toTarget = new THREE.Vector3(...centre).add(offset);
    const toDistance = Math.min(Math.max(spread * 2.6, 70), controls.maxDistance);
    const direction = camera.position.clone().sub(controls.target).normalize();
    flight = {
      start: performance.now(),
      fromTarget: controls.target.clone(),
      toTarget,
      fromPosition: camera.position.clone(),
      toPosition: toTarget.clone().addScaledVector(direction, toDistance),
    };
    autoRotateWanted = false;
  }

  function stepFlight() {
    if (!flight) return;
    const progress = Math.min((performance.now() - flight.start) / FLIGHT_MS, 1);
    const eased = progress < 0.5 ? 4 * progress ** 3 : 1 - (-2 * progress + 2) ** 3 / 2;
    controls.target.lerpVectors(flight.fromTarget, flight.toTarget, eased);
    camera.position.lerpVectors(flight.fromPosition, flight.toPosition, eased);
    if (progress >= 1) flight = null;
  }

  createSearch({
    input: document.getElementById("search-input"),
    status: document.getElementById("search-status"),
    clearButton: document.getElementById("search-clear"),
    onResult(result) {
      searchActive = true;
      searchRings.set(flat(result.indices));
      flyToNeurons(result.indices);
    },
    onClear() {
      searchActive = false;
      searchRings.set([]);
    },
  });

  // ---------- мышь и клавиши ----------
  controls.addEventListener("start", () => {
    autoRotateWanted = false;
    flight = null;
  });
  controls.addEventListener("change", () => { pickDirty = true; });

  canvas.addEventListener("pointermove", (event) => {
    mouse.x = event.clientX;
    mouse.y = event.clientY;
    mouse.inside = true;
    pickDirty = true;
    if (hovered >= 0 && !tooltipElement.hidden) tooltip.moveTo(mouse.x, mouse.y);
  });
  canvas.addEventListener("pointerleave", () => {
    mouse.inside = false;
    setHovered(-1);
  });
  canvas.addEventListener("pointerdown", (event) => {
    pressedAt = { x: event.clientX, y: event.clientY };
    dragging = true;
    tooltip.hide();
  });
  canvas.addEventListener("pointerup", (event) => {
    dragging = false;
    pickDirty = true;
    const wasClick = pressedAt && event.button === 0 &&
      Math.hypot(event.clientX - pressedAt.x, event.clientY - pressedAt.y) <= CLICK_MAX_MOVE_PX;
    pressedAt = null;
    if (!wasClick) return;
    if (hovered >= 0) pin(hovered);
    else if (pinned >= 0) unpin();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && pinned >= 0 && document.activeElement?.tagName !== "INPUT") unpin();
  });

  function resize() {
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    renderer.setPixelRatio(pixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    material.uniforms.uViewportHeight.value = window.innerHeight * pixelRatio;
    material.uniforms.uMinPx.value = MIN_POINT_PX * pixelRatio;
    material.uniforms.uMaxPx.value = MAX_POINT_PX * pixelRatio;
    picker.resize(window.innerWidth, window.innerHeight);
    // Свечение — в обычных пикселях экрана, без поправки на плотность: оно и так размытое, а на встроенной
    // видеокарте ноутбука это в 3 раза дешевле (замер C9: 3,4 мс против 11 мс на кадр).
    glow.resize(window.innerWidth, window.innerHeight, Math.min(pixelRatio, 1));
    for (const layer of rings) layer.setPixelRatio(pixelRatio);
    pickDirty = true;
  }
  window.addEventListener("resize", resize);
  resize();

  // ---------- живой мозг ----------
  const panel = createLivePanel((command) => live.send(command));
  const live = connectLive({
    count: classes.length,
    onFrame: (neurons, spikes) => glow.spike(neurons, spikes),
    onStatus: (status) => panel.setStatus(status),
    onReset: () => glow.clear(),
    onConnection: (connected) => panel.setConnected(connected),
  });

  function rateText(index) {
    const rate = index >= 0 ? live.rateOf(index) : null;
    if (!rate) return "— (модель ещё не считала)";
    const span = rate.spanMs < 995 ? ` (за ${formatNumber(rate.spanMs / 1000, 2)} с)` : "";
    return `${formatNumber(rate.hz, 1)} Гц${span}`;
  }

  function refreshLiveText() {
    panel.refresh();
    const tooltipRate = tooltipElement.querySelector(".live-rate");
    if (tooltipRate && !tooltipElement.hidden) tooltipRate.textContent = rateText(hovered);
    const cardRate = document.querySelector("#card .live-rate");
    if (cardRate) {
      cardRate.textContent = rateText(pinned);
      cardRate.title = RATE_HINT;
    }
  }
  setInterval(refreshLiveText, LIVE_TEXT_EVERY_MS);

  let lastFrameAt = performance.now();
  renderer.setAnimationLoop(() => {
    const frameAt = performance.now();
    // Свечение гаснет по часам показа, которые стоят на паузе: вспышки замирают вместе с моделью.
    if (panel.running) glow.advance(Math.min((frameAt - lastFrameAt) / 1000, 0.1));
    lastFrameAt = frameAt;
    stepFlight();
    controls.autoRotate = autoRotateWanted && hovered < 0 && pinned < 0 && !searchActive;
    controls.update();
    if (pickDirty && mouse.inside && !dragging) {
      pickDirty = false;
      setHovered(picker.pick(renderer, camera, mouse.x, mouse.y));
    }
    renderer.render(scene, camera);
    glow.render(renderer, camera);
  });

  buildLegend(meta);
  statusEl.textContent = `Показано нейронов: ${formatInt(meta.shown)} из ${formatInt(meta.total)}`;
  document.getElementById("search").hidden = false;
  window.flybrain = { camera, controls, renderer, scene, picker, pin, unpin, live, glow }; // для отладки в консоли браузера
}

loadData().then(start).catch((error) => showError(`Ошибка: ${error.message}`));
