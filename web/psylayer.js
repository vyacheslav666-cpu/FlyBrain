// Слой «серотониновые нейроны и их мишени» (P2, D-066): поверх облака, по переключателю в панели «Психоделик».
// Розовые — серотониновые нейроны (строгий критерий, D-054), жёлтые — мишени: чем больше у нейрона серотониновых
// синапсов, тем ярче — по той же h(s) = s / (s + s_half), с которой доза снижает порог (D-055).
import * as THREE from "three";
import { NO_POSITION } from "./constants.js";

const SEROTONIN_RGB = [1.0, 0.37, 0.82];
const TARGET_RGB = [1.0, 0.76, 0.28];
const TARGET_MIN_ALPHA = 0.05; // мишень с одним синапсом ещё видна, но тусклая
const TARGET_MAX_ALPHA = 0.45; // и самые «нагруженные» не выжигают картинку в белое
const SEROTONIN_SIZE = 3.2; // серотониновых нейронов всего 62 — рисуем их крупнее

/**
 * `positions`, `classes` — облако; `offset` — сдвиг облака; `sizeUniforms` — те же uniform-ы размера точки, что у
 * облака (обновляются при изменении окна); `info` — ответ /api/psy.
 */
export function createPsyLayer(positions, classes, offset, sizeUniforms, info) {
  const object = new THREE.Group();
  object.visible = false;
  object.position.copy(offset);
  if (!info.available) return { object, setVisible() {} };

  const targets = info.targets.map((index, k) => {
    const h = info.target_synapses[k] / (info.target_synapses[k] + info.s_half);
    return { index, alpha: TARGET_MIN_ALPHA + (TARGET_MAX_ALPHA - TARGET_MIN_ALPHA) * h };
  });
  const serotonin = info.serotonin.map((index) => ({ index, alpha: 1 }));
  // Мишени складываются светом (как облако); серотониновые нейроны рисуются поверх обычной смесью, крупнее —
  // иначе розовый на жёлтом сливался бы в белый.
  object.add(makePoints(targets, TARGET_RGB, 1, THREE.AdditiveBlending));
  const top = makePoints(serotonin, SEROTONIN_RGB, SEROTONIN_SIZE, THREE.NormalBlending);
  top.renderOrder = 1;
  object.add(top);

  function makePoints(items, rgb, sizeFactor, blending) {
    const shown = items.filter((item) => classes[item.index] !== NO_POSITION);
    const position = new Float32Array(shown.length * 3);
    const alpha = new Float32Array(shown.length);
    shown.forEach((item, k) => {
      position.set(positions.subarray(item.index * 3, item.index * 3 + 3), k * 3);
      alpha[k] = item.alpha;
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
    geometry.setAttribute("aAlpha", new THREE.BufferAttribute(alpha, 1));
    const material = new THREE.ShaderMaterial({
      uniforms: { ...sizeUniforms, uColor: { value: new THREE.Vector3(...rgb) }, uSize: { value: sizeFactor } },
      vertexShader: /* glsl */ `
        attribute float aAlpha;
        uniform float uSizeUm, uMinPx, uMaxPx, uViewportHeight, uSize;
        varying float vAlpha;
        void main() {
          vAlpha = aAlpha;
          vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * viewPosition;
          float pixels = uSizeUm * projectionMatrix[1][1] * uViewportHeight * 0.5 / -viewPosition.z;
          gl_PointSize = clamp(pixels, uMinPx, uMaxPx) * 1.6 * uSize;
        }`,
      fragmentShader: /* glsl */ `
        uniform vec3 uColor;
        varying float vAlpha;
        void main() {
          vec2 offset = gl_PointCoord - 0.5;
          float d2 = dot(offset, offset);
          if (d2 > 0.25) discard;
          gl_FragColor = vec4(uColor, vAlpha * smoothstep(0.25, 0.04, d2));
        }`,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending,
    });
    const points = new THREE.Points(geometry, material);
    points.frustumCulled = false;
    return points;
  }
  return { object, setVisible(flag) { object.visible = flag; } };
}
