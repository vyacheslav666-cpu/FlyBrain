// Выбор точки под мышкой через невидимый кадр (D-015): каждая точка рисуется своим цветом-номером,
// затем читаем несколько пикселей вокруг курсора и по цвету узнаём номер нейрона.
import * as THREE from "three";
import { NO_POSITION, POINT_SIZE_UM } from "./constants.js";

const PATCH = 11; // читаем квадрат 11×11 пикселей вокруг курсора: попасть в крошечную точку проще
const HALF = (PATCH - 1) / 2;
const PICK_MIN_PX = 4; // зона попадания чуть больше видимой точки
const PICK_MAX_PX = 14;

export function createPicker(geometry, offset) {
  const scene = new THREE.Scene();
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uSizeUm: { value: POINT_SIZE_UM },
      uMinPx: { value: PICK_MIN_PX },
      uMaxPx: { value: PICK_MAX_PX },
      uViewportHeight: { value: 1 },
    },
    vertexShader: /* glsl */ `
      attribute float aClass;
      attribute float aId;
      uniform float uSizeUm, uMinPx, uMaxPx, uViewportHeight;
      varying float vId;
      void main() {
        if (aClass > ${NO_POSITION - 0.5}) {   // нейрон без координат: выбрать нельзя
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
          gl_PointSize = 0.0;
          vId = 0.0;
          return;
        }
        vId = aId + 1.0;                        // 0 остаётся для «здесь ничего нет»
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewPosition;
        float pixels = uSizeUm * projectionMatrix[1][1] * uViewportHeight * 0.5 / -viewPosition.z;
        gl_PointSize = clamp(pixels, uMinPx, uMaxPx);
      }`,
    fragmentShader: /* glsl */ `
      varying float vId;
      void main() {
        float id = floor(vId + 0.5);
        float red = mod(id, 256.0);
        float green = mod(floor(id / 256.0), 256.0);
        float blue = floor(id / 65536.0);
        gl_FragColor = vec4(red, green, blue, 255.0) / 255.0;
      }`,
    transparent: false,
    depthTest: true,   // из перекрывающихся точек выигрывает ближняя к камере
    depthWrite: true,
    blending: THREE.NoBlending,
  });
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  points.position.copy(offset);
  scene.add(points);

  const target = new THREE.WebGLRenderTarget(1, 1, {
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: true,
  });
  const pixels = new Uint8Array(PATCH * PATCH * 4);
  const savedColor = new THREE.Color();

  return {
    // Кадр для выбора рисуется в обычных пикселях экрана (без поправки на плотность экрана): так он дешевле.
    resize(width, height) {
      target.setSize(width, height);
      material.uniforms.uViewportHeight.value = height;
    },

    /** Номер нейрона под точкой (x, y) окна в пикселях или −1. */
    pick(renderer, camera, x, y) {
      const { width, height } = target;
      const column = Math.min(Math.max(Math.floor(x), 0), width - 1);
      const row = Math.min(Math.max(height - 1 - Math.floor(y), 0), height - 1); // у WebGL строки идут снизу вверх
      const left = Math.min(Math.max(column - HALF, 0), width - PATCH);
      const bottom = Math.min(Math.max(row - HALF, 0), height - PATCH);

      const previousTarget = renderer.getRenderTarget();
      const previousAlpha = renderer.getClearAlpha();
      renderer.getClearColor(savedColor);
      renderer.setRenderTarget(target);
      renderer.setClearColor(0x000000, 0);
      renderer.clear();
      renderer.render(scene, camera);
      renderer.readRenderTargetPixels(target, left, bottom, PATCH, PATCH, pixels);
      renderer.setRenderTarget(previousTarget);
      renderer.setClearColor(savedColor, previousAlpha);

      let best = -1;
      let bestDistance = Infinity;
      for (let j = 0; j < PATCH; j++) {
        for (let i = 0; i < PATCH; i++) {
          const at = (j * PATCH + i) * 4;
          const id = pixels[at] + pixels[at + 1] * 256 + pixels[at + 2] * 65536;
          if (id === 0) continue;
          const distance = (left + i - column) ** 2 + (bottom + j - row) ** 2;
          if (distance < bestDistance) {
            bestDistance = distance;
            best = id - 1;
          }
        }
      }
      return best;
    },
  };
}
