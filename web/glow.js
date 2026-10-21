// Свечение выстреливших нейронов (C9). Точка вспыхивает при спайке и плавно гаснет; вокруг — ореол (bloom).
// Светящиеся точки рисуются отдельно в свою картинку, к ней добавляется ореол (UnrealBloomPass из three.js),
// и результат складывается с обычным кадром. Поэтому ореол есть только у активных нейронов, а не у всего облака.
import * as THREE from "three";
import { FullScreenQuad } from "/web/vendor/jsm/postprocessing/Pass.js";
import { UnrealBloomPass } from "/web/vendor/jsm/postprocessing/UnrealBloomPass.js";
import { MAX_CLASSES, NO_POSITION, POINT_SIZE_UM } from "./constants.js";

export const GLOW_TAU_S = 0.3; // за это время (по часам показа) вспышка гаснет в e ≈ 2,7 раза («шлейф», C12 — ползунок)
const MAX_HEAT = 4; // частые спайки накапливают «жар», но не больше этого
const QUIET_AFTER_TAUS = 12; // после стольких «шлейфов» без спайков свечение не рисуем вовсе
const BLOOM = { strength: 0.75, radius: 0.35, threshold: 0.15 };
const BLOOM_DOWNSCALE = 2;

/** `geometry` — та же геометрия облака (координаты и классы), `palette` — цвета классов. */
export function createGlow(geometry, palette, offset, count) {
  // На нейрон два числа: «жар» в момент последнего спайка и этот момент (часы показа, с).
  const glow = new Float32Array(count * 2);
  const attribute = new THREE.BufferAttribute(glow, 2);
  attribute.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("aGlow", attribute);

  const material = new THREE.ShaderMaterial({
    uniforms: {
      uPalette: { value: palette },
      uNow: { value: 0 },
      uTau: { value: GLOW_TAU_S },
      uGain: { value: 1 },
      uSizeUm: { value: POINT_SIZE_UM * 1.25 },
      uMinPx: { value: 3 },
      uMaxPx: { value: 14 },
      uViewportHeight: { value: 1 },
    },
    vertexShader: /* glsl */ `
      attribute float aClass;
      attribute vec2 aGlow;
      uniform vec3 uPalette[${MAX_CLASSES}];
      uniform float uNow, uTau, uGain, uSizeUm, uMinPx, uMaxPx, uViewportHeight;
      varying vec3 vColor;
      void main() {
        float heat = aGlow.x * exp(-(uNow - aGlow.y) / uTau);
        if (aClass > ${NO_POSITION - 0.5} || heat < 0.04) {   // не светится: не рисуем
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
          gl_PointSize = 0.0;
          vColor = vec3(0.0);
          return;
        }
        float level = 1.0 - exp(-heat);                       // 0…1: одиночный спайк ≈ 0,63, частые — ближе к 1
        vColor = mix(uPalette[int(aClass + 0.5)], vec3(1.0, 0.96, 0.88), 0.45) * (0.25 + 0.8 * level) * uGain;
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewPosition;
        float pixels = uSizeUm * projectionMatrix[1][1] * uViewportHeight * 0.5 / -viewPosition.z;
        gl_PointSize = clamp(pixels, uMinPx, uMaxPx) * (0.75 + 0.35 * level);
      }`,
    fragmentShader: /* glsl */ `
      varying vec3 vColor;
      void main() {
        vec2 offset = gl_PointCoord - 0.5;
        float d2 = dot(offset, offset);
        if (d2 > 0.25) discard;
        gl_FragColor = vec4(vColor, smoothstep(0.25, 0.03, d2));
      }`,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    blending: THREE.AdditiveBlending,
  });
  const scene = new THREE.Scene();
  const points = new THREE.Points(geometry, material);
  points.frustumCulled = false;
  points.position.copy(offset);
  scene.add(points);

  // Картинка светящихся точек в «длинных» числах (half float): яркость может быть больше 1, ореол это использует.
  const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType });
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), BLOOM.strength, BLOOM.radius, BLOOM.threshold);
  // Складываем со всем, что уже нарисовано. Цвета — как есть, без пересчёта цветового пространства (как в облаке).
  const composite = new FullScreenQuad(new THREE.ShaderMaterial({
    uniforms: { tGlow: { value: target.texture } },
    vertexShader: /* glsl */ `
      varying vec2 vUv;
      void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform sampler2D tGlow;
      varying vec2 vUv;
      void main() { gl_FragColor = vec4(texture2D(tGlow, vUv).rgb, 1.0); }`,
    blending: THREE.AdditiveBlending,
    depthTest: false,
    depthWrite: false,
    transparent: true,
  }));

  let now = 0; // часы показа, с: идут, только пока симуляция не на паузе
  let tau = GLOW_TAU_S;
  let lastSpikeAt = -Infinity;
  const blackColor = new THREE.Color(0, 0, 0);
  const savedColor = new THREE.Color();

  return {
    /** Общие с выбором мышью числа: по ним картинка выбора тоже знает, какие нейроны сейчас светятся. */
    uniforms: { uNow: material.uniforms.uNow, uTau: material.uniforms.uTau },

    /** Яркость свечения (1 — как было в C9): сама точка и ореол вокруг неё. */
    setBrightness(value) {
      material.uniforms.uGain.value = value;
      bloom.strength = BLOOM.strength * value;
    },

    /** «Шлейф»: за сколько секунд показа вспышка гаснет в e ≈ 2,7 раза. */
    setTrail(seconds) {
      tau = seconds;
      material.uniforms.uTau.value = seconds;
    },

    /** Ничего не светится (давно не было спайков). */
    isQuiet() {
      return now - lastSpikeAt > tau * QUIET_AFTER_TAUS;
    },

    /** Продвинуть часы показа (вызывается каждый кадр экрана, пока симуляция идёт). */
    advance(seconds) {
      now += seconds;
      material.uniforms.uNow.value = now;
    },

    /** Нейроны `neurons` выстрелили `spikes` раз: зажечь их. */
    spike(neurons, spikes) {
      for (let k = 0; k < neurons.length; k++) {
        const at = neurons[k] * 2;
        const heat = glow[at] * Math.exp(-(now - glow[at + 1]) / tau);
        glow[at] = Math.min(MAX_HEAT, heat + spikes[k]);
        glow[at + 1] = now;
      }
      if (neurons.length > 0) {
        lastSpikeAt = now;
        attribute.needsUpdate = true;
      }
    },

    /** Погасить всё сразу (сброс модели). */
    clear() {
      glow.fill(0);
      attribute.needsUpdate = true;
      lastSpikeAt = -Infinity;
    },

    resize(width, height, pixelRatio) {
      const w = Math.max(1, Math.round(width * pixelRatio));
      const h = Math.max(1, Math.round(height * pixelRatio));
      target.setSize(w, h);
      // Ореол размытый, поэтому считаем его на картинке вдвое меньше: на встроенной видеокарте это заметно быстрее.
      bloom.setSize(Math.max(1, Math.round(w / BLOOM_DOWNSCALE)), Math.max(1, Math.round(h / BLOOM_DOWNSCALE)));
      material.uniforms.uViewportHeight.value = h;
      material.uniforms.uMinPx.value = 3 * pixelRatio;
      material.uniforms.uMaxPx.value = 14 * pixelRatio;
    },

    /** Нарисовать свечение поверх уже нарисованного кадра. */
    render(renderer, camera) {
      if (this.isQuiet()) return; // ничего не светится — не тратим видеокарту
      const previousTarget = renderer.getRenderTarget();
      const previousAlpha = renderer.getClearAlpha();
      const previousAutoClear = renderer.autoClear;
      renderer.getClearColor(savedColor);

      renderer.setRenderTarget(target);
      renderer.setClearColor(blackColor, 0);
      renderer.clear();
      renderer.render(scene, camera);
      bloom.render(renderer, null, target, 0, false); // ореол добавляется в ту же картинку

      renderer.setRenderTarget(previousTarget);
      renderer.autoClear = false;
      composite.render(renderer);
      renderer.autoClear = previousAutoClear;
      renderer.setClearColor(savedColor, previousAlpha);
    },
  };
}
