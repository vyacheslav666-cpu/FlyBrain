// Кольца вокруг выбранных нейронов и тонкие линии связей. Все объекты лежат в тех же координатах, что и облако.
import * as THREE from "three";

/** Набор колец одного цвета и размера (в пикселях экрана). `set` принимает плоский массив x, y, z, x, y, z… */
export function createRings({ color, sizePx, alpha = 1 }) {
  const material = new THREE.ShaderMaterial({
    uniforms: {
      // цвет задан уже в цветах экрана: свой шейдер three.js не пересчитывает
      uColor: { value: new THREE.Color().setHex(color, THREE.LinearSRGBColorSpace) },
      uSize: { value: sizePx },
      uAlpha: { value: alpha },
    },
    vertexShader: /* glsl */ `
      uniform float uSize;
      void main() {
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = uSize;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uAlpha;
      void main() {
        float d = length(gl_PointCoord - 0.5) * 2.0;
        float ring = smoothstep(0.70, 0.80, d) * (1.0 - smoothstep(0.90, 1.0, d));
        if (ring < 0.02) discard;
        gl_FragColor = vec4(uColor, ring * uAlpha);
      }`,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const geometry = new THREE.BufferGeometry();
  const object = new THREE.Points(geometry, material);
  object.frustumCulled = false;
  object.renderOrder = 10;
  object.visible = false;

  return {
    object,
    set(coordinates) {
      geometry.setAttribute("position", new THREE.BufferAttribute(Float32Array.from(coordinates), 3));
      object.visible = coordinates.length > 0;
    },
    setPixelRatio(ratio) {
      material.uniforms.uSize.value = sizePx * ratio;
    },
  };
}

/** Тонкие линии (толщина в WebGL всегда 1 пиксель). У каждой линии свой цвет: яркость зависит от силы связи. */
export function createLinks() {
  const material = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const geometry = new THREE.BufferGeometry();
  const object = new THREE.LineSegments(geometry, material);
  object.frustumCulled = false;
  object.renderOrder = 5;
  object.visible = false;

  return {
    object,
    /** `lines` — список {from: [x, y, z], to: [x, y, z], color: THREE.Color, strength: 0…1}. */
    set(lines) {
      const positions = new Float32Array(lines.length * 6);
      const colors = new Float32Array(lines.length * 6);
      lines.forEach((line, n) => {
        positions.set(line.from, n * 6);
        positions.set(line.to, n * 6 + 3);
        const brightness = 0.45 + 0.55 * line.strength;
        for (const end of [0, 3]) {
          colors[n * 6 + end] = line.color.r * brightness;
          colors[n * 6 + end + 1] = line.color.g * brightness;
          colors[n * 6 + end + 2] = line.color.b * brightness;
        }
      });
      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      object.visible = lines.length > 0;
    },
  };
}
