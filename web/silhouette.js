// Полупрозрачный силуэт мозга (C12, D-036): сетка из shellmesh.js, у краёв светлее (эффект «стекла»).
import * as THREE from "three";
import { NO_POSITION } from "./constants.js";
import { buildShellMesh } from "./shellmesh.js";

const COLOR = new THREE.Vector3(0.45, 0.62, 1.0);
const OPACITY = 0.42;

/** Возвращает `{ object, info, setVisible, setBrightness }`; `offset` — тот же сдвиг, что у облака. */
export function createSilhouette(positions, classes, offset) {
  const mesh = buildShellMesh(positions, classes, NO_POSITION);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(mesh.vertices, 3));
  geometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  const material = new THREE.ShaderMaterial({
    uniforms: { uColor: { value: COLOR }, uOpacity: { value: OPACITY } },
    vertexShader: /* glsl */ `
      varying vec3 vNormal;
      varying vec3 vToCamera;
      void main() {
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vToCamera = normalize(-viewPosition.xyz);
        gl_Position = projectionMatrix * viewPosition;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      uniform float uOpacity;
      varying vec3 vNormal;
      varying vec3 vToCamera;
      void main() {
        // Сторона сетки не важна: смотрим только, насколько поверхность «боком» к взгляду.
        float facing = abs(dot(normalize(vNormal), normalize(vToCamera)));
        // Потолок яркости нужен для стен, которые видны ровно «ребром»: без него они рисовались бы яркими нитями.
        float rim = min(pow(1.0 - facing, 2.2), 0.6);
        gl_FragColor = vec4(uColor, uOpacity * (0.05 + rim));
      }`,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });
  const object = new THREE.Mesh(geometry, material);
  object.frustumCulled = false;
  object.renderOrder = -1;
  object.position.copy(offset);

  return {
    object,
    info: mesh.info,
    setVisible(visible) { object.visible = visible; },
    setBrightness(value) { material.uniforms.uOpacity.value = OPACITY * value; },
  };
}
