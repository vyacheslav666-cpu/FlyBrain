// Форма закреплённого нейрона из скелетов FlyWire (C12b, D-037): каждая точка скелета соединена тонкой линией
// со своим «родителем», получаются ветки нейрона. Координаты те же, что у облака (мкм, X отражён, D-011).
import * as THREE from "three";

const HEADER_BYTES = 16; // uint32 версия, uint32 число точек, float32 расстояние от точки нейрона до формы, запас
const WIRE_VERSION = 1;
const CACHE_SIZE = 12; // столько последних форм держим в памяти: повторный щелчок не ждёт сервер
// Белые и непрозрачные, поверх облака: при сложении с облаком (как у точек) тонкие ветки в нём терялись.
// Белый не путается ни с синими и оранжевыми линиями связей, ни с бирюзовым кольцом закреплённого нейрона.
const COLOR = 0xffffff;
const OPACITY = 0.95;

const cache = new Map();

/** Форма нейрона с сервера: {positions, parents, count, distanceUm}; `null` — формы для него нет. */
export async function getSkeleton(index) {
  if (cache.has(index)) return cache.get(index);
  const response = await fetch(`/api/neuron/${index}/skeleton`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`сервер ответил ${response.status}`);
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  const version = view.getUint32(0, true);
  const count = view.getUint32(4, true);
  if (version !== WIRE_VERSION || buffer.byteLength !== HEADER_BYTES + count * 16) {
    throw new Error("форма нейрона пришла в незнакомом виде");
  }
  const skeleton = {
    count,
    distanceUm: view.getFloat32(8, true),
    positions: new Float32Array(buffer, HEADER_BYTES, count * 3),
    parents: new Int32Array(buffer, HEADER_BYTES + count * 12, count),
  };
  cache.set(index, skeleton);
  if (cache.size > CACHE_SIZE) cache.delete(cache.keys().next().value);
  return skeleton;
}

/** Ветки одного нейрона. `offset` — тот же сдвиг, что у облака (мозг в начале координат). */
export function createSkeletonView(offset) {
  const material = new THREE.LineBasicMaterial({
    color: COLOR,
    transparent: true,
    opacity: OPACITY,
    depthTest: false,
    depthWrite: false,
    blending: THREE.NormalBlending,
  });
  let geometry = new THREE.BufferGeometry();
  const object = new THREE.LineSegments(geometry, material);
  object.position.copy(offset);
  object.frustumCulled = false;
  object.renderOrder = 4; // под линиями связей и кольцами
  object.visible = false;

  return {
    object,
    show(skeleton) {
      const pairs = [];
      skeleton.parents.forEach((parent, child) => { if (parent >= 0) pairs.push(child, parent); });
      geometry.dispose();
      geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(skeleton.positions, 3));
      geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(pairs), 1));
      object.geometry = geometry;
      object.visible = pairs.length > 0;
    },
    clear() {
      object.visible = false;
    },
    setBrightness(value) {
      material.opacity = Math.min(1, OPACITY * value);
    },
  };
}
