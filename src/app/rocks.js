import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/Addons.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { heightAt, biomeWeightsAt } from './terrain';

let loaded = false;
let _rock1Mesh = null;
let _rock2Mesh = null;
let _rock3Mesh = null;

async function fetchAssets() {
  if (loaded) return;
  const gltfLoader = new GLTFLoader();
  const dracoLoader = new DRACOLoader();
  dracoLoader.setDecoderPath('https://www.gstatic.com/draco/versioned/decoders/1.5.7/');
  gltfLoader.setDRACOLoader(dracoLoader);
  _rock1Mesh = (await gltfLoader.loadAsync('rock1.glb')).scene.children[0];
  _rock2Mesh = (await gltfLoader.loadAsync('rock2.glb')).scene.children[0];
  _rock3Mesh = (await gltfLoader.loadAsync('rock3.glb')).scene.children[0];
  loaded = true;
}

export class RockOptions {
  size = { x: 2, y: 2, z: 2 };
  sizeVariation = { x: 3, y: 3, z: 3 };
}

const ROCK_CELL = 40;
const ROCK_RADIUS_CELLS = 21;
const ROCK_PER_CELL = 8;
// Cheap 2D value noise [0,1] for density modulation; mirrors shader vnoise(wp*0.05).
function _hash21(x, y) {
  let h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return h - Math.floor(h);
}
function _vnoise2(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const a = _hash21(ix, iy);
  const b = _hash21(ix + 1, iy);
  const c = _hash21(ix, iy + 1);
  const d = _hash21(ix + 1, iy + 1);
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  return (a * (1 - ux) + b * ux) * (1 - uy) + (c * (1 - ux) + d * ux) * uy;
}
const ROCK_MAX = ROCK_PER_CELL * (ROCK_RADIUS_CELLS * 2 + 1) * (ROCK_RADIUS_CELLS * 2 + 1);
const FRAME_BUDGET_MS = 3;
const MAX_BUILDS_PER_FRAME = 1;
const _ZERO_MATRIX = new THREE.Matrix4().makeScale(0, 0, 0);

function scoreRock(b, w) {
  let score = 0.0;
  if (b === 'snow_peak' || b === 'alpine_rock') score = 0.95;
  else if (b === 'desert_dunes') score = 0.25;
  else if (b === 'river_channel') score = 0.20;
  else if (b === 'forest_hills') score = 0.18;
  else if (b === 'grassland' || b === 'plains') score = 0.08;
  else if (b === 'river_valley' || b === 'lakeshore') score = 0.10;
  else if (b === 'floodplain') score = 0.05;
  return Math.max(score, w.rock * 1.0, Math.min(1, w.slope * 1.4));
}

export class Rocks extends THREE.Group {
  constructor(options = new RockOptions()) {
    super();
    this.options = options;
    this.streams = [];
    this.lastCx = 9999; this.lastCz = 9999;
    this._cellCache = new Map();
    fetchAssets().then(() => {
      [_rock1Mesh, _rock2Mesh, _rock3Mesh].forEach((mesh, idx) => {
        // Clone material so per-instance color tints don't leak across kinds; ensure vertexColors on.
        const mat = mesh.material.clone();
        mat.vertexColors = true;
        // Lighten rock base — source GLB textures are quite dark. Push base color
        // brighter so instance-color multiplier (≤1) lands in a visible mid-gray
        // range instead of crushed-black. Also dial roughness down slightly so
        // sun catches the surface rather than absorbing entirely.
        // Source GLB has a dark map texture multiplied by mat.color. Pushing the
        // color tint well above 1 brightens the visible mid-grey range by lifting
        // the dark map values into a visible band. roughness=1 + envMapIntensity=0
        // kills specular hot-spots (the "shiny" the user reported).
        if (mat.color) mat.color.setRGB(2.6, 2.5, 2.3);
        if (typeof mat.roughness === 'number') mat.roughness = 1.0;
        if (typeof mat.metalness === 'number') mat.metalness = 0.0;
        if (typeof mat.envMapIntensity === 'number') mat.envMapIntensity = 0.0;
        if (mat.emissive) { mat.emissive.setRGB(0.05, 0.05, 0.05); mat.emissiveIntensity = 0.4; }
        // Force material to NOT use the dark ao baked into the GLB texture's blue channel.
        mat.aoMap = null;
        const inst = new THREE.InstancedMesh(mesh.geometry, mat, ROCK_MAX);
        inst.count = 0;
        inst.castShadow = true;
        inst.frustumCulled = false;
        // Allocate instanceColor buffer so setColorAt works from frame 1.
        const colors = new Float32Array(ROCK_MAX * 3);
        for (let k = 0; k < colors.length; k++) colors[k] = 1.0;
        inst.instanceColor = new THREE.InstancedBufferAttribute(colors, 3);
        // Per-stream slot bookkeeping. cellSlots: cellKey -> number[] of slot indexes.
        // freeSlots: stack of slot indexes available for reuse. highWater: count for inst.count.
        this.streams.push({
          inst, kindIdx: idx, salt: 0x9e3779b9 ^ (idx * 0x85ebca6b),
          cellSlots: new Map(), freeSlots: [], highWater: 0, dirty: false,
          addQ: [], removeQ: [],
        });
        this.add(inst);
      });
      this._wantedKeys = new Set();
      this._scheduleDiff(0, 0);
    });
  }

  update(elapsedTime, camera) {
    if (!camera || this.streams.length === 0) return;
    const cx = Math.floor(camera.position.x / ROCK_CELL);
    const cz = Math.floor(camera.position.z / ROCK_CELL);
    if (cx !== this.lastCx || cz !== this.lastCz) {
      this.lastCx = cx; this.lastCz = cz;
      this._scheduleDiff(cx, cz);
    }
    this._drainQueues();
  }

  _cellEntries(cx, cz, salt, kindIdx) {
    const key = `${cx},${cz},${kindIdx}`;
    const hit = this._cellCache.get(key);
    if (hit) return hit;
    let seed = ((cx * 73856093) ^ (cz * 19349663) ^ salt) >>> 0;
    const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 0x100000000; };
    const out = [];
    const dummy = new THREE.Object3D();
    const tmpColor = new THREE.Color();
    // 2x candidates per cell, then halve via density gate — net same screen
    // count but stronger clustering in high-density zones.
    for (let i = 0; i < ROCK_PER_CELL; i++) {
      const lx = (cx + rnd()) * ROCK_CELL;
      const lz = (cz + rnd()) * ROCK_CELL;
      const w = biomeWeightsAt(lx, lz);
      if (w.height < 0.5) continue;
      const score = scoreRock(w.biome, w);
      // Density noise field — same low-freq 0.05 as terrain shader AO + tree placement.
      const densityNoise = _vnoise2(lx * 0.05, lz * 0.05);
      // Anti-tree gate: trees prefer high density patches; rocks must avoid them.
      // _densityNoise (scene.js) and _vnoise2 (here) share hash + smoothstep, identical field.
      if (densityNoise > 0.55) continue;
      const densityScale = 0.25 / (1.0 + densityNoise * 9.0);
      if (rnd() > score * densityNoise * densityScale) continue;
      const isCliff = (w.biome === 'alpine_rock' || w.biome === 'snow_peak' || w.slope > 0.4);
      // Size scales with rock-weight × density: dense rocky zones up to ~16x base.
      const densityBoost = 1.0 + w.rock * 8.0 * densityNoise;
      // Doubled cap: cliff 6.0+rnd*9.6 (peak 21.6), open 1.6+rnd*5.6 (peak 8.8).
      const sizeMul = (isCliff ? (6.0 + rnd() * 9.6) : (1.6 + rnd() * 5.6)) * densityBoost;
      const sx = (this.options.sizeVariation.x * rnd() + this.options.size.x) * sizeMul;
      const sy = (this.options.sizeVariation.y * rnd() + this.options.size.y) * sizeMul;
      const sz = (this.options.sizeVariation.z * rnd() + this.options.size.z) * sizeMul;
      // Sink proportional to max scale so big rocks embed deep into ground.
      const sink = Math.max(sx, sy, sz) * 0.4;
      dummy.position.set(lx, heightAt(lx, lz) + 0.3 - sink, lz);
      dummy.rotation.set(0, rnd() * Math.PI * 2, 0);
      dummy.scale.set(sx, sy, sz);
      dummy.updateMatrix();
      // Tight, lighter, less-saturated palette: warm-gray, near-white.
      const hue = 0.06 + (rnd() - 0.5) * 0.04;        // narrow warm-gray band
      const sat = 0.06 + rnd() * 0.10;                // very low saturation
      const light = 0.55 + rnd() * 0.30;              // lighter, max ~0.85
      tmpColor.setHSL(hue, sat, light);
      out.push({ matrix: dummy.matrix.clone(), color: tmpColor.clone() });
    }
    this._cellCache.set(key, out);
    return out;
  }

  _scheduleDiff(camCx, camCz) {
    const wanted = new Set();
    for (let dz = -ROCK_RADIUS_CELLS; dz <= ROCK_RADIUS_CELLS; dz++) {
      for (let dx = -ROCK_RADIUS_CELLS; dx <= ROCK_RADIUS_CELLS; dx++) {
        wanted.add(`${camCx + dx},${camCz + dz}`);
      }
    }
    for (const stream of this.streams) {
      // Removed = current cells not in wanted.
      for (const cellKey of stream.cellSlots.keys()) {
        if (!wanted.has(cellKey)) stream.removeQ.push(cellKey);
      }
      // Added = wanted cells not yet placed.
      for (const cellKey of wanted) {
        if (!stream.cellSlots.has(cellKey) && !stream.addQ.includes(cellKey)) {
          stream.addQ.push(cellKey);
        }
      }
    }
    // Bound cache memory.
    if (this._cellCache.size > 4000) {
      const keep = ROCK_RADIUS_CELLS + 4;
      for (const k of this._cellCache.keys()) {
        const [cx, cz] = k.split(',').map(Number);
        if (Math.abs(cx - camCx) > keep || Math.abs(cz - camCz) > keep) this._cellCache.delete(k);
      }
    }
  }

  _drainQueues() {
    const t0 = performance.now();
    let totalBuilds = 0;
    // Round-robin: drain across streams to share the global budget.
    let any = true;
    while (any && totalBuilds < MAX_BUILDS_PER_FRAME && (performance.now() - t0) < FRAME_BUDGET_MS) {
      any = false;
      for (const stream of this.streams) {
        if ((performance.now() - t0) >= FRAME_BUDGET_MS) break;
        // Removes (cheap)
        while (stream.removeQ.length && (performance.now() - t0) < FRAME_BUDGET_MS) {
          const cellKey = stream.removeQ.shift();
          const slots = stream.cellSlots.get(cellKey);
          if (!slots) continue;
          for (const slot of slots) {
            stream.inst.setMatrixAt(slot, _ZERO_MATRIX);
            stream.freeSlots.push(slot);
          }
          stream.cellSlots.delete(cellKey);
          stream.dirty = true;
          any = true;
        }
        // One add per stream per pass.
        if (stream.addQ.length && totalBuilds < MAX_BUILDS_PER_FRAME) {
          const cellKey = stream.addQ.shift();
          if (!stream.cellSlots.has(cellKey)) {
            const [cx, cz] = cellKey.split(',').map(Number);
            const entries = this._cellEntries(cx, cz, stream.salt, stream.kindIdx);
            const slots = [];
            for (const e of entries) {
              let slot;
              if (stream.freeSlots.length) slot = stream.freeSlots.pop();
              else if (stream.highWater < ROCK_MAX) slot = stream.highWater++;
              else break;
              stream.inst.setMatrixAt(slot, e.matrix);
              if (e.color) stream.inst.setColorAt(slot, e.color);
              slots.push(slot);
            }
            stream.cellSlots.set(cellKey, slots);
            stream.dirty = true;
            totalBuilds++;
            any = true;
          }
        }
      }
    }
    for (const stream of this.streams) {
      if (stream.dirty) {
        stream.inst.count = stream.highWater;
        stream.inst.instanceMatrix.needsUpdate = true;
        if (stream.inst.instanceColor) stream.inst.instanceColor.needsUpdate = true;
        stream.dirty = false;
      }
    }
  }
}
