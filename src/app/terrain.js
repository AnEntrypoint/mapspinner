import * as THREE from 'three';
import vertexShader from './shaders/terrain.vert?raw';
import fragmentShader from './shaders/terrain.frag?raw';
import bakeVertex from './shaders/terrainBake.vert?raw';
import bakeHeightFrag from './shaders/terrainBakeHeight.frag?raw';
import bakeGeomFinalFrag from './shaders/terrainBakeGeomFinal.frag?raw';
import bakeMatFrag from './shaders/terrainBakeMat.frag?raw';
import bakeAlbedoFrag from './shaders/terrainBakeAlbedo.frag?raw';
import skyVertex from './shaders/terrainSky.vert?raw';
import skyFragment from './shaders/terrainSky.frag?raw';
import waterVertex from './shaders/terrainWater.vert?raw';
import waterFragment from './shaders/terrainWater.frag?raw';
import waterSimFrag from './shaders/waterSim.frag?raw';

const MAX_ANCHORS = 64;
const MAX_BIOMES = 12;
const FALLOFF = 570;
const TERRAIN_HEIGHT = 20;
const POLAR_M = 192;
const POLAR_K = 384;
const POLAR_MAX_R = 2200;
const SAMPLE_ALT = 300;
const BAKE_SIZE = 1024;

export const BIOMES = [
  { name: 'snow_peak',     amp: 42, freq: 0.0021, elevation:  75, continentMix: 0.55, ridgeMix: 0.95, rollMix: 0.18, snowline:  55, rockSlope: 0.40 },
  { name: 'alpine_rock',   amp: 34, freq: 0.0034, elevation:  55, continentMix: 0.45, ridgeMix: 0.85, rollMix: 0.25, snowline:  85, rockSlope: 0.30 },
  { name: 'forest_hills',  amp: 20, freq: 0.0042, elevation:  14, continentMix: 0.75, ridgeMix: 0.0,  rollMix: 0.65, snowline: 140, rockSlope: 0.55 },
  { name: 'grassland',     amp: 11, freq: 0.0028, elevation:   6, continentMix: 0.65, ridgeMix: 0.0,  rollMix: 0.75, snowline: 140, rockSlope: 0.58 },
  { name: 'plains',        amp:  3, freq: 0.0022, elevation:   3, continentMix: 0.45, ridgeMix: 0.0,  rollMix: 0.55, snowline: 160, rockSlope: 0.65 },
  { name: 'desert_dunes',  amp: 16, freq: 0.0026, elevation:   8, continentMix: 0.30, ridgeMix: 0.0,  rollMix: 1.20, snowline: 220, rockSlope: 0.55 },
  { name: 'river_valley',  amp:  3, freq: 0.0150, elevation: -18, continentMix: 0.20, ridgeMix: 0.0,  rollMix: 0.60, snowline: 140, rockSlope: 0.55 },
  { name: 'river_channel', amp:  1, freq: 0.0180, elevation: -28, continentMix: 0.15, ridgeMix: 0.0,  rollMix: 0.40, snowline: 140, rockSlope: 0.55 },
  { name: 'lakeshore',     amp:  4, freq: 0.0038, elevation:  -4, continentMix: 0.30, ridgeMix: 0.0,  rollMix: 0.40, snowline: 220, rockSlope: 0.65 },
];
export const BIOME_INDEX = {};
BIOMES.forEach((b, i) => { BIOME_INDEX[b.name] = i; });
const BIOME_EROSION_STRENGTH = [1.20, 1.30, 0.85, 0.55, 0.45, 0.10, 0.65, 0.50, 0.40];

function permute3(x) {
  const r = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const v = ((x[i] * 34.0) + 1.0) * x[i];
    r[i] = v - Math.floor(v / 289.0) * 289.0;
  }
  return r;
}
function snoise(vx, vy) {
  const Cx = 0.211324865405187, Cy = 0.366025403784439, Cz = -0.577350269189626, Cw = 0.024390243902439;
  const dotVCyy = vx * Cy + vy * Cy;
  let ix = Math.floor(vx + dotVCyy);
  let iy = Math.floor(vy + dotVCyy);
  const dotICxx = ix * Cx + iy * Cx;
  const x0x = vx - ix + dotICxx;
  const x0y = vy - iy + dotICxx;
  const i1x = x0x > x0y ? 1.0 : 0.0;
  const i1y = x0x > x0y ? 0.0 : 1.0;
  const x12x = x0x + Cx - i1x;
  const x12y = x0y + Cx - i1y;
  const x12z = x0x + Cz;
  const x12w = x0y + Cz;
  ix = ix - Math.floor(ix / 289.0) * 289.0;
  iy = iy - Math.floor(iy / 289.0) * 289.0;
  const inner = permute3([iy + 0.0, iy + i1y, iy + 1.0]);
  const p = permute3([inner[0] + ix + 0.0, inner[1] + ix + i1x, inner[2] + ix + 1.0]);
  const px = 2.0 * (p[0] * Cw - Math.floor(p[0] * Cw)) - 1.0;
  const py = 2.0 * (p[1] * Cw - Math.floor(p[1] * Cw)) - 1.0;
  const pz = 2.0 * (p[2] * Cw - Math.floor(p[2] * Cw)) - 1.0;
  const hx = Math.abs(px) - 0.5, hy = Math.abs(py) - 0.5, hz = Math.abs(pz) - 0.5;
  const oxv = Math.floor(px + 0.5), oyv = Math.floor(py + 0.5), ozv = Math.floor(pz + 0.5);
  const a0x = px - oxv, a0y = py - oyv, a0z = pz - ozv;
  const normx = 1.79284291400159 - 0.85373472095314 * (a0x * a0x + hx * hx);
  const normy = 1.79284291400159 - 0.85373472095314 * (a0y * a0y + hy * hy);
  const normz = 1.79284291400159 - 0.85373472095314 * (a0z * a0z + hz * hz);
  const g0x = a0x * normx, g0y = hx * normx;
  const g1x = a0y * normy, g1y = hy * normy;
  const g2x = a0z * normz, g2y = hz * normz;
  const t0 = Math.max(0.5 - (x0x * x0x + x0y * x0y), 0.0);
  const t1 = Math.max(0.5 - (x12x * x12x + x12y * x12y), 0.0);
  const t2 = Math.max(0.5 - (x12z * x12z + x12w * x12w), 0.0);
  const t40 = t0 * t0 * t0 * t0, t41 = t1 * t1 * t1 * t1, t42 = t2 * t2 * t2 * t2;
  const gd0 = g0x * x0x + g0y * x0y;
  const gd1 = g1x * x12x + g1y * x12y;
  const gd2 = g2x * x12z + g2y * x12w;
  return 130.0 * (t40 * gd0 + t41 * gd1 + t42 * gd2);
}
function fbm3(px, py, freq, sx, sy) {
  let h = 0, a = 1, f = freq, ox = sx, oy = sy;
  h += a * snoise(px * f + ox, py * f + oy);
  a *= 0.5; f *= 2; { const tx = oy, ty = -ox; ox = tx + 11.1; oy = ty + 7.3; }
  h += a * snoise(px * f + ox, py * f + oy);
  a *= 0.5; f *= 2; { const tx = oy, ty = -ox; ox = tx + 11.1; oy = ty + 7.3; }
  h += a * snoise(px * f + ox, py * f + oy);
  return h;
}
function fbm2(px, py, freq, sx, sy) {
  let h = 0, a = 1, f = freq, ox = sx, oy = sy;
  h += a * snoise(px * f + ox, py * f + oy);
  a *= 0.5; f *= 2; { const tx = oy, ty = -ox; ox = tx + 11.1; oy = ty + 7.3; }
  h += a * snoise(px * f + ox, py * f + oy);
  return h;
}
function ridgeFbm(px, py, freq, sx, sy) {
  let h = 0, a = 1, f = freq, ox = sx, oy = sy;
  h += a * Math.pow(1 - Math.abs(snoise(px * f + ox, py * f + oy)), 0.7);
  a *= 0.5; f *= 2; { const tx = oy, ty = -ox; ox = tx + 11.1; oy = ty + 7.3; }
  h += a * Math.pow(1 - Math.abs(snoise(px * f + ox, py * f + oy)), 0.7);
  a *= 0.5; f *= 2; { const tx = oy, ty = -ox; ox = tx + 11.1; oy = ty + 7.3; }
  h += a * Math.pow(1 - Math.abs(snoise(px * f + ox, py * f + oy)), 0.7);
  return h - 0.875;
}
function applyErosionDeltaCPU(px, py, bi) {
  const em = fbm3(px, py, 0.005, 3.7, 5.1);
  const ed = fbm2(px, py, 0.02, 8.3, 1.9);
  const channels = (1.0 - Math.abs(em)) * (1.0 - Math.abs(ed)) * 0.7;
  const rough = fbm2(px, py, 0.08, 17.7, 23.1) * 0.10;
  const strength = BIOME_EROSION_STRENGTH[bi] != null ? BIOME_EROSION_STRENGTH[bi] : 0.7;
  return (-channels * 0.15 + rough * 0.15) * strength * 20.0;
}
function biomeContributionCPU(px, py, bi) {
  const b = BIOMES[bi];
  const bf = bi;
  const saltCx = 13.0 + bf * 7.1, saltCy = 29.0 - bf * 3.3;
  const saltRx = 91.0 - bf * 5.7, saltRy = 41.0 + bf * 2.9;
  const saltLx = 57.0 + bf * 11.0, saltLy = 17.0 - bf * 6.4;
  const cont = fbm3(px, py, b.freq, saltCx, saltCy);
  let ridge = ridgeFbm(px, py, b.freq * 2.0, saltRx, saltRy);
  const roll = fbm2(px, py, b.freq * 0.5, saltLx, saltLy);
  const erosion = fbm2(px, py, b.freq * 3.0, saltLx + 31.7, saltLy + 19.1) * 0.3;
  let broadGate = cont * 0.5 + 0.5;
  if (broadGate < 0) broadGate = 0; else if (broadGate > 1) broadGate = 1;
  ridge = ridge * (1.0 - erosion) * broadGate;
  let h = b.elevation + b.amp * (cont * b.continentMix + ridge * b.ridgeMix + roll * b.rollMix);
  h += applyErosionDeltaCPU(px, py, bi);
  return h;
}
function rng(seed) {
  let s = seed | 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; };
}
export function buildContinent() {
  const r = rng(7);
  const A = [];
  const add = (biome, x, z, elev) => A.push({ pos: [x, z], biome, elevation: elev != null ? elev : BIOMES[BIOME_INDEX[biome]].elevation });
  const jitter = (x, z, amt) => [x + (r() - 0.5) * amt * 2, z + (r() - 0.5) * amt * 2];
  for (let i = 0; i < 5; i++) {
    const tt = i / 4;
    const [x, z] = jitter(-700 + tt * 400, -700 + tt * 200, 60);
    add('snow_peak', x, z, 70 + Math.sin(tt * Math.PI) * 30);
  }
  for (let i = 0; i < 4; i++) {
    const tt = i / 3;
    const [x, z] = jitter(-500 + tt * 350, -500 + tt * 350, 80);
    add('alpine_rock', x, z, 50);
  }
  for (let i = 0; i < 6; i++) {
    const ang = (i / 6) * Math.PI * 2;
    const [x, z] = jitter(-200 + Math.cos(ang) * 350, -100 + Math.sin(ang) * 250, 70);
    add('forest_hills', x, z);
  }
  for (let i = 0; i < 6; i++) {
    const ang = (i / 6) * Math.PI * 2 + 0.4;
    const [x, z] = jitter(Math.cos(ang) * 450, Math.sin(ang) * 450, 80);
    add('grassland', x, z);
  }
  for (let i = 0; i < 5; i++) {
    const tt = i / 4;
    const [x, z] = jitter(-400 + tt * 900, 200 + Math.sin(tt * Math.PI * 2) * 120, 50);
    add('river_valley', x, z);
  }
  for (let i = 0; i < 7; i++) {
    const tt = i / 6;
    const [x, z] = jitter(-400 + tt * 900, 200 + Math.sin(tt * Math.PI * 2) * 120, 20);
    add('river_channel', x, z);
  }
  for (let i = 0; i < 4; i++) {
    const ang = r() * Math.PI * 2;
    const d = Math.sqrt(r()) * 350;
    add('plains', 700 + Math.cos(ang) * d, -200 + Math.sin(ang) * d);
  }
  for (let i = 0; i < 4; i++) {
    const ang = r() * Math.PI * 2;
    const d = Math.sqrt(r()) * 300;
    add('desert_dunes', 600 + Math.cos(ang) * d, 600 + Math.sin(ang) * d);
  }
  for (let i = 0; i < 4; i++) {
    const ang = (i / 4) * Math.PI * 2;
    const [x, z] = jitter(-500 + Math.cos(ang) * 180, 500 + Math.sin(ang) * 180, 30);
    add('lakeshore', x, z, -8);
  }
  for (let i = 0; i < 3; i++) {
    const [x, z] = jitter(-200 + i * 200, -1100, 100);
    add('grassland', x, z);
  }
  return A.slice(0, MAX_ANCHORS);
}
export const ANCHORS = buildContinent();
export function anchoredHeightCPU(px, py) {
  const n = ANCHORS.length;
  const ws = new Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const dx = px - ANCHORS[i].pos[0];
    const dy = py - ANCHORS[i].pos[1];
    const d = Math.sqrt(dx * dx + dy * dy);
    const w = Math.exp(-d / FALLOFF);
    ws[i] = w;
    sum += w;
  }
  const inv = 1 / Math.max(sum, 1e-6);
  let y = 0;
  for (let i = 0; i < n; i++) {
    const w = ws[i];
    const bi = BIOME_INDEX[ANCHORS[i].biome];
    y += (w * inv) * (biomeContributionCPU(px, py, bi) + ANCHORS[i].elevation);
  }
  const t = Math.max(0, Math.min(1, (sum - 0.3) / (2.5 - 0.3)));
  const anchorInfluence = t * t * (3 - 2 * t);
  const oceanFloor = -25 + 4 * fbm2(px, py, 0.003, 2.1, 4.7);
  return oceanFloor + (y - oceanFloor) * anchorInfluence;
}
export function heightAt(x, z) {
  return anchoredHeightCPU(x, z) * (TERRAIN_HEIGHT / 20);
}
export function dominantBiomeAt(x, z) {
  let bestW = -1, bestI = 0, sum = 0;
  for (let i = 0; i < ANCHORS.length; i++) {
    const dx = x - ANCHORS[i].pos[0], dy = z - ANCHORS[i].pos[1];
    const d = Math.sqrt(dx * dx + dy * dy);
    const w = Math.exp(-d / FALLOFF);
    sum += w;
    if (w > bestW) { bestW = w; bestI = i; }
  }
  if (sum < 1.0) {
    return { name: 'ocean', index: -1, anchor: null };
  }
  return { name: ANCHORS[bestI].biome, index: BIOME_INDEX[ANCHORS[bestI].biome], anchor: ANCHORS[bestI] };
}
function smoothstepJ(e0, e1, x) {
  const t = Math.max(0, Math.min(1, (x - e0) / (e1 - e0)));
  return t * t * (3 - 2 * t);
}
export function biomeWeightsAt(x, z) {
  const dominant = dominantBiomeAt(x, z);
  const h = heightAt(x, z);
  const eps = 1.5;
  const hxp = heightAt(x + eps, z);
  const hzp = heightAt(x, z + eps);
  const gx = (hxp - h) / eps;
  const gz = (hzp - h) / eps;
  const slope = Math.sqrt(gx * gx + gz * gz) / Math.sqrt(1 + gx * gx + gz * gz);
  const realSlopeForMat = slope * 0.65;
  const b = dominant.index >= 0 ? BIOMES[dominant.index] : { snowline: 200 };
  const effSnowline = b.snowline;
  const grassBand = smoothstepJ(3, 12, h) * (1 - smoothstepJ(effSnowline - 30, effSnowline - 5, h));
  const rockMix = smoothstepJ(0.35, 0.62, realSlopeForMat);
  const snowMix = smoothstepJ(effSnowline - 5, effSnowline + 5, h);
  const forestMix = (dominant.name === 'forest_hills') ? grassBand * 0.8 : grassBand * 0.1;
  return { rock: rockMix, grass: grassBand, snow: snowMix, forest: forestMix, slope, height: h, biome: dominant.name };
}

function ringK(r, K) { return Math.max(8, Math.min(K, 8 << Math.floor(r / 6))); }
function buildPolarGeometry(M, K) {
  const Kr = new Array(M), ringStart = new Array(M);
  let total = 1;
  for (let r = 0; r < M; r++) { Kr[r] = ringK(r, K); ringStart[r] = total; total += Kr[r]; }
  const positions = new Float32Array(total * 3);
  positions[0] = 0; positions[1] = 0; positions[2] = 0;
  for (let r = 0; r < M; r++) {
    const t = (r + 1) / M;
    const Krr = Kr[r];
    for (let k = 0; k < Krr; k++) {
      const idx = ringStart[r] + k;
      const theta = (k / Krr) * Math.PI * 2;
      positions[idx * 3 + 0] = t;
      positions[idx * 3 + 1] = 0;
      positions[idx * 3 + 2] = theta;
    }
  }
  const tris = [];
  const K0 = Kr[0];
  for (let k = 0; k < K0; k++) {
    const a = ringStart[0] + k;
    const b = ringStart[0] + ((k + 1) % K0);
    tris.push(0, b, a);
  }
  for (let r = 0; r < M - 1; r++) {
    const Ka = Kr[r], Kb = Kr[r + 1];
    const sa = ringStart[r], sb = ringStart[r + 1];
    if (Ka === Kb) {
      for (let k = 0; k < Ka; k++) {
        const a = sa + k, b = sa + ((k + 1) % Ka), c = sb + k, d = sb + ((k + 1) % Kb);
        tris.push(a, b, c, b, d, c);
      }
    } else if (Kb > Ka) {
      for (let i = 0; i < Ka; i++) {
        const ai = sa + i, aiNext = sa + ((i + 1) % Ka);
        const kStart = Math.round((i / Ka) * Kb);
        const kEnd = Math.round(((i + 1) / Ka) * Kb);
        for (let k = kStart; k < kEnd; k++) {
          const c = sb + (k % Kb), d = sb + ((k + 1) % Kb);
          tris.push(ai, d, c);
        }
        const dBoundary = sb + (kEnd % Kb);
        tris.push(ai, aiNext, dBoundary);
      }
    } else {
      for (let k = 0; k < Ka; k++) {
        const a = sa + k, b = sa + ((k + 1) % Ka);
        const midAngleIdx = (k + 0.5) / Ka;
        const outerIdx = Math.floor(midAngleIdx * Kb) % Kb;
        const co = sb + outerIdx;
        tris.push(a, b, co);
      }
    }
  }
  const indices = new Uint32Array(tris);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setIndex(new THREE.BufferAttribute(indices, 1));
  geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 1e6);
  geo.userData = { Kr, total, ringStart };
  return geo;
}

function packAnchors() {
  const arr = [];
  for (let i = 0; i < MAX_ANCHORS; i++) {
    if (i < ANCHORS.length) {
      const a = ANCHORS[i];
      arr.push(new THREE.Vector4(a.pos[0], a.pos[1], a.elevation, BIOME_INDEX[a.biome]));
    } else {
      arr.push(new THREE.Vector4(0, 0, 0, 0));
    }
  }
  return arr;
}
function packBiomes() {
  const s0 = [], s1 = [];
  for (let i = 0; i < MAX_BIOMES; i++) {
    if (i < BIOMES.length) {
      const b = BIOMES[i];
      s0.push(new THREE.Vector4(b.amp, b.freq, b.elevation, b.continentMix));
      s1.push(new THREE.Vector4(b.ridgeMix, b.rollMix, b.snowline, b.rockSlope));
    } else {
      s0.push(new THREE.Vector4()); s1.push(new THREE.Vector4());
    }
  }
  return { s0, s1 };
}

export class TerrainSystem {
  constructor(renderer) {
    this.renderer = renderer;
    const biomePack = packBiomes();
    this.uniforms = {
      uSampleCameraPos: { value: new THREE.Vector3(0, SAMPLE_ALT, 0) },
      uHeight: { value: TERRAIN_HEIGHT },
      uConcentration: { value: 3.0 },
      uDebugMode: { value: 0 },
      uAnchorCount: { value: ANCHORS.length },
      uFalloff: { value: FALLOFF },
      uAnchors: { value: packAnchors() },
      uBiomes0: { value: biomePack.s0 },
      uBiomes1: { value: biomePack.s1 },
      uTime: { value: 0 },
      uTimeOfDay: { value: 0.5 },
      uSeason: { value: 0.25 },
      uTGeom: { value: null },
      uTMat: { value: null },
      uMaxR: { value: POLAR_MAX_R },
    };
    this._buildBakeRTs();
    this._buildBakePipeline();
    this.terrainMesh = this._buildTerrain();
    // Inverted-sphere skybox using vox's sky color formula (skyColorDir from
    // terrain.frag). The sphere follows the camera each frame.
    const skyMat = new THREE.ShaderMaterial({
      uniforms: { uTimeOfDay: this.uniforms.uTimeOfDay },
      vertexShader: /* glsl */`
        varying vec3 vDir;
        void main() {
          vDir = normalize(position);
          vec4 worldPos = modelMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * viewMatrix * worldPos;
          gl_Position.z = gl_Position.w;
        }`,
      fragmentShader: /* glsl */`
        precision highp float;
        varying vec3 vDir;
        uniform float uTimeOfDay;
        const float PI=3.14159265359;
        vec3 sunDirFromTOD(float tt){ float el=sin((tt-0.25)*2.0*PI); float az=tt*2.0*PI; float horiz=sqrt(max(1.0-el*el,0.0)); return normalize(vec3(cos(az)*horiz, el, sin(az)*horiz)); }
        void main(){
          vec3 d=normalize(vDir);
          vec3 sd=sunDirFromTOD(uTimeOfDay);
          float dayAmt=smoothstep(-0.05,0.30,sd.y);
          float twiAmt=(1.0-dayAmt)*smoothstep(-0.30,0.05,sd.y);
          float nightAmt=1.0-smoothstep(-0.15,0.05,sd.y);
          float t=clamp(d.y*0.5+0.5,0.0,1.0);
          vec3 dHor=vec3(0.78,0.84,0.92), dMid=vec3(0.55,0.72,0.90), dZen=vec3(0.20,0.42,0.74);
          vec3 day=mix(mix(dHor,dMid,smoothstep(0.0,0.45,t)),dZen,smoothstep(0.45,1.0,t));
          float sunAlign=max(dot(normalize(vec3(d.x,0.0,d.z)),normalize(vec3(sd.x,0.0,sd.z))),0.0);
          vec3 tHor=mix(vec3(0.55,0.32,0.40),vec3(1.05,0.55,0.30),sunAlign);
          vec3 tMid=vec3(0.40,0.32,0.46), tZen=vec3(0.10,0.14,0.32);
          vec3 twi=mix(mix(tHor,tMid,smoothstep(0.0,0.45,t)),tZen,smoothstep(0.45,1.0,t));
          vec3 nHor=vec3(0.04,0.06,0.12), nMid=vec3(0.02,0.03,0.08), nZen=vec3(0.005,0.01,0.04);
          vec3 night=mix(mix(nHor,nMid,smoothstep(0.0,0.45,t)),nZen,smoothstep(0.45,1.0,t));
          gl_FragColor=vec4(day*dayAmt + twi*twiAmt + night*nightAmt, 1.0);
        }`,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
    });
    this.skyMesh = new THREE.Mesh(new THREE.SphereGeometry(1500, 32, 16), skyMat);
    this.skyMesh.renderOrder = -9999;
    this.skyMesh.frustumCulled = false;
    this.waterMesh = this._buildWater();
    this._lastSnap = { x: 9999, z: 9999 };
    this.bakeCount = 0;
    if (typeof window !== 'undefined') {
      window.__terrain = {
        heightAt, biomeWeightsAt, dominantBiomeAt, BIOMES, ANCHORS,
        system: this,
        get bakeCount() { return self.bakeCount; },
      };
      window.__debug = window.__debug || {};
      window.__debug.waterSim = this._waterSim;
    }
    const self = this;
    this._bake();
  }
  _buildBakeRTs() {
    const renderer = this.renderer;
    const isWebGL2 = renderer.capabilities && renderer.capabilities.isWebGL2;
    // Modern three.js (r152+) supports MSAA via WebGLRenderTarget options.samples directly.
    const MSAA_OK = !!isWebGL2;
    const gl = renderer.getContext();
    const HF_EXT = isWebGL2 ? (gl.getExtension('EXT_color_buffer_half_float') || gl.getExtension('EXT_color_buffer_float')) : null;
    const HALF_OK = !!HF_EXT;
    const HF_TYPE = HALF_OK ? THREE.HalfFloatType : THREE.UnsignedByteType;
    const MAX_ANISO = (renderer.capabilities && typeof renderer.capabilities.getMaxAnisotropy === 'function') ? renderer.capabilities.getMaxAnisotropy() : 1;
    const opts8 = { format: THREE.RGBAFormat, type: THREE.UnsignedByteType, minFilter: THREE.LinearMipmapLinearFilter, magFilter: THREE.LinearFilter, wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping, depthBuffer: false, stencilBuffer: false, generateMipmaps: true };
    const optsHF = { format: THREE.RGBAFormat, type: HF_TYPE, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, wrapS: THREE.ClampToEdgeWrapping, wrapT: THREE.ClampToEdgeWrapping, depthBuffer: false, stencilBuffer: false, generateMipmaps: false };
    // MSAA + HalfFloat is unreliable across drivers — disable MSAA for HF render targets (matches vox).
    const HF_MSAA_OK = false;
    const mkRTHF = () => {
      const o = Object.assign({}, optsHF);
      if (HF_MSAA_OK && MSAA_OK) o.samples = 4;
      return new THREE.WebGLRenderTarget(BAKE_SIZE, BAKE_SIZE, o);
    };
    const mkRT8 = () => {
      const o = Object.assign({}, opts8);
      if (MSAA_OK) o.samples = 4;
      return new THREE.WebGLRenderTarget(BAKE_SIZE, BAKE_SIZE, o);
    };
    this.tHeightRT = HALF_OK ? mkRTHF() : mkRT8();
    this.tGeomRT   = HALF_OK ? mkRTHF() : mkRT8();
    this.tMatRT    = HALF_OK ? mkRTHF() : mkRT8();
    this.tAlbedoRT = HALF_OK ? mkRTHF() : mkRT8();
    try { this.tHeightRT.texture.anisotropy = MAX_ANISO; this.tHeightRT.texture.generateMipmaps = true; this.tHeightRT.texture.minFilter = THREE.LinearMipmapLinearFilter; } catch (e) {}
    try { this.tGeomRT.texture.anisotropy = MAX_ANISO; this.tGeomRT.texture.generateMipmaps = true; this.tGeomRT.texture.minFilter = THREE.LinearMipmapLinearFilter; } catch (e) {}
    try { this.tMatRT.texture.anisotropy = MAX_ANISO; this.tMatRT.texture.generateMipmaps = true; this.tMatRT.texture.minFilter = THREE.LinearMipmapLinearFilter; } catch (e) {}
    try { this.tAlbedoRT.texture.anisotropy = MAX_ANISO; this.tAlbedoRT.texture.generateMipmaps = true; this.tAlbedoRT.texture.minFilter = THREE.LinearMipmapLinearFilter; } catch (e) {}
    this.uniforms.uTGeom.value = this.tGeomRT.texture;
    this.uniforms.uTMat.value = this.tMatRT.texture;
    if (typeof window !== 'undefined' && window.__debug && window.__debug.registerRT) {
      window.__debug.registerRT('tHeight', this.tHeightRT);
      window.__debug.registerRT('tGeom', this.tGeomRT);
      window.__debug.registerRT('tMat', this.tMatRT);
      window.__debug.registerRT('tAlbedo', this.tAlbedoRT);
    }
  }
  _buildBakePipeline() {
    this.bakeScene = new THREE.Scene();
    this.bakeCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    const bakeUniforms = {
      uSampleCameraPos: this.uniforms.uSampleCameraPos,
      uMaxR: this.uniforms.uMaxR,
      uConcentration: this.uniforms.uConcentration,
      uHeight: this.uniforms.uHeight,
      uSeason: this.uniforms.uSeason,
      uAnchorCount: this.uniforms.uAnchorCount,
      uFalloff: this.uniforms.uFalloff,
      uAnchors: this.uniforms.uAnchors,
      uBiomes0: this.uniforms.uBiomes0,
      uBiomes1: this.uniforms.uBiomes1,
      uTHeight: { value: this.tHeightRT.texture },
      uBakeSize: { value: BAKE_SIZE },
    };
    this.bakeUniforms = bakeUniforms;
    this.bakeHeightMat = new THREE.ShaderMaterial({ uniforms: bakeUniforms, vertexShader: bakeVertex, fragmentShader: bakeHeightFrag, depthTest: false, depthWrite: false });
    this.bakeGeomFinalMat = new THREE.ShaderMaterial({ uniforms: bakeUniforms, vertexShader: bakeVertex, fragmentShader: bakeGeomFinalFrag, depthTest: false, depthWrite: false });
    this.bakeMatMat = new THREE.ShaderMaterial({ uniforms: bakeUniforms, vertexShader: bakeVertex, fragmentShader: bakeMatFrag, depthTest: false, depthWrite: false });
    const albedoUniforms = Object.assign({}, bakeUniforms, { uTMat: { value: this.tMatRT.texture } });
    this.bakeAlbedoUniforms = albedoUniforms;
    this.bakeAlbedoMat = new THREE.ShaderMaterial({ uniforms: albedoUniforms, vertexShader: bakeVertex, fragmentShader: bakeAlbedoFrag, depthTest: false, depthWrite: false });
    this.bakeQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.bakeHeightMat);
    this.bakeScene.add(this.bakeQuad);
  }
  _buildTerrain() {
    const geo = buildPolarGeometry(POLAR_M, POLAR_K);
    const mat = new THREE.ShaderMaterial({
      uniforms: this.uniforms,
      vertexShader,
      fragmentShader,
      extensions: { derivatives: true },
      defines: {},
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false;
    return mesh;
  }
  _buildWater() {
    // ---- Wave-equation sim (ping-pong) per madebyevan.com/webgl-water ----
    const SIM_SIZE = 256;
    const SIM_WORLD_SIZE = 400; // metres of water plane covered by the tile
    const isWebGL2 = this.renderer.capabilities && this.renderer.capabilities.isWebGL2;
    const gl = this.renderer.getContext();
    const HF_EXT = isWebGL2 ? (gl.getExtension('EXT_color_buffer_half_float') || gl.getExtension('EXT_color_buffer_float')) : null;
    const SIM_TYPE = HF_EXT ? THREE.HalfFloatType : THREE.UnsignedByteType;
    const simRTOpts = {
      format: THREE.RGBAFormat,
      type: SIM_TYPE,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      wrapS: THREE.RepeatWrapping,
      wrapT: THREE.RepeatWrapping,
      depthBuffer: false,
      stencilBuffer: false,
      generateMipmaps: false,
    };
    const simA = new THREE.WebGLRenderTarget(SIM_SIZE, SIM_SIZE, simRTOpts);
    const simB = new THREE.WebGLRenderTarget(SIM_SIZE, SIM_SIZE, simRTOpts);
    if (typeof window !== 'undefined' && window.__debug && window.__debug.registerRT) {
      window.__debug.registerRT('waterSimA', simA);
      window.__debug.registerRT('waterSimB', simB);
    }
    const simUniforms = {
      uPrev: { value: simA.texture },
      uTexel: { value: new THREE.Vector2(1 / SIM_SIZE, 1 / SIM_SIZE) },
      uDamping: { value: 0.997 },
      uImpulse: { value: new THREE.Vector4(0.5, 0.5, 0.04, 0.0) },
    };
    const simMat = new THREE.ShaderMaterial({
      uniforms: simUniforms,
      vertexShader: bakeVertex,
      fragmentShader: waterSimFrag,
      depthTest: false,
      depthWrite: false,
    });
    const simScene = new THREE.Scene();
    const simQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), simMat);
    simScene.add(simQuad);
    const simCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this._waterSim = {
      size: SIM_SIZE,
      worldSize: SIM_WORLD_SIZE,
      worldCenter: new THREE.Vector2(0, 0),
      rtA: simA,
      rtB: simB,
      readIsA: true,
      uniforms: simUniforms,
      scene: simScene,
      cam: simCam,
      mat: simMat,
      step: 0,
      nextImpulseAt: 0.5,
    };
    // Display mesh: PlaneGeometry segmented for vertex displacement potential later;
    // V1 uses fragment-only normal perturbation, so 1x1 is fine.
    const geo = new THREE.PlaneGeometry(4000, 4000, 1, 1);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: this.uniforms.uTime,
        uTimeOfDay: this.uniforms.uTimeOfDay,
        uSeason: this.uniforms.uSeason,
        uHeight: this.uniforms.uHeight,
        uAnchorCount: this.uniforms.uAnchorCount,
        uFalloff: this.uniforms.uFalloff,
        uAnchors: this.uniforms.uAnchors,
        uBiomes0: this.uniforms.uBiomes0,
        uBiomes1: this.uniforms.uBiomes1,
        uWaterSim: { value: simA.texture },
        uWaterSimCenter: { value: this._waterSim.worldCenter },
        uWaterSimTexel: { value: new THREE.Vector2(1 / SIM_SIZE, 1 / SIM_SIZE) },
        uWaterSimWorldSize: { value: SIM_WORLD_SIZE },
        uWaterSimAmp: { value: 18.0 },
      },
      vertexShader: waterVertex,
      fragmentShader: waterFragment,
      transparent: true,
      depthWrite: false,
    });
    this._waterDisplayMat = mat;
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = 0;
    mesh.frustumCulled = false;
    return mesh;
  }
  _stepWaterSim() {
    const r = this.renderer;
    const W = this._waterSim;
    const read = W.readIsA ? W.rtA : W.rtB;
    const write = W.readIsA ? W.rtB : W.rtA;
    W.uniforms.uPrev.value = read.texture;
    const oldTarget = r.getRenderTarget();
    r.setRenderTarget(write);
    r.render(W.scene, W.cam);
    r.setRenderTarget(oldTarget);
    W.readIsA = !W.readIsA;
    W.step++;
    // After each step, drop the impulse strength back to zero so it only fires once.
    W.uniforms.uImpulse.value.w = 0;
    // The display shader samples the just-written texture.
    this._waterDisplayMat.uniforms.uWaterSim.value = write.texture;
  }
  _scheduleWaterImpulses(time) {
    const W = this._waterSim;
    if (time < W.nextImpulseAt) return;
    // Random uv inside [0.05, 0.95] so impulse is fully inside tile.
    const u = 0.05 + Math.random() * 0.9;
    const v = 0.05 + Math.random() * 0.9;
    const radius = 0.012 + Math.random() * 0.012;     // uv-units
    const strength = 0.06 + Math.random() * 0.08;
    W.uniforms.uImpulse.value.set(u, v, radius, strength);
    W.nextImpulseAt = time + (1.5 + Math.random() * 2.5);
  }
  _bake() {
    const r = this.renderer;
    const oldTarget = r.getRenderTarget();
    this.bakeQuad.material = this.bakeHeightMat;
    r.setRenderTarget(this.tHeightRT); r.clear(); r.render(this.bakeScene, this.bakeCam);
    this.bakeQuad.material = this.bakeGeomFinalMat;
    r.setRenderTarget(this.tGeomRT); r.clear(); r.render(this.bakeScene, this.bakeCam);
    this.bakeQuad.material = this.bakeMatMat;
    r.setRenderTarget(this.tMatRT); r.clear(); r.render(this.bakeScene, this.bakeCam);
    this.bakeAlbedoUniforms.uTMat.value = this.tMatRT.texture;
    this.bakeQuad.material = this.bakeAlbedoMat;
    r.setRenderTarget(this.tAlbedoRT); r.clear(); r.render(this.bakeScene, this.bakeCam);
    r.setRenderTarget(oldTarget);
    this.bakeCount++;
  }
  update(camera, time) {
    this.uniforms.uTime.value = time;
    if (this._waterSim) {
      this._scheduleWaterImpulses(time);
      this._stepWaterSim();
    }
    if (!camera) return;
    this.skyMesh.position.copy(camera.position);
    const tx = camera.position.x, tz = camera.position.z;
    const snap = 10;
    const sx = Math.floor(tx / snap + 0.5) * snap;
    const sz = Math.floor(tz / snap + 0.5) * snap;
    if (sx !== this._lastSnap.x || sz !== this._lastSnap.z) {
      this._lastSnap.x = sx; this._lastSnap.z = sz;
      this.uniforms.uSampleCameraPos.value.set(sx, SAMPLE_ALT, sz);
      this._bake();
    }
  }
  renderSky(renderer) {
    renderer.render(this.skyScene, this.skyCam);
  }
  getHeightTexture() { return this.tHeightRT.texture; }
  getMatTexture() { return this.tMatRT.texture; }
  getAlbedoTexture() { return this.tAlbedoRT.texture; }
  getBakeUniforms() {
    return {
      uSampleCameraPos: this.uniforms.uSampleCameraPos,
      uMaxR: this.uniforms.uMaxR,
      uConcentration: this.uniforms.uConcentration,
      uAnchors: this.uniforms.uAnchors,
      uBiomes0: this.uniforms.uBiomes0,
      uBiomes1: this.uniforms.uBiomes1,
      uAnchorCount: this.uniforms.uAnchorCount,
      uFalloff: this.uniforms.uFalloff,
      uHeight: this.uniforms.uHeight,
    };
  }
}

if (typeof window !== 'undefined' && !window.__terrain) {
  window.__terrain = { heightAt, biomeWeightsAt, dominantBiomeAt, BIOMES, ANCHORS };
}
