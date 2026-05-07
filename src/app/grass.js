/*
 * GPU-procedural grass — IMPLEMENTED.
 *
 * Single InstancedMesh of 20000 blades. instanceMatrix is identity, set ONCE.
 * Vertex shader computes per-blade world position from gl_InstanceID, samples
 * the terrain bake (tHeight, tMat) for height + grass mask, then applies the
 * existing wind-sway logic. JS each frame writes ONE uniform: uCameraXZ.
 *
 * Flowers remain CPU-streamed (low instance count, simpler).
 */
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/Addons.js';
import { biomeWeightsAt, dominantBiomeAt, heightAt } from './terrain';

const _ZERO_MATRIX_GRASS = new THREE.Matrix4().makeScale(0, 0, 0);

let loaded = false;
let _grassMesh = null;
let _blueFlower = null;
let _whiteFlower = null;
let _yellowFlower = null;

const _LOW_MEM = (typeof window !== 'undefined') && (
  window.__lowMem === true ||
  (typeof navigator !== 'undefined' && typeof navigator.deviceMemory === 'number' && navigator.deviceMemory < 4)
);
const _GRASS_INSTANCE_COUNT = _LOW_MEM ? 12000 : 20000;

export class GrassOptions {
  instanceCount = _GRASS_INSTANCE_COUNT;
  maxInstanceCount = _GRASS_INSTANCE_COUNT;
  flowerCount = 50;
  scale = 100;
  patchiness = 0.7;
  size = { x: 5, y: 4, z: 5 };
  sizeVariation = { x: 1, y: 2, z: 1 };
  windStrength = { x: 0.3, y: 0, z: 0.3 };
  windFrequency = 1.0;
  windScale = 400.0;
}

export class Grass extends THREE.Object3D {
  constructor(options = new GrassOptions(), terrainSystem = null) {
    super();
    this.options = options;
    this.terrainSystem = terrainSystem;
    this.flowers = new THREE.Group();
    this.add(this.flowers);
    this._shaders = [];
    this._grassShader = null;
    this._uCameraXZ = { value: new THREE.Vector2(0, 0) };
    this._uViewForwardXZ = { value: new THREE.Vector2(1, 0) };
    this._tmpForward = new THREE.Vector3();

    this.fetchAssets().then(() => {
      this.generateGrass();
    });
  }

  get instanceCount() { return this.grassMesh?.count ?? this.options.instanceCount; }
  set instanceCount(v) { this.grassMesh.count = v; }

  async fetchAssets() {
    if (loaded) return;
    const gltfLoader = new GLTFLoader();
    _grassMesh = (await gltfLoader.loadAsync('grass.glb')).scene.children[0];
    _whiteFlower = (await gltfLoader.loadAsync('flower_white.glb')).scene.children[0];
    _blueFlower = (await gltfLoader.loadAsync('flower_blue.glb')).scene.children[0];
    _yellowFlower = (await gltfLoader.loadAsync('flower_yellow.glb')).scene.children[0];
    [_whiteFlower, _blueFlower, _yellowFlower].forEach((mesh) => {
      mesh.traverse((o) => {
        if (o.isMesh && o.material) {
          if (o.material.map) {
            o.material = new THREE.MeshPhongMaterial({
              map: o.material.map,
              alphaTest: 0.5,
              transparent: false,
              depthWrite: true,
              depthTest: true,
              side: THREE.DoubleSide,
            });
          } else {
            o.visible = false;
          }
          this.appendFlowerWindShader(o.material);
        }
      });
    });
    loaded = true;
  }

  update(elapsedTime, camera, centerXZ) {
    for (let i = 0; i < this._shaders.length; i++) {
      const sh = this._shaders[i];
      if (sh && sh.uniforms && sh.uniforms.uTime) sh.uniforms.uTime.value = elapsedTime;
    }
    if (camera) {
      this._uCameraXZ.value.set(camera.position.x, camera.position.z);
      camera.getWorldDirection(this._tmpForward);
      const fx = this._tmpForward.x, fz = this._tmpForward.z;
      const flen = Math.hypot(fx, fz);
      if (flen > 1e-4) this._uViewForwardXZ.value.set(fx / flen, fz / flen);
    }
    if (camera && this._flowerStreamReady) this._streamFlowers(camera);
    if (this._flowerStreamReady) this._drainFlowerQueue();
  }

  generateGrass() {
    const grassMaterial = new THREE.MeshBasicMaterial({
      map: _grassMesh.material.map,
      transparent: true,
      alphaTest: 0.15,
      depthTest: true,
      depthWrite: true,
      side: THREE.DoubleSide,
    });
    this.appendGrassProceduralShader(grassMaterial);
    this.grassMaterial = grassMaterial;

    const N = this.options.instanceCount;
    this.grassMesh = new THREE.InstancedMesh(_grassMesh.geometry, grassMaterial, N);
    this.grassMesh.count = N;
    this.grassMesh.frustumCulled = false;
    this.grassMesh.receiveShadow = true;
    this.grassMesh.castShadow = true;
    const id = new THREE.Matrix4();
    for (let i = 0; i < N; i++) this.grassMesh.setMatrixAt(i, id);
    this.grassMesh.instanceMatrix.needsUpdate = true;
    this.grassMesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    this.add(this.grassMesh);
  }

  appendGrassProceduralShader(material) {
    const t = this.terrainSystem;
    const bake = t ? t.getBakeUniforms() : null;
    material.onBeforeCompile = (shader) => {
      shader.uniforms.uTime = { value: 0 };
      shader.uniforms.uWindStrength = { value: this.options.windStrength };
      shader.uniforms.uWindFrequency = { value: this.options.windFrequency };
      shader.uniforms.uWindScale = { value: this.options.windScale };
      shader.uniforms.uCameraXZ = this._uCameraXZ;
      shader.uniforms.uViewForwardXZ = this._uViewForwardXZ;
      shader.uniforms.uTHeight = { value: t ? t.getHeightTexture() : null };
      shader.uniforms.uTMat = { value: t ? t.getMatTexture() : null };
      shader.uniforms.uTAlbedo = { value: t && t.getAlbedoTexture ? t.getAlbedoTexture() : null };
      shader.uniforms.uSampleCameraPos = bake ? bake.uSampleCameraPos : { value: new THREE.Vector3() };
      shader.uniforms.uMaxR = bake ? bake.uMaxR : { value: 800.0 };
      shader.uniforms.uConcentration = bake ? bake.uConcentration : { value: 3.0 };
      shader.uniforms.uAnchors = bake && bake.uAnchors ? bake.uAnchors : { value: [] };
      shader.uniforms.uBiomes0 = bake && bake.uBiomes0 ? bake.uBiomes0 : { value: [] };
      shader.uniforms.uBiomes1 = bake && bake.uBiomes1 ? bake.uBiomes1 : { value: [] };
      shader.uniforms.uAnchorCount = bake && bake.uAnchorCount ? bake.uAnchorCount : { value: 0 };
      shader.uniforms.uFalloff = bake && bake.uFalloff ? bake.uFalloff : { value: 570.0 };
      shader.uniforms.uHeightAmp = bake && bake.uHeight ? bake.uHeight : { value: 20.0 };

      shader.vertexShader = `
        #define MAX_ANCHORS_G 64
        #define MAX_BIOMES_G 12
        uniform float uTime;
        uniform vec3 uWindStrength;
        uniform float uWindFrequency;
        uniform float uWindScale;
        uniform vec2 uCameraXZ;
        uniform vec2 uViewForwardXZ;
        uniform vec3 uSampleCameraPos;
        uniform float uMaxR;
        uniform float uConcentration;
        uniform sampler2D uTHeight;
        uniform sampler2D uTMat;
        uniform sampler2D uTAlbedo;
        uniform vec4 uAnchors[MAX_ANCHORS_G];
        uniform vec4 uBiomes0[MAX_BIOMES_G];
        uniform vec4 uBiomes1[MAX_BIOMES_G];
        uniform int uAnchorCount;
        uniform float uFalloff;
        uniform float uHeightAmp;
        varying vec3 vBladeColor;
        varying float vDistFade;
      ` + shader.vertexShader;

      shader.vertexShader = shader.vertexShader.replace(`void main() {`, `
        vec3 permute3T(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
        float snoiseT(vec2 v) {
          const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
          vec2 i  = floor(v + dot(v, C.yy));
          vec2 x0 = v - i + dot(i, C.xx);
          vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
          vec4 x12 = x0.xyxy + C.xxzz; x12.xy -= i1;
          i = mod(i, 289.0);
          vec3 p = permute3T(permute3T(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
          vec3 x = 2.0 * fract(p * C.www) - 1.0;
          vec3 h = abs(x) - 0.5;
          vec3 ox = floor(x + 0.5);
          vec3 a0 = x - ox;
          vec2 g0 = vec2(a0.x, h.x); vec2 g1 = vec2(a0.y, h.y); vec2 g2 = vec2(a0.z, h.z);
          vec3 norm = 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
          g0 *= norm.x; g1 *= norm.y; g2 *= norm.z;
          vec3 t = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
          vec3 t2 = t*t; vec3 t4 = t2*t2;
          float gd0 = dot(g0, x0), gd1 = dot(g1, x12.xy), gd2 = dot(g2, x12.zw);
          return 130.0 * dot(t4, vec3(gd0, gd1, gd2));
        }
        float fbm3T(vec2 p, float freq, vec2 salt) {
          float h=0.0,a=1.0,f=freq; vec2 o=salt;
          h+=a*snoiseT(p*f+o); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
          h+=a*snoiseT(p*f+o); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
          h+=a*snoiseT(p*f+o); return h;
        }
        float fbm2T(vec2 p, float freq, vec2 salt) {
          float h=0.0,a=1.0,f=freq; vec2 o=salt;
          h+=a*snoiseT(p*f+o); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
          h+=a*snoiseT(p*f+o); return h;
        }
        float ridgeFbmT(vec2 p, float freq, vec2 salt) {
          float h=0.0,a=1.0,f=freq; vec2 o=salt;
          h+=a*pow(1.0-abs(snoiseT(p*f+o)),0.7); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
          h+=a*pow(1.0-abs(snoiseT(p*f+o)),0.7); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
          h+=a*pow(1.0-abs(snoiseT(p*f+o)),0.7); return h-0.875;
        }
        float biomeErosionStrengthT(int bi) {
          if (bi==0) return 1.20; if (bi==1) return 1.30; if (bi==2) return 0.85;
          if (bi==3) return 0.55; if (bi==4) return 0.45; if (bi==5) return 0.10;
          if (bi==6) return 0.65; if (bi==7) return 0.50; if (bi==8) return 0.40;
          return 0.7;
        }
        float applyErosionDeltaT(vec2 p, int bi) {
          float em = fbm3T(p, 0.005, vec2(3.7, 5.1));
          float ed = fbm2T(p, 0.02,  vec2(8.3, 1.9));
          float channels = (1.0 - abs(em)) * (1.0 - abs(ed)) * 0.7;
          float rough = fbm2T(p, 0.08, vec2(17.7, 23.1)) * 0.10;
          return (-channels * 0.15 + rough * 0.15) * biomeErosionStrengthT(bi) * 20.0;
        }
        float biomeContributionT(vec2 p, int bi) {
          vec4 b0 = uBiomes0[bi]; vec4 b1 = uBiomes1[bi];
          float amp=b0.x, freq=b0.y, elev=b0.z, continentMix=b0.w;
          float ridgeMix=b1.x, rollMix=b1.y;
          float bf = float(bi);
          vec2 saltC = vec2(13.0+bf*7.1, 29.0-bf*3.3);
          vec2 saltR = vec2(91.0-bf*5.7, 41.0+bf*2.9);
          vec2 saltL = vec2(57.0+bf*11.0, 17.0-bf*6.4);
          float cont = fbm3T(p,freq,saltC);
          float ridge = ridgeFbmT(p,freq*2.0,saltR);
          float roll = fbm2T(p,freq*0.5,saltL);
          float erosion = fbm2T(p, freq*3.0, saltL+vec2(31.7,19.1)) * 0.3;
          float broadGate = clamp(cont*0.5+0.5, 0.0, 1.0);
          ridge = ridge * (1.0 - erosion) * broadGate;
          float h = elev + amp*(cont*continentMix + ridge*ridgeMix + roll*rollMix);
          h += applyErosionDeltaT(p, bi);
          return h;
        }
        float oceanFloorHeightT(vec2 p) {
          return -25.0 + 4.0 * fbm2T(p, 0.003, vec2(2.1, 4.7));
        }
        float anchoredHeightT(vec2 p) {
          float falloff = uFalloff;
          int n = uAnchorCount;
          float ws[MAX_ANCHORS_G];
          float sum = 0.0;
          for (int i = 0; i < MAX_ANCHORS_G; i++) {
            if (i >= n) break;
            vec2 dv = p - uAnchors[i].xy;
            float d = sqrt(dot(dv, dv));
            float w = exp(-d / falloff);
            ws[i] = w; sum += w;
          }
          float inv = 1.0 / max(sum, 1e-6);
          float y = 0.0;
          for (int i = 0; i < MAX_ANCHORS_G; i++) {
            if (i >= n) break;
            float w = ws[i];
            int bi = int(uAnchors[i].w + 0.5);
            y += (w * inv) * (biomeContributionT(p, bi) + uAnchors[i].z);
          }
          float anchorInfluence = smoothstep(0.3, 2.5, sum);
          return mix(oceanFloorHeightT(p), y, anchorInfluence);
        }
        vec2 worldToBakeUV(vec2 worldXZ) {
          vec2 dxz = worldXZ - uSampleCameraPos.xz;
          float r = length(dxz);
          if (r < 1e-6) return vec2(0.5);
          float rNorm = clamp(r / uMaxR, 0.0, 1.0);
          float t = pow(rNorm, 1.0 / max(uConcentration, 1e-3));
          vec2 dir = dxz / r;
          return clamp(0.5 + 0.5 * t * dir, vec2(0.0), vec2(1.0));
        }
        vec3 mod289v3(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec2 mod289v2(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
        vec3 permute3(vec3 x) { return mod289v3(((x * 34.0) + 1.0) * x); }
        float simplex2d(vec2 v) {
          const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
          vec2 i = floor(v + dot(v, C.yy));
          vec2 x0 = v - i + dot(i, C.xx);
          vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
          vec4 x12 = x0.xyxy + C.xxzz; x12.xy -= i1;
          i = mod289v2(i);
          vec3 p = permute3(permute3(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
          vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
          m = m*m; m = m*m;
          vec3 xx = 2.0 * fract(p * C.www) - 1.0;
          vec3 hh = abs(xx) - 0.5;
          vec3 ox = floor(xx + 0.5);
          vec3 a0 = xx - ox;
          m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + hh*hh);
          vec3 g; g.x = a0.x*x0.x + hh.x*x0.y; g.yz = a0.yz*x12.xz + hh.yz*x12.yw;
          return 130.0 * dot(m, g);
        }
        vec3 hsl2rgb(vec3 c) {
          vec3 rgb = clamp(abs(mod(c.x*6.0 + vec3(0.0,4.0,2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
          float v = c.z + c.y * (c.z < 0.5 ? c.z : (1.0 - c.z));
          float s = (v == 0.0) ? 0.0 : 2.0 - 2.0 * c.z / v;
          return v * mix(vec3(1.0), rgb, s);
        }
        vec3 terrainPalette(vec4 m, float bh) {
          vec3 sand   = vec3(0.86, 0.78, 0.58);
          vec3 grassC = vec3(0.52, 0.70, 0.26);
          vec3 forest = vec3(0.18, 0.36, 0.14);
          vec3 rock   = vec3(0.62, 0.55, 0.46);
          vec3 snow   = vec3(0.95, 0.96, 0.98);
          float rockW   = clamp(m.r, 0.0, 1.0);
          float snowW   = clamp(m.g, 0.0, 1.0);
          float grassW  = clamp(m.b, 0.0, 1.0);
          float forestW = clamp(m.a, 0.0, 1.0);
          float sandW   = clamp(1.0 - max(max(grassW, forestW), max(rockW, snowW)), 0.0, 1.0);
          float total = sandW + grassW + forestW + rockW + snowW + 1e-5;
          vec3 c = (sand * sandW + grassC * grassW + forest * forestW + rock * rockW + snow * snowW) / total;
          return c;
        }
        void main() {`);

      const procedural = `
        // ----- WORLD-ANCHORED GRID PLACEMENT -----
        // gid -> cellIndex in [0, gridN*gridN). cellOff = cellIndex relative to player cell.
        // worldCellI = pCell + cellOff. Each WORLD CELL is rendered by exactly one gid;
        // when player moves, the cell that left the disk is reclaimed by the gid that
        // was rendering it (via the modulo permutation), so the cell at world XZ=W stays
        // visually anchored — only the gid index identifying it changes.
        const float CELL = 6.0;
        const int gridN = 150;
        const int gridN2 = 22500; // gridN*gridN
        ivec2 pCell = ivec2(floor(uCameraXZ / CELL));
        int cellIndex = gl_InstanceID % gridN2;
        ivec2 cellOff = ivec2(cellIndex - (cellIndex / gridN) * gridN, cellIndex / gridN) - ivec2(gridN / 2);
        ivec2 worldCellI = pCell + cellOff;
        vec2 cellCenter = (vec2(worldCellI) + vec2(0.5)) * CELL;
        // Hash visual params by worldCellI (NOT gl_InstanceID) so each WORLD cell looks the same regardless of which gid renders it.
        vec2 wci = vec2(worldCellI);
        float u01x = fract(sin(dot(wci, vec2(127.1, 311.7))) * 43758.5453);
        float u01z = fract(sin(dot(wci, vec2(269.5, 183.3))) * 43758.5453);
        float u01t = fract(sin(dot(wci, vec2(419.2, 371.9))) * 43758.5453);
        vec2 wp = cellCenter + (vec2(u01x, u01z) - 0.5) * (CELL * 0.93);

        float FWD_BIAS = 175.0;
        vec2 testPoint = uCameraXZ + uViewForwardXZ * FWD_BIAS;
        float dTest = length(wp - testPoint);

        float MAX_R = 450.0;
        float densityFalloff = 1.0 - smoothstep(50.0, MAX_R, dTest);
        float gidThresh = u01t;
        if (gidThresh > densityFalloff) {
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
          vBladeColor = vec3(1.0);
          vDistFade = 0.0;
          return;
        }

        vec2 buv = worldToBakeUV(wp);
        float bh = texture2D(uTHeight, buv).r * 300.0 - 100.0;
        vec4 mat = texture2D(uTMat, buv);
        float biomeDensity = 0.0;
        biomeDensity += clamp(mat.a * 1.0, 0.0, 1.0);
        biomeDensity += clamp(mat.b * 0.6, 0.0, 0.6);
        biomeDensity -= clamp(mat.r * 0.8, 0.0, 0.8);
        biomeDensity -= clamp(mat.g * 1.5, 0.0, 1.5);
        biomeDensity = clamp(biomeDensity, 0.0, 1.0);
        float clump = simplex2d(wp * 0.02);
        biomeDensity *= 0.5 + 0.5 * smoothstep(-0.6, 0.6, clump);
        if (gidThresh > biomeDensity || bh < 1.0) {
          gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
          vBladeColor = vec3(1.0);
          vDistFade = 0.0;
          return;
        }
        float hx = u01x;
        float hz = u01z;
        float scaleX = 4.5 + hx * 1.5;
        float scaleY = 3.5 + hz * 2.5;
        float scaleZ = 4.5 + hx * 1.5;
        float yaw = (hx + hz) * 6.2831853;
        float cy = cos(yaw), sy = sin(yaw);
        vec3 local = vec3(transformed.x * scaleX, transformed.y * scaleY, transformed.z * scaleZ);
        local.xz = mat2(cy, -sy, sy, cy) * local.xz;
        float windOffset = 2.0 * 3.14 * simplex2d(wp / uWindScale);
        vec3 windSway = local.y * uWindStrength *
          sin(uTime * uWindFrequency + windOffset) *
          cos(uTime * 1.4 * uWindFrequency + windOffset);
        local += windSway;
        vec3 worldPos = local + vec3(wp.x, bh, wp.y);
        // DECISIVE COLOR PATH: dominant terrain albedo (already contains biome blend +
        // biotic-AO baked-in), small grass-tint to keep blade hue alive, slope darken,
        // per-blade brightness hash. No accent stack, no double biotic-AO.
        vec3 terrainAlbedo = texture2D(uTAlbedo, buv).rgb;
        // Fallback if albedo texture not yet baked (first frame).
        if (dot(terrainAlbedo, terrainAlbedo) < 1e-6) terrainAlbedo = vec3(0.40, 0.50, 0.30);
        vec3 hsl = hsl2rgb(vec3(0.28 + (u01x - 0.5) * 0.05, 0.55, 0.30 + (u01z - 0.5) * 0.10));
        vec3 spotTone = mix(terrainAlbedo, hsl, 0.30);
        float slope = clamp(mat.r, 0.0, 1.0);
        spotTone *= 1.0 - slope * 0.35;
        spotTone *= 0.92 + u01x * 0.16;
        vBladeColor = clamp(spotTone, vec3(0.0), vec3(2.0));
        // Full fade-out at MAX_R: alpha goes to 0 so distant grass disappears
        // rather than mixing toward fog tint.
        vDistFade = 1.0 - smoothstep(120.0, 440.0, length(wp - uCameraXZ));
        // Use viewMatrix directly (NOT modelViewMatrix) so any non-identity parent
        // transform on the Grass Object3D / Environment doesn't double-apply on top
        // of an already-world-space worldPos. Bypasses the slide-with-camera bug.
        vec4 mvPosition = viewMatrix * vec4(worldPos, 1.0);
        gl_Position = projectionMatrix * mvPosition;
      `;
      shader.vertexShader = shader.vertexShader.replace(`#include <project_vertex>`, procedural);

      // Fragment: multiply diffuse by vBladeColor.
      shader.fragmentShader = `varying vec3 vBladeColor;\nvarying float vDistFade;\n` + shader.fragmentShader;
      shader.fragmentShader = shader.fragmentShader.replace(
        `#include <color_fragment>`,
        `#include <color_fragment>\n diffuseColor.rgb *= vBladeColor;\n diffuseColor.a *= vDistFade;\n if (diffuseColor.a < 0.02) discard;`
      );

      material.userData.shader = shader;
      this._grassShader = shader;
      this._shaders.push(shader);
    };
  }

  appendFlowerWindShader(material) {
    material.onBeforeCompile = (shader) => {
      shader.uniforms.uTime = { value: 0 };
      shader.uniforms.uWindStrength = { value: this.options.windStrength };
      shader.uniforms.uWindFrequency = { value: this.options.windFrequency };
      shader.uniforms.uWindScale = { value: this.options.windScale };
      shader.vertexShader = `
        uniform float uTime;
        uniform vec3 uWindStrength;
        uniform float uWindFrequency;
        uniform float uWindScale;
      ` + shader.vertexShader;
      shader.vertexShader = shader.vertexShader.replace(`void main() {`, `
        vec3 mod289fv3(vec3 x) { return x - floor(x*(1.0/289.0))*289.0; }
        vec2 mod289fv2(vec2 x) { return x - floor(x*(1.0/289.0))*289.0; }
        vec3 permutef(vec3 x) { return mod289fv3(((x*34.0)+1.0)*x); }
        float simplex2dF(vec2 v) {
          const vec4 C = vec4(0.211324865405187,0.366025403784439,-0.577350269189626,0.024390243902439);
          vec2 i = floor(v + dot(v, C.yy));
          vec2 x0 = v - i + dot(i, C.xx);
          vec2 i1 = (x0.x > x0.y) ? vec2(1.0,0.0) : vec2(0.0,1.0);
          vec4 x12 = x0.xyxy + C.xxzz; x12.xy -= i1;
          i = mod289fv2(i);
          vec3 p = permutef(permutef(i.y + vec3(0.0,i1.y,1.0)) + i.x + vec3(0.0,i1.x,1.0));
          vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
          m=m*m; m=m*m;
          vec3 xx = 2.0*fract(p*C.www) - 1.0;
          vec3 hh = abs(xx) - 0.5;
          vec3 ox = floor(xx+0.5);
          vec3 a0 = xx - ox;
          m *= 1.79284291400159 - 0.85373472095314 * (a0*a0 + hh*hh);
          vec3 g; g.x=a0.x*x0.x + hh.x*x0.y; g.yz=a0.yz*x12.xz + hh.yz*x12.yw;
          return 130.0*dot(m,g);
        }
        void main() {`);
      const block = `
        vec4 mvPosition = instanceMatrix * vec4(transformed, 1.0);
        float windOffset = 2.0 * 3.14 * simplex2dF((modelMatrix * mvPosition).xz / uWindScale);
        vec3 windSway = position.y * uWindStrength *
          sin(uTime * uWindFrequency + windOffset) *
          cos(uTime * 1.4 * uWindFrequency + windOffset);
        mvPosition.xyz += windSway;
        mvPosition = modelViewMatrix * mvPosition;
        gl_Position = projectionMatrix * mvPosition;
      `;
      shader.vertexShader = shader.vertexShader.replace(`#include <project_vertex>`, block);
      material.userData.shader = shader;
      this._shaders.push(shader);
    };
  }

  // ---- Flower streaming (CPU-side, low instance count) ----
  _buildFlowerInstancedMeshes() {
    const colors = { white: _whiteFlower, blue: _blueFlower, yellow: _yellowFlower };
    const meshes = {};
    const MAX = 200;
    for (const [name, root] of Object.entries(colors)) {
      let chosen = null;
      root.traverse((o) => { if (chosen) return; if (o.isMesh && o.material && o.material.map) chosen = o; });
      if (!chosen) continue;
      const inst = new THREE.InstancedMesh(chosen.geometry, chosen.material, MAX);
      inst.count = 0;
      inst.castShadow = false;
      inst.receiveShadow = true;
      inst.frustumCulled = false;
      this.flowers.add(inst);
      meshes[name] = inst;
    }
    this._flowerStream = {
      CELL: 50, RADIUS_CELLS: 4, PER_CELL: 2,
      lastCx: 9999, lastCz: 9999,
      meshes,
      cellSlots: new Map(),
      freeSlots: { white: [], blue: [], yellow: [] },
      highWater: { white: 0, blue: 0, yellow: 0 },
      addQ: [], removeQ: [], dirty: { white: false, blue: false, yellow: false },
    };
    this._flowerStreamReady = true;
  }

  _streamFlowers(camera) {
    const s = this._flowerStream;
    const cx = Math.floor(camera.position.x / s.CELL);
    const cz = Math.floor(camera.position.z / s.CELL);
    if (cx === s.lastCx && cz === s.lastCz) return;
    s.lastCx = cx; s.lastCz = cz;
    this._scheduleFlowerDiff(cx, cz);
  }

  _scheduleFlowerDiff(camCx, camCz) {
    const s = this._flowerStream;
    if (!s) return;
    const wanted = new Set();
    for (let dz = -s.RADIUS_CELLS; dz <= s.RADIUS_CELLS; dz++) {
      for (let dx = -s.RADIUS_CELLS; dx <= s.RADIUS_CELLS; dx++) {
        wanted.add(`${camCx + dx},${camCz + dz}`);
      }
    }
    for (const cellKey of s.cellSlots.keys()) {
      if (!wanted.has(cellKey)) s.removeQ.push(cellKey);
    }
    for (const cellKey of wanted) {
      if (!s.cellSlots.has(cellKey) && !s.addQ.includes(cellKey)) s.addQ.push(cellKey);
    }
  }

  _drainFlowerQueue() {
    const s = this._flowerStream;
    if (!s) return;
    const t0 = performance.now();
    const FRAME_BUDGET_MS = 4;
    const MAX_BUILDS = 2;
    let builds = 0;
    while (s.removeQ.length && (performance.now() - t0) < FRAME_BUDGET_MS) {
      const cellKey = s.removeQ.shift();
      const placements = s.cellSlots.get(cellKey);
      if (!placements) continue;
      for (const p of placements) {
        const inst = s.meshes[p.color]; if (!inst) continue;
        inst.setMatrixAt(p.slot, _ZERO_MATRIX_GRASS);
        s.freeSlots[p.color].push(p.slot);
        s.dirty[p.color] = true;
      }
      s.cellSlots.delete(cellKey);
    }
    while (s.addQ.length && builds < MAX_BUILDS && (performance.now() - t0) < FRAME_BUDGET_MS) {
      const cellKey = s.addQ.shift();
      if (s.cellSlots.has(cellKey)) continue;
      const [cx, cz] = cellKey.split(',').map(Number);
      const entries = this._flowerCellEntries(cx, cz);
      const placements = [];
      for (const e of entries) {
        const inst = s.meshes[e.color]; if (!inst) continue;
        const cap = inst.instanceMatrix.array.length / 16;
        let slot;
        if (s.freeSlots[e.color].length) slot = s.freeSlots[e.color].pop();
        else if (s.highWater[e.color] < cap) slot = s.highWater[e.color]++;
        else continue;
        inst.setMatrixAt(slot, e.matrix);
        placements.push({ color: e.color, slot });
        s.dirty[e.color] = true;
      }
      s.cellSlots.set(cellKey, placements);
      builds++;
    }
    for (const c of ['white', 'blue', 'yellow']) {
      if (!s.dirty[c]) continue;
      const inst = s.meshes[c]; if (!inst) continue;
      inst.count = s.highWater[c];
      inst.instanceMatrix.needsUpdate = true;
      s.dirty[c] = false;
    }
  }

  _flowerCellEntries(cx, cz) {
    const s = this._flowerStream;
    if (!this._flowerCellCache) this._flowerCellCache = new Map();
    const key = cx * 100000 + cz;
    const hit = this._flowerCellCache.get(key);
    if (hit) return hit;
    const colors = ['white', 'blue', 'yellow'];
    let seed = (cx * 49979693) ^ (cz * 86028157) ^ 0xfeedf00d;
    const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 0x100000000; };
    const dummy = new THREE.Object3D();
    const out = [];
    for (let i = 0; i < s.PER_CELL; i++) {
      const lx = (cx + rnd()) * s.CELL;
      const lz = (cz + rnd()) * s.CELL;
      const w = biomeWeightsAt(lx, lz);
      const dom = dominantBiomeAt(lx, lz).name;
      if (dom !== 'grassland' && dom !== 'forest_hills' && dom !== 'floodplain') continue;
      if (w.grass < 0.4) continue;
      if (w.slope > 0.4) continue;
      if (w.height < 4) continue;
      if (w.height > 60) continue;
      const colorPick = colors[Math.floor(rnd() * 3)];
      dummy.position.set(lx, heightAt(lx, lz), lz);
      dummy.rotation.set(0, rnd() * Math.PI * 2, 0);
      const sc = 0.025 + 0.04 * rnd();
      dummy.scale.set(sc, sc, sc);
      dummy.updateMatrix();
      out.push({ color: colorPick, matrix: dummy.matrix.clone() });
    }
    this._flowerCellCache.set(key, out);
    return out;
  }
}
