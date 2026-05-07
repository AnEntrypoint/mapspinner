import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Tree, TreePreset } from 'mapspinner';
import { Environment } from './environment.js';
import { heightAt, dominantBiomeAt, biomeWeightsAt } from './terrain.js';

// Cheap 2D value noise [0,1] mirroring shader vnoise(wp*0.05) for density clustering.
function _hash21d(x, y) {
  let h = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return h - Math.floor(h);
}
function _densityNoise(x, y) {
  const ix = Math.floor(x), iy = Math.floor(y);
  const fx = x - ix, fy = y - iy;
  const a = _hash21d(ix, iy);
  const b = _hash21d(ix + 1, iy);
  const c = _hash21d(ix, iy + 1);
  const d = _hash21d(ix + 1, iy + 1);
  const ux = fx * fx * (3 - 2 * fx);
  const uy = fy * fy * (3 - 2 * fy);
  return (a * (1 - ux) + b * ux) * (1 - uy) + (c * (1 - ux) + d * ux) * uy;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function paintUI() {
  return new Promise(resolve => requestAnimationFrame(resolve));
}

/**
 * Creates a new instance of the Three.js scene
 * @param {THREE.WebGLRenderer} renderer 
 * @returns 
 */
export async function createScene(renderer) {
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x94b9f8, 0.0015);

  const environment = new Environment(renderer);
  scene.add(environment);

  const camera = new THREE.PerspectiveCamera(
    60,
    window.innerWidth / window.innerHeight,
    0.1,
    2000,
  );
  camera.position.set(100, 20, 0);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.enablePan = true;
  // Allow looking up/down a bit so user can see the sky/ground while walking.
  controls.minPolarAngle = 0.2;
  controls.maxPolarAngle = Math.PI - 0.2;
  controls.minDistance = 5;
  controls.maxDistance = 250;
  controls.target.set(0, 25, 0);
  controls.update();

  // ---- WASD walking ----
  // Translate the OrbitControls target along the camera-forward / right axes;
  // OrbitControls' camera follows the target. The target's y is locked to the
  // terrain height + a 1.7m head height so the camera stays above ground.
  const keys = Object.create(null);
  window.addEventListener('keydown', (e) => { keys[e.code] = true; });
  window.addEventListener('keyup',   (e) => { keys[e.code] = false; });
  const walkState = {
    pos: new THREE.Vector3(0, heightAt(0, 0) + 1.7, 0),
    vel: new THREE.Vector3(),
    enabled: true,
    headHeight: 1.7,
  };
  // Position camera directly behind the player along -Z, slightly elevated.
  // 14m back + 6m up gives a near-horizontal third-person view (~23° down)
  // that reads as "looking forward" instead of an oblique top-down angle.
  controls.target.copy(walkState.pos);
  camera.position.set(walkState.pos.x, walkState.pos.y + 6, walkState.pos.z + 14);
  controls.update();
  const _walkFwd = new THREE.Vector3();
  const _walkRight = new THREE.Vector3();
  const _walkUp = new THREE.Vector3(0, 1, 0);
  const _walkOffset = new THREE.Vector3();
  function updateWalk(dt) {
    if (!walkState.enabled) return;
    let ix = 0, iz = 0;
    if (keys['KeyW']) iz += 1;
    if (keys['KeyS']) iz -= 1;
    if (keys['KeyD']) ix += 1;
    if (keys['KeyA']) ix -= 1;
    if (ix === 0 && iz === 0) return;
    const speed = (keys['ShiftLeft'] || keys['ShiftRight']) ? 28 : 10;
    camera.getWorldDirection(_walkFwd); _walkFwd.y = 0; _walkFwd.normalize();
    _walkRight.crossVectors(_walkFwd, _walkUp).normalize();
    const len = Math.hypot(ix, iz);
    ix /= len; iz /= len;
    walkState.pos.x += _walkFwd.x * iz * speed * dt + _walkRight.x * ix * speed * dt;
    walkState.pos.z += _walkFwd.z * iz * speed * dt + _walkRight.z * ix * speed * dt;
    const groundY = heightAt(walkState.pos.x, walkState.pos.z);
    walkState.pos.y = Math.max(groundY, 0) + walkState.headHeight;
    _walkOffset.copy(camera.position).sub(controls.target);
    controls.target.copy(walkState.pos);
    camera.position.copy(walkState.pos).add(_walkOffset);
    const camGround = heightAt(camera.position.x, camera.position.z);
    const camFloor = Math.max(camGround, 0) + 0.6;
    if (camera.position.y < camFloor) camera.position.y = camFloor;
  }
  controls.updateWalk = updateWalk;

  // Central editable tree lives on the editor page (editor.html). The world
  // view only streams pooled trees around the player.
  const tree = null;

  // Add a forest of trees in the background
  const forest = new THREE.Group();
  forest.name = 'Forest';

  const progressElement = document.getElementById('loading-text');

  if (progressElement) progressElement.innerHTML = 'LOADING... 0%';

  const treeCount = 100;
  const minDistance = 175;
  const maxDistance = 500;

  // Deterministic RNG override for harness reproducibility.
  // Set window.__seed before scene creation to lock layout.
  if (typeof window !== 'undefined' && window.__seed !== undefined) {
    let s = (window.__seed | 0) || 1;
    Math.random = function () {
      s = (s * 16807) % 2147483647;
      return s / 2147483647;
    };
  }

  // Biome → weighted preset table. Captures the ecological logic the user asked
  // for: bushes near water, big trees in green belt, conifers up high.
  const BIOME_PRESETS = {
    lakeshore:    [['Bush 1', 3], ['Bush 2', 3], ['Bush 3', 2], ['Aspen Small', 2], ['Aspen Medium', 1]],
    river_valley: [['Bush 1', 3], ['Bush 2', 2], ['Aspen Small', 3], ['Aspen Medium', 2], ['Ash Small', 1]],
    plains:       [['Bush 1', 2], ['Aspen Small', 2], ['Aspen Medium', 2], ['Oak Small', 1], ['Ash Small', 1]],
    grassland:    [['Bush 1', 2], ['Bush 2', 1], ['Ash Medium', 2], ['Oak Medium', 2], ['Aspen Medium', 2], ['Oak Small', 1]],
    forest_hills: [['Oak Large', 3], ['Oak Medium', 3], ['Ash Large', 3], ['Ash Medium', 2], ['Pine Medium', 2], ['Aspen Large', 2], ['Bush 3', 1]],
    alpine_rock:  [['Pine Small', 3], ['Pine Medium', 2], ['Bush 3', 1]],
    snow_peak:    [['Pine Small', 3]],
  };
  function pickPresetForBiome(biome) {
    const table = BIOME_PRESETS[biome];
    if (!table) return null;
    let total = 0;
    for (const [, w] of table) total += w;
    let r = Math.random() * total;
    for (const [name, w] of table) {
      r -= w;
      if (r <= 0) return name;
    }
    return table[0][0];
  }
  // Elevation bands (meters). Water y<0, beach 0..6, floodplain 6..10,
  // lowland 10..16, green belt 16..50, sub-alpine 50..90, alpine 90+.
  function pickPresetByElevation(biome, h) {
    if (h < 6) return null;                          // beach / shore — no vegetation
    if (h < 12) {
      // Floodplain + lower transition: bushes only — no real trees this close to shore
      return ['Bush 1', 'Bush 2', 'Bush 3'][Math.floor(Math.random() * 3)];
    }
    if (h < 16) {
      // Lowland transition: medium oak/ash mixed in
      const r = Math.random();
      if (r < 0.4) return 'Aspen Medium';
      if (r < 0.7) return 'Oak Medium';
      return ['Ash Medium', 'Oak Small', 'Ash Small'][Math.floor(Math.random() * 3)];
    }
    if (h < 50) {
      // Green belt: big trees dominant
      return pickPresetForBiome('forest_hills') || pickPresetForBiome(biome);
    }
    if (h < 90) {
      // Sub-alpine: medium pines + occasional medium oak
      return Math.random() < 0.7 ? 'Pine Medium' : 'Pine Small';
    }
    // Snow line and above: only hardy small pines
    return Math.random() < 0.85 ? 'Pine Small' : null;
  }
  // ---- Tree pool (streaming) ----
  // We pre-generate a small pool of trees per preset (Tree.generate is too
  // expensive to run per-frame). Streaming = on chunk-cross we walk the pool
  // and assign each pooled tree to a chunk slot around the camera, sized down
  // when no valid spot is found.
  const POOL_PRESETS = ['Bush 1', 'Bush 2', 'Bush 3', 'Aspen Small', 'Aspen Medium', 'Aspen Large', 'Oak Small', 'Oak Medium', 'Oak Large', 'Ash Small', 'Ash Medium', 'Ash Large', 'Pine Small', 'Pine Medium', 'Pine Large'];
  const POOL_PER_PRESET = 40;
  const POOL_OVERRIDE = { 'Bush 1': 80, 'Bush 2': 80, 'Bush 3': 80 };
  const treePool = { byPreset: new Map() };
  for (const preset of POOL_PRESETS) treePool.byPreset.set(preset, []);

  // opt.md §7: branch polycount override — 73% triangle reduction.
  function reduceBranchPoly(opts) {
    const b = opts.branch;
    b.sections = { 0: 6, 1: 5, 2: 4, 3: 3 };
    b.segments = { 0: 5, 1: 4, 2: 3, 3: 3 };
    b.children = { 0: 5, 1: 5, 2: 4 };
  }
  async function buildPool() {
    let total = 0;
    for (const preset of POOL_PRESETS) total += (POOL_OVERRIDE[preset] || POOL_PER_PRESET);
    let made = 0;
    for (const preset of POOL_PRESETS) {
      const perPreset = POOL_OVERRIDE[preset] || POOL_PER_PRESET;
      for (let i = 0; i < perPreset; i++) {
        const t = new Tree();
        t.loadPreset(preset);
        t.options.seed = (preset.length * 7919 + i * 3001) & 0xffff;
        if (!preset.startsWith('Pine')) reduceBranchPoly(t.options);
        if (typeof window !== 'undefined' && typeof window.__leafScale === 'number') {
          t.options.leaves.count = Math.max(1, Math.floor(t.options.leaves.count * window.__leafScale));
          t.options.leaves.size = t.options.leaves.size * (window.__leafSizeScale || 1);
        }
        t.generate();
        t.castShadow = true;
        t.receiveShadow = true;
        t.userData.dynamic = true;       // exclude from freezeStatics
        t.traverse((o) => { o.userData = o.userData || {}; o.userData.dynamic = true; });
        t.position.set(0, -10000, 0);    // park below ground until placed
        treePool.byPreset.get(preset).push(t);
        forest.add(t);
        made++;
        const progress = Math.floor(100 * made / total);
        if (progressElement) progressElement.innerText = `LOADING... ${progress}%`;
        if (made % 8 === 0) await paintUI();
      }
    }
  }
  await buildPool();

  // ---- Tree streamer (opt.md §2,§3,§4,§7,§7a) ----
  const TREE_CELL = 80;
  const TREE_NEAR_CELLS = 3;       // 7×7 = 49 near cells (real trees)
  const TREE_FAR_CELLS  = 30;      // 61×61 ring (billboards), 2400m radius
  const TREE_PER_CELL = 1;

  const _streamStats = { cacheHits: 0, billboardsBaked: 0, billboardChunks: 0, nearChunks: 0, gpuUploadsThisFrame: 0 };

  // §4 placement cache: keyed by cell hash, holds deterministic placements
  // (lx, lz, ry, presetName, biome). Compute once per cell, reuse across
  // LOD bands and revisits.
  const _placementCache = new Map();
  function _cellKey(cx, cz) { return cx * 100000 + cz; }
  const BUSH_NAMES = ['Bush 1', 'Bush 2', 'Bush 3'];
  const BUSH_PER_CELL = 2;
  function _bushPlacementsForCell(cx, cz) {
    let seed = ((cx * 73856093) ^ (cz * 19349663) ^ 0xb05ec0de) >>> 0;
    const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 0x100000000; };
    const out = [];
    for (let i = 0; i < BUSH_PER_CELL; i++) {
      const lx = (cx + rnd()) * TREE_CELL;
      const lz = (cz + rnd()) * TREE_CELL;
      const w = biomeWeightsAt(lx, lz);
      // Open grassy band where canopy trees DON'T grow
      if (w.height < 6 || w.height >= 50) continue;
      if (w.slope > 0.3) continue;
      if (w.grass < 0.3 && w.forest < 0.3) continue;
      // Cluster in same density patches as shader AO
      const densityNoise = _densityNoise(lx * 0.05, lz * 0.05);
      if (rnd() > 0.25 + densityNoise * 0.25) continue;
      const ry = rnd() * Math.PI * 2;
      const presetName = BUSH_NAMES[Math.floor(rnd() * 3)];
      const sy = heightAt(lx, lz);
      const scale = 0.7 + rnd() * 0.6;
      out.push({ lx, ly: sy, lz, ry, presetName, biome: 'bush', scale, isBush: true });
    }
    return out;
  }
  function _placementsForCell(cx, cz) {
    const key = _cellKey(cx, cz);
    const cached = _placementCache.get(key);
    if (cached) { _streamStats.cacheHits++; return cached; }
    let seed = ((cx * 73856093) ^ (cz * 19349663) ^ 0xc0ffee01) >>> 0;
    const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 0x100000000; };
    const out = [];
    for (let i = 0; i < TREE_PER_CELL; i++) {
      const lx = (cx + rnd()) * TREE_CELL;
      const lz = (cz + rnd()) * TREE_CELL;
      // Cluster trees in same density patches as rocks and shader AO.
      const densityNoise = _densityNoise(lx * 0.05, lz * 0.05);
      if (rnd() > densityNoise * 0.5) continue;
      const ry = rnd() * Math.PI * 2;
      const biome = dominantBiomeAt(lx, lz).name;
      const w = biomeWeightsAt(lx, lz);
      if (biome === 'desert_dunes' || biome === 'river_channel') continue;
      if (w.height < 6) continue;
      const presetName = pickPresetByElevation(biome, w.height);
      if (!presetName) continue;
      const isBush = presetName.startsWith('Bush');
      const isPine = presetName.startsWith('Pine');
      if (!isBush && w.slope > 0.45) continue;
      if (isBush && w.slope > 0.65) continue;
      const isBig = (/Large|Medium/).test(presetName) && !isPine && !isBush;
      if (isBig && w.grass < 0.3 && w.forest < 0.2) continue;
      if (isPine && w.height < 15 && rnd() > 0.2) continue;
      const sy = heightAt(lx, lz);
      const scale = 0.85 + rnd() * 0.4;
      out.push({ lx, ly: sy, lz, ry, presetName, biome, scale });
    }
    // Append bush placements (10x density in open grasslands)
    const bushes = _bushPlacementsForCell(cx, cz);
    for (const b of bushes) out.push(b);
    _placementCache.set(key, out);
    return out;
  }

  // §7a billboard impostor pipeline — PER-PRESET.
  // Each preset gets its own 256×512 RenderTarget + its own InstancedMesh + its
  // own ShaderMaterial sampling that RT. ~16 draw calls instead of 1; no atlas
  // RT (saves ~16 MB VRAM); no per-instance UV attribute. instanceSize is the
  // only size source — matrix scale is identity.
  const _BB_SUB_W = 256, _BB_SUB_H = 512;
  // preset -> { rt, texture, size:{x,y}, mesh, material, highWater, freeSlots, instSize, capacity }
  const _billboardByPreset = new Map();

  const _quadGeo = new THREE.PlaneGeometry(1, 1).translate(0, 0.5, 0);

  // Shared sun-state uniforms updated each frame from Environment so billboards
  // get cheap directional shading consistent with the rest of the scene.
  const _bbSunDir = new THREE.Vector3(0.4, 1.0, 0.3).normalize();
  const _bbSunCol = new THREE.Color(1.0, 0.95, 0.8);
  const _bbAmbCol = new THREE.Color(0.5, 0.6, 0.75);
  function _makeBillboardMaterial(texture) {
    const mat = new THREE.ShaderMaterial({
      uniforms: {
        uMap: { value: texture },
        uSunDir: { value: _bbSunDir },
        uSunCol: { value: _bbSunCol },
        uAmbCol: { value: _bbAmbCol },
      },
      vertexShader: `
        attribute vec2 instanceSize;
        varying vec2 vUv;
        varying vec3 vWorldPos;
        void main() {
          vUv = uv;
          vec3 instPos = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);
          vec3 worldInst = (modelMatrix * vec4(instPos, 1.0)).xyz;
          vec3 camToInst = worldInst - cameraPosition;
          vec3 right = normalize(vec3(-camToInst.z, 0.0, camToInst.x));
          vec3 up = vec3(0.0, 1.0, 0.0);
          vec3 worldOffset = right * (position.x * instanceSize.x) + up * (position.y * instanceSize.y);
          vec3 wp = worldInst + worldOffset;
          vWorldPos = wp;
          gl_Position = projectionMatrix * viewMatrix * vec4(wp, 1.0);
        }
      `,
      fragmentShader: `
        uniform sampler2D uMap;
        uniform vec3 uSunDir;
        uniform vec3 uSunCol;
        uniform vec3 uAmbCol;
        varying vec2 vUv;
        varying vec3 vWorldPos;
        void main() {
          vec4 c = texture2D(uMap, vUv);
          if (c.a < 0.4) discard;
          // Texture is sampled in linear space already (RT colorSpace=sRGB → auto-decode).
          // Manual sRGB decode here would double-darken the result.
          vec3 base = c.rgb;
          // Hemispheric directional shading. The billboard quad always faces the
          // camera (vertex shader rotates around world Y), so we can't take a
          // surface normal — but we CAN approximate canopy lighting by taking
          // the sun's elevation as a "top vs bottom" gradient and the sun's
          // horizontal azimuth relative to camera as a "left vs right" gradient.
          // vUv.y 0→1 goes base→canopy top. vUv.x 0→1 goes left→right of the
          // camera-facing quad.
          float vert = vUv.y;
          float horiz = vUv.x * 2.0 - 1.0;                                // -1 left, +1 right
          // Top-vs-bottom: sun overhead lights canopy more than base.
          float topShade = mix(0.55, 1.0, vert) * max(0.0, uSunDir.y);
          // Side-vs-side: sun on one azimuth lights that side of the quad more.
          // We don't have a true camera-right axis here — use sun's horizontal
          // direction projected against arbitrary horizontal: sign(uSunDir.x)
          // approximates which side faces sun. This gives subtle left/right
          // modulation so multiple billboards at different positions don't
          // all look identical.
          vec2 sunHoriz = normalize(uSunDir.xz + vec2(0.001));
          float sideShade = 0.5 + 0.5 * horiz * sunHoriz.x;               // [0,1]
          // Translucency: backlit foliage. When sun is BEHIND camera-facing
          // quad (sunDir.z < 0 relative to camera), add tinted glow that
          // brightens the canopy edges.
          float backlit = max(-uSunDir.z, 0.0) * smoothstep(0.4, 0.95, vert);
          // Baseline ambient floor — guarantee billboards are visible even
          // when uAmbCol/uSunCol are zero (e.g. before _updateBillboardSun
          // first runs, or at night).
          vec3 ambient = max(uAmbCol, vec3(0.35));
          vec3 sunC = max(uSunCol, vec3(0.0));
          float sunUp = max(uSunDir.y, 0.0);
          vec3 lit = base * (ambient * 1.10 + sunC * (topShade * 1.4 + sideShade * 0.35));
          lit += base * sunC * backlit * 0.6;                             // glow
          // Final floor: never let the lit color go below the texture base × 0.45
          // — keeps billboards readable through the worst lighting paths.
          lit = max(lit, base * 0.45);
          float fogDistance = length(vWorldPos - cameraPosition);
          float fogDensity = 0.0015;
          float fogFactor = 1.0 - exp(-fogDensity * fogDensity * fogDistance * fogDistance);
          vec3 fogColor = vec3(0.58, 0.73, 0.97);
          lit = mix(lit, fogColor, clamp(fogFactor, 0.0, 1.0));
          gl_FragColor = vec4(lit, 1.0);
        }
      `,
      transparent: false,
      depthWrite: true,
    });
    return mat;
  }
  // Update billboard sun uniforms from Environment lights (called each frame
  // by ux below — but cheap: just write to shared Vector3/Color references).
  function _updateBillboardSun() {
    if (!environment.sun) return;
    _bbSunDir.copy(environment.sun.position).normalize();
    _bbSunCol.copy(environment.sun.color).multiplyScalar(environment.sun.intensity / 5);
    if (environment.ambient) _bbAmbCol.copy(environment.ambient.color).multiplyScalar(environment.ambient.intensity);
  }

  async function _waitTextures(tree, timeoutMs = 4000) {
    const textures = [];
    tree.traverse((o) => {
      if (!o.material) return;
      const m = o.material;
      for (const k of ['map', 'alphaMap', 'normalMap']) {
        if (m[k] && m[k].image) textures.push(m[k]);
      }
    });
    const t0 = performance.now();
    while (performance.now() - t0 < timeoutMs) {
      const allReady = textures.every(t => t.image && (t.image.complete || t.image.width > 0));
      if (allReady) break;
      await new Promise(r => setTimeout(r, 50));
    }
    await new Promise(r => requestAnimationFrame(r));
  }

  async function _bakeBillboard(preset) {
    const list = treePool.byPreset.get(preset);
    if (!list || !list.length) return null;
    const treeGroup = list[0];
    await _waitTextures(treeGroup);
    const tmp = new THREE.Scene();
    tmp.add(new THREE.HemisphereLight(0xffffff, 0x808080, 1.2));
    tmp.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirs = [[1,1,0.5],[-1,1,0.5],[0.5,1,-1],[-0.5,1,-1]];
    for (const d of dirs) {
      const dl = new THREE.DirectionalLight(0xffffff, 0.6);
      dl.position.set(d[0], d[1], d[2]).normalize().multiplyScalar(50);
      tmp.add(dl);
    }
    const oldParent = treeGroup.parent;
    const oldPos = treeGroup.position.clone();
    const wrapper = new THREE.Group();
    if (oldParent) oldParent.remove(treeGroup);
    treeGroup.position.set(0, 0, 0);
    wrapper.add(treeGroup);
    tmp.add(wrapper);
    wrapper.updateMatrixWorld(true);
    const bbox = new THREE.Box3().setFromObject(wrapper);
    const size = new THREE.Vector3(); bbox.getSize(size);
    const center = new THREE.Vector3(); bbox.getCenter(center);
    const w = Math.max(size.x, size.z, 1.0);
    const h = Math.max(size.y, 1.0);
    const cam = new THREE.OrthographicCamera(-w/2, w/2, h/2, -h/2, 0.1, 1000);
    cam.position.set(center.x, center.y, center.z + Math.max(w, h) * 2);
    cam.lookAt(center.x, center.y, center.z);
    const rt = new THREE.WebGLRenderTarget(_BB_SUB_W, _BB_SUB_H, {
      format: THREE.RGBAFormat,
      minFilter: THREE.LinearMipmapLinearFilter,
      magFilter: THREE.LinearFilter,
      generateMipmaps: true,
    });
    rt.texture.colorSpace = THREE.SRGBColorSpace;
    const prevTarget = renderer.getRenderTarget();
    const prevCS = renderer.outputColorSpace;
    const prevClear = renderer.getClearColor(new THREE.Color());
    const prevAlpha = renderer.getClearAlpha();
    const prevPixelRatio = renderer.getPixelRatio();
    renderer.setPixelRatio(1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setRenderTarget(rt);
    renderer.setClearColor(0x000000, 0);
    renderer.clear(true, true, true);
    renderer.render(tmp, cam);
    renderer.setRenderTarget(prevTarget);
    renderer.outputColorSpace = prevCS;
    renderer.setClearColor(prevClear, prevAlpha);
    renderer.setPixelRatio(prevPixelRatio);
    wrapper.remove(treeGroup);
    if (oldParent) {
      oldParent.add(treeGroup);
      treeGroup.position.copy(oldPos);
      treeGroup.updateMatrix();
      treeGroup.matrixWorld.copy(treeGroup.matrix);
    }
    const capacity = _bbCapFor(preset);
    const material = _makeBillboardMaterial(rt.texture);
    const mesh = new THREE.InstancedMesh(_quadGeo, material, capacity);
    mesh.count = 0;
    mesh.userData.dynamic = true;
    mesh.frustumCulled = false;
    mesh.castShadow = false;
    mesh.receiveShadow = false;
    const instSize = new THREE.InstancedBufferAttribute(new Float32Array(capacity * 2), 2);
    instSize.setUsage(THREE.DynamicDrawUsage);
    mesh.geometry.setAttribute('instanceSize', instSize);
    forest.add(mesh);
    const entry = {
      rt,
      texture: rt.texture,
      size: { x: w, y: h },
      mesh,
      material,
      instSize,
      capacity,
      highWater: 0,
      freeSlots: [],
      dirty: false,
    };
    _billboardByPreset.set(preset, entry);
    return entry;
  }

  // Per-preset capacity. Bushes emit BUSH_PER_CELL per cell vs trees TREE_PER_CELL.
  const _BB_FAR_CELLS_TOTAL = (TREE_FAR_CELLS * 2 + 1) * (TREE_FAR_CELLS * 2 + 1);
  function _bbCapFor(preset) {
    const isBush = preset && preset.startsWith('Bush');
    const perCell = isBush ? BUSH_PER_CELL : TREE_PER_CELL;
    return _BB_FAR_CELLS_TOTAL * perCell;
  }

  // Bake all impostors BEFORE dismissing the loading screen so the
  // re-parent dance is invisible to the player. Sequential, one tree at
  // a time, with progress reported in the loading text.
  let _dirtyAllPlacements = true;
  for (let i = 0; i < POOL_PRESETS.length; i++) {
    const preset = POOL_PRESETS[i];
    try { await _bakeBillboard(preset); }
    catch (e) { console.warn('[veg] bake failed', preset, e); }
    _dirtyAllPlacements = true;
    const pct = Math.floor(100 * (i + 1) / POOL_PRESETS.length);
    if (progressElement) progressElement.innerText = `BAKING IMPOSTORS... ${pct}%`;
    await new Promise(r => requestAnimationFrame(r));
  }
  _streamStats.billboardsBaked = _billboardByPreset.size;
  await sleep(300);
  const loadingScreen = document.getElementById('loading-screen');
  if (loadingScreen) loadingScreen.style.display = 'none';

  // Per-preset slot machinery lives on each entry in _billboardByPreset.
  function _bbAllocSlot(entry) {
    if (entry.freeSlots.length) return entry.freeSlots.pop();
    if (entry.highWater >= entry.capacity) return -1;
    return entry.highWater++;
  }

  // §3 chunk records — owned, atomic-swap on band transitions.
  // chunk = { key, cx, cz, lod (0=near,1=far), placements, group, billboardMeshes:Map<preset,InstancedMesh>, claimedTrees:Tree[], pendingEvict }
  const _chunks = new Map();

  // §3 GPU upload queue — 1 chunk per RAF cap. Far chunks (billboard
  // InstancedMesh creation) are the only real GPU uploads; near chunks
  // reuse pre-uploaded pool tree geometry.
  const _gpuUploadQueue = [];

  function _disposeChunk(c) {
    if (c.claimedTrees) {
      for (const t of c.claimedTrees) {
        t.position.y = -10000;
        t.matrixWorldAutoUpdate = true;
        t.matrixAutoUpdate = true;
        t._used = false;
      }
    }
  }

  function _stablePoolIndex(presetName, lx, lz, poolSize) {
    const h = Math.imul(Math.floor(lx * 16) | 0, 73856093) ^
              Math.imul(Math.floor(lz * 16) | 0, 19349663) ^
              Math.imul(presetName.length, 0x9e3779b1);
    return ((h >>> 0) % poolSize);
  }

  function _assembleNearChunk(cx, cz) {
    const placements = _placementsForCell(cx, cz);
    const claimed = [];
    for (const p of placements) {
      const list = treePool.byPreset.get(p.presetName);
      if (!list) continue;
      const N = list.length;
      const start = _stablePoolIndex(p.presetName, p.lx, p.lz, N);
      let t = null;
      for (let k = 0; k < N; k++) {
        const idx = (start + k) % N;
        if (!list[idx]._used) { t = list[idx]; break; }
      }
      if (!t) {
        // Pool exhausted — drop placement rather than emitting near-band billboards.
        // Billboards live in the FAR ring only; the near→far mip transition is the
        // only LOD swap. Bigger pools keep this branch rare.
        continue;
      }
      t._used = true;
      t.position.set(p.lx, p.ly, p.lz);
      t.rotation.y = p.ry;
      t.scale.setScalar(p.scale);
      t.matrixWorldAutoUpdate = true;
      t.matrixAutoUpdate = true;
      claimed.push(t);
    }
    return { lod: 0, placements, claimedTrees: claimed, billboardMeshes: null };
  }

  function _assembleFarChunk(cx, cz) {
    const placements = _placementsForCell(cx, cz);
    return { lod: 1, placements, claimedTrees: null };
  }

  function _lodForChunk(cx, cz, camCx, camCz) {
    const dx = cx - camCx, dz = cz - camCz;
    const dmax = Math.max(Math.abs(dx), Math.abs(dz));
    if (dmax <= TREE_NEAR_CELLS) return 0;
    if (dmax <= TREE_FAR_CELLS) return 1;
    return -1;
  }

  // ---- Closest-N streaming model ----
  // Walk wanted ring, sort placements by distance to player, claim pool by rank,
  // billboard the rest. Trigger only on chunk-cross.
  let _lastCamCx = NaN, _lastCamCz = NaN;
  const _tmpBbM = new THREE.Matrix4();
  function _bbRemoveAll() {
    for (const entry of _billboardByPreset.values()) {
      entry.highWater = 0;
      entry.freeSlots.length = 0;
      entry.mesh.count = 0;
      entry.dirty = true;
    }
  }
  let _bbFallbackEntry = null;
  // Matrix carries position only — identity scale. instanceSize is the only
  // size source the shader reads; scale-on-matrix would double-multiply.
  function _bbPlace(p) {
    let entry = _billboardByPreset.get(p.presetName);
    if (!entry) {
      if (!_bbFallbackEntry && _billboardByPreset.size) {
        _bbFallbackEntry = _billboardByPreset.values().next().value;
      }
      entry = _bbFallbackEntry;
      if (!entry) { _streamStats.lostPlacements = (_streamStats.lostPlacements|0) + 1; return false; }
    }
    const slot = _bbAllocSlot(entry);
    if (slot < 0) { _streamStats.lostPlacements = (_streamStats.lostPlacements|0) + 1; return false; }
    _tmpBbM.identity();
    _tmpBbM.setPosition(p.lx, p.ly, p.lz);
    entry.mesh.setMatrixAt(slot, _tmpBbM);
    const o = slot * 2;
    entry.instSize.array[o]     = entry.size.x * p.scale;
    entry.instSize.array[o + 1] = entry.size.y * p.scale;
    entry.dirty = true;
    return true;
  }
  function _bbCommit() {
    for (const entry of _billboardByPreset.values()) {
      if (!entry.dirty) continue;
      entry.mesh.count = entry.highWater;
      entry.mesh.instanceMatrix.needsUpdate = true;
      entry.instSize.needsUpdate = true;
      entry.dirty = false;
    }
  }
  function streamTrees(camera) {
    _updateBillboardSun();
    // Center streaming on the PLAYER (controls.target), not the orbit camera.
    // The camera sits 14m behind the player on the orbit; using camera.position
    // shifts the streaming ring backwards relative to where the player actually is.
    const center = (controls && controls.target) ? controls.target : camera.position;
    const centerX = center.x, centerZ = center.z;
    const camCx = Math.floor(centerX / TREE_CELL);
    const camCz = Math.floor(centerZ / TREE_CELL);
    if (camCx === _lastCamCx && camCz === _lastCamCz && !_dirtyAllPlacements) return;
    _lastCamCx = camCx; _lastCamCz = camCz; _dirtyAllPlacements = false;
    const all = [];
    for (let dz = -TREE_FAR_CELLS; dz <= TREE_FAR_CELLS; dz++) {
      for (let dx = -TREE_FAR_CELLS; dx <= TREE_FAR_CELLS; dx++) {
        const cx = camCx + dx, cz = camCz + dz;
        for (const p of _placementsForCell(cx, cz)) {
          const ddx = p.lx - centerX, ddz = p.lz - centerZ;
          all.push({ p, d2: ddx * ddx + ddz * ddz });
        }
      }
    }
    all.sort((a, b) => a.d2 - b.d2);
    for (const list of treePool.byPreset.values()) {
      for (const t of list) { t._used = false; t.position.y = -10000; t.visible = false; }
    }
    _bbRemoveAll();
    _streamStats.lostPlacements = 0;
    const REAL_TREE_BUDGET = 60;
    let realClaimed = 0, billboarded = 0;
    for (const { p } of all) {
      if (realClaimed < REAL_TREE_BUDGET) {
        const list = treePool.byPreset.get(p.presetName);
        if (list && list.length) {
          const N = list.length;
          const start = _stablePoolIndex(p.presetName, p.lx, p.lz, N);
          let t = null;
          for (let k = 0; k < N; k++) {
            const idx = (start + k) % N;
            if (!list[idx]._used) { t = list[idx]; break; }
          }
          if (t) {
            t._used = true;
            t.visible = true;
            t.position.set(p.lx, p.ly, p.lz);
            t.rotation.y = p.ry;
            t.scale.setScalar(p.scale);
            t.matrixWorldAutoUpdate = true;
            t.matrixAutoUpdate = true;
            realClaimed++;
            continue;
          }
        }
      }
      if (_bbPlace(p)) billboarded++;
    }
    _bbCommit();
    _streamStats.realTrees = realClaimed;
    _streamStats.billboards = billboarded;
    _streamStats.totalPlacements = all.length;
    _streamStats.nearChunks = realClaimed;     // legacy fields kept for debug HUD
    _streamStats.billboardChunks = billboarded;
  }
  controls.streamTrees = streamTrees;
  streamTrees(camera);

  // §7 frustum cull via group bounding sphere — forest is a Group at origin
  // covering the whole streamed area.
  forest.frustumCulled = false; // children handle their own culling

  if (typeof window !== 'undefined') {
    window.__debug = window.__debug || {};
    window.__debug.veg = _streamStats;
    window.__debug.vegChunks = _chunks;
    window.__debug.vegBillboards = _billboardByPreset;
    window.__debug.vegPlacementCache = _placementCache;
    window.__debug.vegPool = treePool;
    window.__debug.vegBbMeshes = _billboardByPreset;
    window.__debug.vegCamera = camera;
    window.__debug.vegRenderer = renderer;
    window.__debug.terrainAPI = { heightAt, dominantBiomeAt, biomeWeightsAt };
    if (environment && environment.terrainSystem) {
      window.__debug.terrainSystem = environment.terrainSystem;
      window.__debug.terrainMesh = environment.terrainSystem.terrainMesh;
    }
    window.__debug.vegForceStream = () => { _dirtyAllPlacements = true; streamTrees(camera); };
    window.__debug.veg.assertNoOrphans = () => {
      const s = _streamStats;
      return (s.realTrees + s.billboards) === s.totalPlacements && (s.lostPlacements|0) === 0;
    };
  }

  scene.add(forest);

  // Freeze world-matrix auto-update for all static scene content. The camera
  // and any node with `userData.dynamic === true` keep auto-update. Async loaders
  // (Grass flowers, Rocks) attach after scene creation, so re-run after a delay.
  function freezeStatics() {
    scene.updateMatrixWorld(true);
    scene.traverse((o) => {
      if (o.userData && o.userData.dynamic) return;
      o.matrixWorldAutoUpdate = false;
      o.matrixAutoUpdate = false;
    });
  }
  freezeStatics();
  setTimeout(freezeStatics, 500);
  setTimeout(freezeStatics, 2000);

  return {
    scene,
    environment,
    tree,
    camera,
    controls
  }
}