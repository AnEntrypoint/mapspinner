import * as THREE from 'three';

const f16to32 = (u16) => {
  const s = (u16 >> 15) & 1, e = (u16 >> 10) & 0x1f, m = u16 & 0x3ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
  if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
};

function readPixel(renderer, rt, x, y) {
  const tex = rt.texture;
  const isHalf = tex.type === THREE.HalfFloatType;
  const isFloat = tex.type === THREE.FloatType;
  const buf = isFloat ? new Float32Array(4) : isHalf ? new Uint16Array(4) : new Uint8Array(4);
  try { renderer.readRenderTargetPixels(rt, x, y, 1, 1, buf); } catch (e) { return null; }
  if (isHalf) return { r: f16to32(buf[0]), g: f16to32(buf[1]), b: f16to32(buf[2]), a: f16to32(buf[3]) };
  if (isFloat) return { r: buf[0], g: buf[1], b: buf[2], a: buf[3] };
  return { r: buf[0] / 255, g: buf[1] / 255, b: buf[2] / 255, a: buf[3] / 255 };
}

function findRT(name) {
  const rts = (window.__debug && window.__debug.rts) || [];
  const e = rts.find(x => x.name === name);
  return e ? e.rt : null;
}

function worldToBakeUV(wx, wz) {
  const ts = window.__debug && window.__debug.terrainSystem;
  if (!ts) return null;
  const sp = ts.uniforms.uSampleCameraPos.value;
  const maxR = ts.uniforms.uMaxR.value;
  const dx = wx - sp.x, dz = wz - sp.z;
  const r = Math.sqrt(dx * dx + dz * dz) / maxR;
  if (r > 1) return null;
  const theta = Math.atan2(dz, dx);
  return { u: 0.5 + 0.5 * r * Math.cos(theta), v: 0.5 + 0.5 * r * Math.sin(theta) };
}

function materialKey(o, mat) {
  const uuid = (mat.uuid || '').replace(/-/g, '').slice(0, 8);
  return (o.name || mat.name || mat.type || 'mat') + '_' + uuid;
}

function collectMaterials() {
  const scene = window.__debug && window.__debug.scene;
  const out = [];
  if (!scene) return out;
  const counts = new Map();
  scene.traverse((o) => {
    if (!o.material) return;
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    for (const m of mats) {
      const key = materialKey(o, m);
      counts.set(key, (counts.get(key) || 0) + 1);
      out.push({ key, name: m.name || o.name || '', type: m.type, material: m, owner: o });
    }
  });
  for (const e of out) e.drawCount = counts.get(e.key) || 1;
  const seen = new Set();
  return out.filter(e => { if (seen.has(e.key)) return false; seen.add(e.key); return true; });
}

function findMaterial(nameOrKey) {
  const all = collectMaterials();
  return all.find(e => e.key === nameOrKey || e.name === nameOrKey || e.material.name === nameOrKey)
    || all.find(e => (e.name || '').toLowerCase().includes(nameOrKey.toLowerCase()));
}

function uniformValueDescriptor(v) {
  if (v == null) return v;
  if (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string') return v;
  if (Array.isArray(v)) return v.slice();
  if (v.isColor) return { __type: 'Color', hex: '#' + v.getHexString(), r: v.r, g: v.g, b: v.b };
  if (v.isVector2) return { __type: 'Vector2', x: v.x, y: v.y };
  if (v.isVector3) return { __type: 'Vector3', x: v.x, y: v.y, z: v.z };
  if (v.isVector4) return { __type: 'Vector4', x: v.x, y: v.y, z: v.z, w: v.w };
  if (v.isMatrix3 || v.isMatrix4) return { __type: v.type || 'Matrix', elements: Array.from(v.elements) };
  if (v.isTexture) return { __type: 'Texture', uuid: v.uuid, image: v.image ? { width: v.image.width, height: v.image.height } : null };
  if (v.isWebGLRenderTarget) return { __type: 'RenderTarget', uuid: v.uuid, width: v.width, height: v.height };
  if (typeof v === 'function') return { __type: 'Function', name: v.name };
  if (typeof v === 'object') {
    const out = {};
    for (const k of Object.keys(v)) {
      if (typeof v[k] === 'function') continue;
      try { out[k] = uniformValueDescriptor(v[k]); } catch (e) { out[k] = '<err>'; }
    }
    return out;
  }
  return String(v);
}

function getUniforms(mat) {
  if (mat.userData && mat.userData.shader && mat.userData.shader.uniforms) return mat.userData.shader.uniforms;
  if (mat.uniforms) return mat.uniforms;
  return null;
}

function djb2(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) h = ((h << 5) + h + str.charCodeAt(i)) | 0;
  return (h >>> 0).toString(16);
}

export function installDebugApi({ renderer, scene, camera }) {
  const __dbg = (window.__debug = window.__debug || {});
  __dbg.scene = __dbg.scene || scene;
  __dbg.renderer = __dbg.renderer || renderer;
  __dbg.camera = __dbg.camera || camera;

  /** Sample a registered RT at integer texel (x,y). Returns {r,g,b,a} decoded. */
  const sampleRT = (name, x, y) => {
    const rt = findRT(name);
    if (!rt) return null;
    const px = Math.max(0, Math.min(rt.width - 1, Math.floor(x)));
    const py = Math.max(0, Math.min(rt.height - 1, Math.floor(y)));
    return readPixel(renderer, rt, px, py);
  };

  /** Sample a registered RT using world XZ via the polar bake UV mapping. */
  const sampleRTWorld = (name, worldX, worldZ) => {
    const rt = findRT(name);
    if (!rt) return null;
    const uv = worldToBakeUV(worldX, worldZ);
    if (!uv) return null;
    const px = Math.max(0, Math.min(rt.width - 1, Math.floor(uv.u * rt.width)));
    const py = Math.max(0, Math.min(rt.height - 1, Math.floor(uv.v * rt.height)));
    return readPixel(renderer, rt, px, py);
  };

  /** Read a single pixel from the rendered canvas at screen (cssX,cssY). */
  const sampleScreen = (screenX, screenY) => {
    const gl = renderer.getContext();
    const dpr = renderer.getPixelRatio();
    const rect = renderer.domElement.getBoundingClientRect();
    const px = Math.floor(screenX * dpr);
    const py = Math.floor((rect.height - screenY) * dpr);
    const buf = new Uint8Array(4);
    try { gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf); } catch (e) { return null; }
    return { r: buf[0] / 255, g: buf[1] / 255, b: buf[2] / 255, a: buf[3] / 255 };
  };

  /** Return all uniforms of a named/keyed material as plain JS values. */
  const dumpUniforms = (materialName) => {
    const e = findMaterial(materialName);
    if (!e) return null;
    const u = getUniforms(e.material);
    if (!u) return {};
    const out = {};
    for (const k of Object.keys(u)) {
      const entry = u[k];
      if (!entry) continue;
      out[k] = uniformValueDescriptor(entry.value);
    }
    return out;
  };

  /** Mutate one uniform live. value: number|[..]|hexString. */
  const setUniform = (materialName, uniformName, value) => {
    const e = findMaterial(materialName);
    if (!e) return false;
    const u = getUniforms(e.material);
    if (!u || !u[uniformName]) return false;
    const slot = u[uniformName];
    const cur = slot.value;
    if (typeof cur === 'number' || cur == null) { slot.value = +value; return true; }
    if (Array.isArray(value)) {
      if (cur && cur.isVector2 && value.length >= 2) { cur.set(value[0], value[1]); return true; }
      if (cur && cur.isVector3 && value.length >= 3) { cur.set(value[0], value[1], value[2]); return true; }
      if (cur && cur.isVector4 && value.length >= 4) { cur.set(value[0], value[1], value[2], value[3]); return true; }
      if (cur && cur.isColor && value.length >= 3) { cur.setRGB(value[0], value[1], value[2]); return true; }
      slot.value = value.slice();
      return true;
    }
    if (typeof value === 'string' && cur && cur.isColor) { cur.set(value); return true; }
    slot.value = value;
    return true;
  };

  /** Capture next-frame draw call summary (Promise). */
  const captureFrame = () => new Promise((resolve) => {
    const calls = [];
    const orig = renderer.render.bind(renderer);
    const gl = renderer.getContext();
    const origDrawElem = gl.drawElements.bind(gl);
    const origDrawArr = gl.drawArrays.bind(gl);
    const origDrawElemI = gl.drawElementsInstanced ? gl.drawElementsInstanced.bind(gl) : null;
    const origDrawArrI = gl.drawArraysInstanced ? gl.drawArraysInstanced.bind(gl) : null;
    let perCall = [];
    gl.drawElements = function (mode, count) { perCall.push({ kind: 'elem', count, instances: 1 }); return origDrawElem.apply(gl, arguments); };
    gl.drawArrays = function (mode, first, count) { perCall.push({ kind: 'arr', count, instances: 1 }); return origDrawArr.apply(gl, arguments); };
    if (origDrawElemI) gl.drawElementsInstanced = function (mode, count, type, off, instances) { perCall.push({ kind: 'elemI', count, instances }); return origDrawElemI.apply(gl, arguments); };
    if (origDrawArrI) gl.drawArraysInstanced = function (mode, first, count, instances) { perCall.push({ kind: 'arrI', count, instances }); return origDrawArrI.apply(gl, arguments); };
    const mats = collectMaterials();
    renderer.render = function (s, c) {
      const before = renderer.info.render.calls;
      const r = orig(s, c);
      const drewCalls = renderer.info.render.calls - before;
      calls.push({ scene: s.name || 'scene', camera: c.type, drawCalls: drewCalls });
      return r;
    };
    requestAnimationFrame(() => {
      // Restore on the next frame after render has happened
      setTimeout(() => {
        renderer.render = orig;
        gl.drawElements = origDrawElem;
        gl.drawArrays = origDrawArr;
        if (origDrawElemI) gl.drawElementsInstanced = origDrawElemI;
        if (origDrawArrI) gl.drawArraysInstanced = origDrawArrI;
        const totalCount = perCall.reduce((s, c) => s + c.count * (c.instances || 1), 0);
        const triangles = Math.floor(totalCount / 3);
        const uniformDigest = djb2(mats.map(m => m.key + ':' + Object.keys(getUniforms(m.material) || {}).length).join('|'));
        resolve({
          renderCalls: calls,
          drawCallCount: perCall.length,
          triangleCount: triangles,
          materialCount: mats.length,
          uniformDigest,
          drawCalls: perCall,
          info: {
            calls: renderer.info.render.calls,
            triangles: renderer.info.render.triangles,
            geometries: renderer.info.memory.geometries,
            textures: renderer.info.memory.textures,
          },
        });
      }, 16);
    });
  });

  /** Return compiled vertex+fragment shader source for a named material. */
  const shaderSource = (materialName) => {
    const e = findMaterial(materialName);
    if (!e) return null;
    const m = e.material;
    if (m.userData && m.userData.shader) return { vertex: m.userData.shader.vertexShader, fragment: m.userData.shader.fragmentShader, source: 'onBeforeCompile' };
    return { vertex: m.vertexShader || null, fragment: m.fragmentShader || null, source: 'raw' };
  };

  /** Return any GLSL compile/link log for the material's program. */
  const compileLog = (materialName) => {
    const e = findMaterial(materialName);
    if (!e) return null;
    const props = renderer.properties.get(e.material);
    const program = props && props.currentProgram ? props.currentProgram : (props && props.programs ? Array.from(props.programs.values())[0] : null);
    if (!program) return { error: 'no program' };
    const gl = renderer.getContext();
    const p = program.program;
    if (!p) return { error: 'no GL program' };
    return {
      programLog: gl.getProgramInfoLog(p) || '',
      vertexLog: program.vertexShader ? (gl.getShaderInfoLog(program.vertexShader) || '') : '',
      fragmentLog: program.fragmentShader ? (gl.getShaderInfoLog(program.fragmentShader) || '') : '',
    };
  };

  /** Enumerate all materials in the scene. */
  const materialList = () => collectMaterials().map(e => ({
    key: e.key, name: e.name, type: e.type, drawCount: e.drawCount, ownerName: e.owner.name || '', ownerType: e.owner.type
  }));

  /** Wrap fn in performance.now(); return {ms, result}. Pre-canned passNames: grassDraw, terrainBake, mainRender. */
  const bench = (passName, fn) => {
    const preCanned = {
      mainRender: () => { renderer.render(scene, camera); },
      grassDraw: () => {
        const g = window.__debug && window.__debug.environment && window.__debug.environment.grass;
        if (g) renderer.render(g, camera);
      },
      terrainBake: () => {
        const ts = window.__debug && window.__debug.terrainSystem;
        if (ts && typeof ts.bake === 'function') ts.bake();
      },
    };
    const f = typeof fn === 'function' ? fn : preCanned[passName];
    if (!f) return { error: 'no fn or pass', passName };
    const t0 = performance.now();
    let result = null, err = null;
    try { result = f(); } catch (e) { err = e.message; }
    const ms = performance.now() - t0;
    return { passName, ms, error: err };
  };

  /** Throw if predicate(state) returns false. */
  const assert = (predicate, msg) => {
    const state = {
      scene: window.__debug.scene,
      veg: window.__debug.veg,
      rocksStats: window.__debug.rocksStats,
      grassStats: window.__debug.grassStats,
    };
    let ok = false;
    try { ok = !!predicate(state); } catch (e) { throw new Error('assert predicate threw: ' + e.message); }
    if (!ok) throw new Error('assert failed: ' + (msg || 'no message'));
    return true;
  };

  /** Snapshot a registered observable. Names: streamingStats, rocksStats, grassStats. */
  const observable = (name) => {
    const d = window.__debug || {};
    if (name === 'streamingStats') return d.veg ? JSON.parse(JSON.stringify(d.veg)) : null;
    if (name === 'rocksStats') return d.rocksStats ? JSON.parse(JSON.stringify(d.rocksStats)) : null;
    if (name === 'grassStats') return d.grassStats ? JSON.parse(JSON.stringify(d.grassStats)) : null;
    return null;
  };

  /** List every registered RT with format metadata. */
  const allRTs = () => {
    const rts = (window.__debug && window.__debug.rts) || [];
    return rts.map(({ name, rt }) => {
      const tex = rt.texture;
      const channels = 4;
      const bytesPerChannel = tex.type === THREE.FloatType ? 4 : tex.type === THREE.HalfFloatType ? 2 : 1;
      return {
        name,
        width: rt.width,
        height: rt.height,
        format: tex.format,
        type: tex.type,
        byteSize: rt.width * rt.height * channels * bytesPerChannel,
      };
    });
  };

  /** Sample the height field the same way terrain mesh does. */
  const sampleHeightField = (worldX, worldZ) => {
    const t = window.__debug && window.__debug.terrainAPI;
    if (t && typeof t.heightAt === 'function') return t.heightAt(worldX, worldZ);
    return null;
  };

  /** Help: list all APIs. */
  const help = () => Object.keys(window.__gpu).filter(k => k !== 'help').sort();

  window.__gpu = {
    sampleRT, sampleRTWorld, sampleScreen,
    dumpUniforms, setUniform,
    captureFrame, shaderSource, compileLog,
    materialList, bench, assert,
    observable, allRTs, sampleHeightField,
    help,
  };
  return window.__gpu;
}
