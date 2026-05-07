import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { Tree } from 'mapspinner';
import { createScene } from './scene.js';
import { installDebugApi } from './debugApi.js';

/*
 * GPU debug tooling
 * -----------------
 * In-page (no install):
 *   - F2 : toggle render-target thumbnail viewer (window.__debug.rts registry)
 *   - F3 : toggle FPS / draw-call / triangle stats overlay
 *   - window.__debugSpector = 1 BEFORE reload : lazy-load Spector.js from CDN
 *     and add a "Capture Frame" button that opens a full WebGL frame report
 *     (https://spector.babylonjs.com/)
 *
 * Browser extension (one-time install):
 *   "three.js DevTools" — Chrome Web Store. Adds a Three.js panel to DevTools
 *   that walks the scene graph live and lets you inspect any Object3D/Material/
 *   Texture/Geometry. Works against window.__debug.scene with no extra setup.
 *   https://chrome.google.com/webstore/detail/threejs-developer-tools
 */

async function __mapspinnerBoot() {
  const container = document.getElementById('app')
  // Honor ?seed=N for deterministic harness runs
  const __qp = new URLSearchParams(window.location.search);
  if (__qp.has('seed')) window.__seed = parseInt(__qp.get('seed'), 10) || 1;
  if (__qp.has('leafScale')) window.__leafScale = parseFloat(__qp.get('leafScale')) || 1;
  if (__qp.has('leafSize')) window.__leafSizeScale = parseFloat(__qp.get('leafSize')) || 1;

  // User needs to interact with the page before audio will play
  container.addEventListener('click', () => window.toggleAudio?.());

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setClearColor(0);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.NoToneMapping;
  renderer.toneMappingExposure = 1;
  // Vox terrain.html (three r128) wrote frag output directly to the canvas
  // with no extra color-space transform. Modern three (r152+) defaults to
  // SRGBColorSpace which auto-encodes — but the terrain frag already produces
  // display-ready sRGB values (filmic shoulder baked in). Force linear so the
  // canvas sees frag output unchanged.
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const __dbgEarly = (window.__debug = window.__debug || {});
  __dbgEarly.rts = __dbgEarly.rts || [];
  __dbgEarly.registerRT = __dbgEarly.registerRT || ((name, rt) => {
    if (!rt) return;
    if (!__dbgEarly.rts.find(e => e.rt === rt)) __dbgEarly.rts.push({ name, rt });
  });

  const { scene, environment, tree, camera, controls } = await createScene(renderer);

  const composer = new EffectComposer(renderer);

  composer.addPass(new RenderPass(scene, camera));

  const smaaPass = new SMAAPass(
    container.clientWidth * renderer.getPixelRatio(),
    container.clientHeight * renderer.getPixelRatio());
  composer.addPass(smaaPass);

  // No OutputPass — its color-space conversion + tonemapping doubles up on
  // what the terrain frag already does. SMAAPass output goes straight to canvas.

  const clock = new THREE.Clock();
  const __dbg = (window.__debug = window.__debug || {});
  __dbg.fps = 0;
  __dbg.frame = 0;
  __dbg.ready = false;
  __dbg.scene = scene;
  __dbg.tree = tree;
  __dbg.camera = camera;
  __dbg.controls = controls;
  __dbg.renderer = renderer;
  __dbg.composer = composer;
  __dbg.THREE = THREE;
  __dbg.environment = environment;
  __dbg.Tree = Tree;
  let __fpsLast = performance.now();
  let __fpsFrames = 0;
  let __fpsAcc = 0;
  function animate() {
    const now = performance.now();
    const dt = now - __fpsLast;
    __fpsLast = now;
    __fpsAcc = __fpsAcc * 0.9 + dt * 0.1;
    __fpsFrames++;
    __dbg.fps = __fpsAcc > 0 ? 1000 / __fpsAcc : 0;
    __dbg.frame = __fpsFrames;
    const t = clock.getElapsedTime();
    Tree.updateAllShaders(t);
    environment.update(t, camera, camera.position);

    if (controls.updateWalk) controls.updateWalk(Math.min(0.1, dt / 1000));
    if (controls.streamTrees) controls.streamTrees(camera);
    controls.update();
    composer.render();
    if (window.__debug) {
      __updateStats();
      __updateRTOverlay();
    }
    requestAnimationFrame(animate);
  }
  __dbg.setCamera = (px, py, pz, tx, ty, tz) => {
    camera.position.set(px, py, pz);
    controls.target.set(tx, ty, tz);
    controls.update();
  };
  // Synchronous render-loop benchmark — bypasses vsync cap by calling composer.render()
  // back-to-back. Returns mean ms/frame; effective fps = 1000 / ms.
  __dbg.measureRenderMs = (frames = 120) => {
    // Warmup
    for (let i = 0; i < 10; i++) composer.render();
    const t0 = performance.now();
    for (let i = 0; i < frames; i++) {
      const t = clock.getElapsedTime();
      Tree.updateAllShaders(t);
      environment.update(t, camera, camera.position);
      composer.render();
    }
    // Force GPU sync by reading back a single pixel
    const gl = renderer.getContext();
    const px = new Uint8Array(4);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    const dt = performance.now() - t0;
    return { frames, ms: dt, msPerFrame: dt / frames, fps: (frames * 1000) / dt };
  };
  __dbg.measureFPS = (frames = 120) => new Promise((resolve) => {
    const t0 = performance.now();
    const f0 = __fpsFrames;
    const tick = () => {
      if (__fpsFrames - f0 >= frames) {
        const dt = performance.now() - t0;
        resolve({ frames, ms: dt, fps: (frames * 1000) / dt });
      } else requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
  __dbg.ready = true;

  installDebugApi({ renderer, scene, camera });

  const __statsEl = document.createElement('div');
  __statsEl.style.cssText = 'position:fixed;top:6px;left:6px;z-index:9999;background:rgba(0,0,0,0.55);color:#0f0;font:12px/1.3 monospace;padding:6px 8px;pointer-events:none;white-space:pre;border-radius:3px;';
  document.body.appendChild(__statsEl);
  let __statsVisible = true;
  function __updateStats() {
    if (!__statsVisible) return;
    const r = renderer.info.render;
    const m = renderer.info.memory;
    __statsEl.textContent =
      'fps  ' + __dbg.fps.toFixed(1) +
      '\nms   ' + __fpsAcc.toFixed(2) +
      '\ncalls ' + r.calls +
      '\ntris  ' + r.triangles +
      '\ngeom  ' + m.geometries +
      '\ntex   ' + m.textures;
  }
  __dbg.toggleStats = () => {
    __statsVisible = !__statsVisible;
    __statsEl.style.display = __statsVisible ? 'block' : 'none';
  };

  const __rtOverlay = document.createElement('div');
  __rtOverlay.style.cssText = 'position:fixed;top:6px;right:6px;z-index:9999;background:rgba(0,0,0,0.7);padding:6px;display:none;max-width:840px;border-radius:3px;';
  document.body.appendChild(__rtOverlay);
  let __rtVisible = false;
  const __rtCanvases = new Map();
  function __ensureRtCanvas(entry) {
    if (__rtCanvases.has(entry)) return __rtCanvases.get(entry);
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:inline-block;margin:4px;text-align:center;color:#0f0;font:11px monospace;';
    const c = document.createElement('canvas');
    c.width = 192; c.height = 192;
    c.style.cssText = 'background:#222;display:block;border:1px solid #444;';
    const lbl = document.createElement('div');
    lbl.textContent = entry.name;
    wrap.appendChild(c); wrap.appendChild(lbl);
    __rtOverlay.appendChild(wrap);
    const ctx = c.getContext('2d');
    const handle = { canvas: c, ctx, label: lbl };
    __rtCanvases.set(entry, handle);
    return handle;
  }
  function __drawRT(entry, handle) {
    const rt = entry.rt;
    if (!rt || !rt.texture) return;
    const w = Math.min(rt.width, 256);
    const h = Math.min(rt.height, 256);
    const isHalf = rt.texture.type === THREE.HalfFloatType;
    const isFloat = rt.texture.type === THREE.FloatType || isHalf;
    const buf = rt.texture.type === THREE.FloatType ? new Float32Array(w * h * 4)
              : isHalf ? new Uint16Array(w * h * 4)
              : new Uint8Array(w * h * 4);
    try {
      renderer.readRenderTargetPixels(rt, 0, 0, w, h, buf);
    } catch (e) {
      handle.label.textContent = entry.name + ' (readback failed)';
      return;
    }
    const out = handle.ctx.createImageData(192, 192);
    for (let y = 0; y < 192; y++) {
      for (let x = 0; x < 192; x++) {
        const sx = (x * w / 192) | 0;
        const sy = (y * h / 192) | 0;
        const si = (sy * w + sx) * 4;
        const di = ((191 - y) * 192 + x) * 4;
        if (isHalf) {
          const f = (u16) => {
            const s = (u16 >> 15) & 1, e = (u16 >> 10) & 0x1f, m = u16 & 0x3ff;
            if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
            if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
            return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
          };
          out.data[di] = Math.max(0, Math.min(255, f(buf[si]) * 255));
          out.data[di + 1] = Math.max(0, Math.min(255, f(buf[si + 1]) * 255));
          out.data[di + 2] = Math.max(0, Math.min(255, f(buf[si + 2]) * 255));
          out.data[di + 3] = 255;
        } else if (isFloat) {
          out.data[di] = Math.max(0, Math.min(255, buf[si] * 255));
          out.data[di + 1] = Math.max(0, Math.min(255, buf[si + 1] * 255));
          out.data[di + 2] = Math.max(0, Math.min(255, buf[si + 2] * 255));
          out.data[di + 3] = 255;
        } else {
          out.data[di] = buf[si];
          out.data[di + 1] = buf[si + 1];
          out.data[di + 2] = buf[si + 2];
          out.data[di + 3] = 255;
        }
      }
    }
    handle.ctx.putImageData(out, 0, 0);
    handle.label.textContent = entry.name + ' ' + rt.width + 'x' + rt.height;
  }
  __dbg.toggleRTOverlay = () => {
    __rtVisible = !__rtVisible;
    __rtOverlay.style.display = __rtVisible ? 'block' : 'none';
  };
  function __updateRTOverlay() {
    if (!__rtVisible) return;
    for (const entry of __dbg.rts) {
      const h = __ensureRtCanvas(entry);
      __drawRT(entry, h);
    }
  }

  // ---- Tooltip used by pixel inspector + live probe ----
  const __tip = document.createElement('div');
  __tip.style.cssText = 'position:fixed;left:0;top:0;z-index:10000;background:rgba(0,0,0,0.82);color:#0f0;font:11px/1.35 monospace;padding:6px 8px;border:1px solid #0f0;border-radius:3px;pointer-events:none;display:none;white-space:pre;max-width:360px;';
  document.body.appendChild(__tip);
  function __showTip(x, y, text) {
    __tip.style.left = Math.min(window.innerWidth - 380, x + 14) + 'px';
    __tip.style.top = Math.min(window.innerHeight - 200, y + 14) + 'px';
    __tip.textContent = text;
    __tip.style.display = 'block';
  }
  function __hideTip() { __tip.style.display = 'none'; }

  function __sampleWorld(wx, wz) {
    const t = __dbg.terrainAPI;
    if (!t) return null;
    const h = t.heightAt(wx, wz);
    const dom = t.dominantBiomeAt(wx, wz);
    const w = t.biomeWeightsAt(wx, wz);
    return { h, biome: dom.name, weights: w };
  }
  function __readTMatPixel(wx, wz) {
    // Read tMat RT at the bake-UV for this world point. tMat is polar around
    // uSampleCameraPos with radius uMaxR mapped to UV unit circle.
    const ts = __dbg.terrainSystem;
    if (!ts || !ts.tMatRT) return null;
    const sp = ts.uniforms.uSampleCameraPos.value;
    const maxR = ts.uniforms.uMaxR.value;
    const dx = wx - sp.x, dz = wz - sp.z;
    const r = Math.sqrt(dx * dx + dz * dz) / maxR;
    if (r > 1) return null;
    const theta = Math.atan2(dz, dx);
    const u = 0.5 + 0.5 * r * Math.cos(theta);
    const v = 0.5 + 0.5 * r * Math.sin(theta);
    const W = ts.tMatRT.width, H = ts.tMatRT.height;
    const px = Math.max(0, Math.min(W - 1, Math.floor(u * W)));
    const py = Math.max(0, Math.min(H - 1, Math.floor(v * H)));
    const isHalf = ts.tMatRT.texture.type === THREE.HalfFloatType;
    const buf = isHalf ? new Uint16Array(4) : new Uint8Array(4);
    try { renderer.readRenderTargetPixels(ts.tMatRT, px, py, 1, 1, buf); }
    catch (e) { return null; }
    if (isHalf) {
      const f = (u16) => {
        const s = (u16 >> 15) & 1, e = (u16 >> 10) & 0x1f, m = u16 & 0x3ff;
        if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
        if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
      };
      return [f(buf[0]), f(buf[1]), f(buf[2]), f(buf[3])];
    }
    return [buf[0] / 255, buf[1] / 255, buf[2] / 255, buf[3] / 255];
  }
  function __formatSample(s, tMat, screenRGBA) {
    const fmt = (n) => (n === null || n === undefined) ? '-' : (typeof n === 'number' ? n.toFixed(3) : String(n));
    let out = '';
    if (screenRGBA) out += 'screen RGBA  ' + screenRGBA.map(n => n.toString().padStart(3)).join(' ') + '\n';
    if (s) {
      out += 'world height ' + fmt(s.h) + '\n';
      out += 'biome        ' + s.biome + '\n';
      out += 'slope        ' + fmt(s.weights.slope) + '\n';
      out += 'rock         ' + fmt(s.weights.rock) + '\n';
      out += 'grass        ' + fmt(s.weights.grass) + '\n';
      out += 'snow         ' + fmt(s.weights.snow) + '\n';
      out += 'forest       ' + fmt(s.weights.forest) + '\n';
    }
    if (tMat) out += 'tMat RGBA    ' + tMat.map(fmt).join(' ');
    return out;
  }

  // ---- Shift+click pixel inspector ----
  const __ray = new THREE.Raycaster();
  const __ndc = new THREE.Vector2();
  function __raycastTerrain(clientX, clientY) {
    const tm = __dbg.terrainMesh;
    if (!tm) return null;
    const rect = renderer.domElement.getBoundingClientRect();
    __ndc.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    __ndc.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    __ray.setFromCamera(__ndc, camera);
    const hits = __ray.intersectObject(tm, false);
    return hits.length ? hits[0].point : null;
  }
  renderer.domElement.addEventListener('mousedown', (ev) => {
    if (!ev.shiftKey) return;
    ev.preventDefault();
    const rect = renderer.domElement.getBoundingClientRect();
    const cssX = ev.clientX - rect.left, cssY = ev.clientY - rect.top;
    const dpr = renderer.getPixelRatio();
    const px = Math.floor(cssX * dpr);
    const py = Math.floor((rect.height - cssY) * dpr);
    const gl = renderer.getContext();
    const buf = new Uint8Array(4);
    try { gl.readPixels(px, py, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, buf); } catch (e) {}
    const hit = __raycastTerrain(ev.clientX, ev.clientY);
    let world = null, tMat = null;
    if (hit) {
      world = __sampleWorld(hit.x, hit.z);
      tMat = __readTMatPixel(hit.x, hit.z);
      world.x = hit.x; world.z = hit.z;
    }
    const text =
      'PIXEL INSPECTOR  (shift+click)\n' +
      (hit ? `world XZ ${hit.x.toFixed(2)} ${hit.z.toFixed(2)}\n` : 'no terrain hit\n') +
      __formatSample(world, tMat, [buf[0], buf[1], buf[2], buf[3]]);
    __showTip(ev.clientX, ev.clientY, text);
  });
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') __hideTip();
  });

  // ---- F4 live mouse-follow probe ----
  let __probeOn = false;
  function __probeMove(ev) {
    if (!__probeOn) return;
    const hit = __raycastTerrain(ev.clientX, ev.clientY);
    if (!hit) { __hideTip(); return; }
    const s = __sampleWorld(hit.x, hit.z);
    const tMat = __readTMatPixel(hit.x, hit.z);
    s.x = hit.x; s.z = hit.z;
    __showTip(ev.clientX, ev.clientY,
      'PROBE  (F4 toggles)\n' +
      `world XZ ${hit.x.toFixed(2)} ${hit.z.toFixed(2)}\n` +
      __formatSample(s, tMat, null));
  }
  window.addEventListener('mousemove', __probeMove);
  __dbg.toggleProbe = () => {
    __probeOn = !__probeOn;
    if (!__probeOn) __hideTip();
  };

  // ---- F6 shader uniform live editor ----
  const __uniPanel = document.createElement('div');
  __uniPanel.style.cssText = 'position:fixed;right:6px;bottom:6px;z-index:9999;width:340px;max-height:60vh;overflow:auto;background:rgba(0,0,0,0.85);color:#0f0;font:11px monospace;padding:6px 8px;border:1px solid #0f0;border-radius:3px;display:none;';
  document.body.appendChild(__uniPanel);
  let __uniVisible = false;
  function __collectUniformSources() {
    const out = [];
    const seen = new Set();
    function add(label, uniforms) {
      if (!uniforms || seen.has(uniforms)) return;
      seen.add(uniforms);
      out.push({ label, uniforms });
    }
    scene.traverse((o) => {
      const m = o.material;
      if (!m) return;
      const mats = Array.isArray(m) ? m : [m];
      for (const mat of mats) {
        if (mat.userData && mat.userData.shader && mat.userData.shader.uniforms) {
          add((mat.name || mat.type || 'mat') + ' [onBeforeCompile]', mat.userData.shader.uniforms);
        }
        if (mat.uniforms) add((mat.name || mat.type || 'ShaderMaterial'), mat.uniforms);
      }
    });
    const ts = __dbg.terrainSystem;
    if (ts) {
      if (ts.uniforms) add('terrainSystem.uniforms', ts.uniforms);
      if (ts.bakeUniforms) add('terrainSystem.bakeUniforms', ts.bakeUniforms);
    }
    if (__dbg.environment && __dbg.environment.grass) {
      const g = __dbg.environment.grass;
      if (g.material && g.material.uniforms) add('grass.material.uniforms', g.material.uniforms);
    }
    return out;
  }
  function __renderUniformPanel() {
    __uniPanel.innerHTML = '';
    const sources = __collectUniformSources();
    const head = document.createElement('div');
    head.textContent = 'UNIFORM EDITOR  (F6)  sources=' + sources.length;
    head.style.cssText = 'color:#0ff;border-bottom:1px solid #0f0;padding-bottom:3px;margin-bottom:4px;';
    __uniPanel.appendChild(head);
    for (const src of sources) {
      const sec = document.createElement('details');
      sec.style.cssText = 'margin:4px 0;';
      const sum = document.createElement('summary');
      sum.textContent = src.label + '  (' + Object.keys(src.uniforms).length + ')';
      sum.style.cssText = 'cursor:pointer;color:#fc0;';
      sec.appendChild(sum);
      for (const k of Object.keys(src.uniforms).sort()) {
        const u = src.uniforms[k];
        if (!u || u.value === undefined || u.value === null) continue;
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;gap:4px;align-items:center;margin:1px 0;';
        const label = document.createElement('span');
        label.textContent = k;
        label.style.cssText = 'flex:1;color:#9f9;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
        row.appendChild(label);
        const v = u.value;
        const mkInput = (initial, on) => {
          const i = document.createElement('input');
          i.type = 'text';
          i.value = String(initial);
          i.style.cssText = 'width:60px;background:#111;color:#0f0;border:1px solid #333;font:11px monospace;';
          i.addEventListener('change', () => { try { on(parseFloat(i.value)); } catch (e) {} });
          return i;
        };
        if (typeof v === 'number') {
          row.appendChild(mkInput(v, (n) => { u.value = n; }));
        } else if (v && typeof v.x === 'number' && typeof v.y === 'number' && v.z === undefined) {
          row.appendChild(mkInput(v.x, (n) => { u.value.x = n; }));
          row.appendChild(mkInput(v.y, (n) => { u.value.y = n; }));
        } else if (v && typeof v.x === 'number' && typeof v.z === 'number' && v.w === undefined) {
          row.appendChild(mkInput(v.x, (n) => { u.value.x = n; }));
          row.appendChild(mkInput(v.y, (n) => { u.value.y = n; }));
          row.appendChild(mkInput(v.z, (n) => { u.value.z = n; }));
        } else if (v && typeof v.w === 'number') {
          row.appendChild(mkInput(v.x, (n) => { u.value.x = n; }));
          row.appendChild(mkInput(v.y, (n) => { u.value.y = n; }));
          row.appendChild(mkInput(v.z, (n) => { u.value.z = n; }));
          row.appendChild(mkInput(v.w, (n) => { u.value.w = n; }));
        } else if (v && v.isColor) {
          const i = document.createElement('input');
          i.type = 'color';
          i.value = '#' + v.getHexString();
          i.addEventListener('input', () => { u.value.set(i.value); });
          row.appendChild(i);
        } else {
          const span = document.createElement('span');
          span.textContent = '<' + (v && v.constructor ? v.constructor.name : typeof v) + '>';
          span.style.color = '#666';
          row.appendChild(span);
        }
        sec.appendChild(row);
      }
      __uniPanel.appendChild(sec);
    }
  }
  __dbg.toggleUniformEditor = () => {
    __uniVisible = !__uniVisible;
    if (__uniVisible) __renderUniformPanel();
    __uniPanel.style.display = __uniVisible ? 'block' : 'none';
  };

  window.addEventListener('keydown', (e) => {
    if (e.key === 'F2') { e.preventDefault(); __dbg.toggleRTOverlay(); }
    else if (e.key === 'F3') { e.preventDefault(); __dbg.toggleStats(); }
    else if (e.key === 'F4') { e.preventDefault(); __dbg.toggleProbe(); }
    else if (e.key === 'F6') { e.preventDefault(); __dbg.toggleUniformEditor(); }
  });

  if (window.__debugSpector) {
    const btn = document.createElement('button');
    btn.textContent = 'Capture Frame (Spector)';
    btn.style.cssText = 'position:fixed;bottom:6px;left:6px;z-index:9999;padding:8px 12px;font:13px monospace;background:#222;color:#0f0;border:1px solid #0f0;cursor:pointer;border-radius:3px;';
    btn.onclick = async () => {
      btn.textContent = 'Loading Spector...';
      try {
        await new Promise((res, rej) => {
          const s = document.createElement('script');
          s.src = 'https://cdn.jsdelivr.net/npm/spectorjs/dist/spector.bundled.js';
          s.onload = res; s.onerror = rej;
          document.head.appendChild(s);
        });
        const spector = new window.SPECTOR.Spector();
        spector.displayUI();
        spector.captureContext(renderer.getContext());
        btn.textContent = 'Capture Frame (Spector)';
      } catch (err) {
        btn.textContent = 'Spector load failed';
        throw err;
      }
    };
    document.body.appendChild(btn);
  }

  function resize() {
    renderer.setSize(container.clientWidth, container.clientHeight);
    smaaPass.setSize(container.clientWidth, container.clientHeight);
    composer.setSize(container.clientWidth, container.clientHeight);
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
  }

  window.addEventListener('resize', resize);

  animate();
  resize();

  document.getElementById('audio-status').style.display = 'block';
}

__mapspinnerBoot();

window.toggleAudio = function () {
  document.getElementById('app').removeEventListener('click', toggleAudio);

  if (window.isAudioPlaying) {
    window.isAudioPlaying = false;
    document.getElementById('audio-status').src = "icon_muted.png";
    document.getElementById('background-audio').pause();
  } else {
    window.isAudioPlaying = true;
    document.getElementById('audio-status').src = "icon_playing.png";
    document.getElementById('background-audio').play();
  }
}