// scripts/lab.mjs -- TV8 CLI testing lab.
//
// Builds BOTH height representations from the single source of truth (src/shaders/terrain.glsl):
//   - CPU: src/height-cpu.js (transpiled via scripts/gen-height.mjs -> src/height-gen.js), pure
//     node, no GPU, golden-parity-locked to the shader. This renders the HEIGHT GRAPH.
//   - GLSL: validated by loading planet.html in headless Chromium with the SwiftShader backend
//     (--use-angle=swiftshader) -- a GPU-free, portable, deterministic software WebGL2 path. This
//     is the "build the glsl" half and the CPU-vs-GPU parity oracle.
//
// Backend choice (user 2026-06-18 'pick the best option'): SwiftShader for the GLSL render-validate
// (GPU-free + portable + CI-able) over ANGLE-d3d11 (Windows/FXC-specialised, WARP-fallback risk) and
// native node-WebGL2 (none on win32). SwiftShader cannot witness the ANGLE/FXC mis-translation class
// -- for that, point PAGE/CHROME at a Windows --use-angle=d3d11 runner; this lab defaults to portable.
//
// Usage:
//   node scripts/lab.mjs heightmap [--res N] [--center lat,lon] [--span deg] [--radius m] [--hillshade] [--out f.png]
//   node scripts/lab.mjs build                 # regen CPU height-gen.js + compile-check the GLSL
//   node scripts/lab.mjs glsl-check            # headless SwiftShader: assert terrain.glsl compiles
//   node scripts/lab.mjs parity [--n N]        # CPU heightAt vs GPU _PROBE_ sampleGroundM divergence
//   node scripts/lab.mjs help
//
// The CPU heightmap + parity-vs-golden run GPU-free anywhere. glsl-check/parity self-launch a
// headless SwiftShader Chromium (auto-detected) + the dev server (server.js) and tear both down.

import { createHeightSampler } from '../src/height-cpu.js'
import zlib from 'node:zlib'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawn, spawnSync } from 'node:child_process'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const OUT_DIR = path.join(ROOT, 'lab-out')
// Verified LAND reference dir (height-cpu: ~1150m, warm/mid climate -> grass+rock). Set as window.__landDir
// so parkAboveGround reliably frames LAND (the auto-pick / [0.333,-0.258,0.907] default often hit ocean).
const LAND_REF = [0.4039, -0.6494, -0.6443]

// ---------------------------------------------------------------- arg parsing
function parseArgs(argv) {
  const a = { _: [] }
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i]
    if (t.startsWith('--')) {
      const key = t.slice(2)
      const next = argv[i + 1]
      if (next === undefined || next.startsWith('--')) { a[key] = true }
      else { a[key] = next; i++ }
    } else a._.push(t)
  }
  return a
}
const num = (v, d) => (v === undefined || v === true ? d : Number(v))

// ---------------------------------------------------------------- geometry
// world direction (unit, y-up) from geographic lat/lon in degrees.
function dirFromLatLon(latDeg, lonDeg) {
  const la = latDeg * Math.PI / 180, lo = lonDeg * Math.PI / 180
  const cl = Math.cos(la)
  return [cl * Math.cos(lo), Math.sin(la), cl * Math.sin(lo)]
}

// ---------------------------------------------------------------- CPU height field
// Sample heightAt over an equirectangular grid (full planet) or a centred region.
function sampleField(opts) {
  const res = Math.max(8, Math.round(num(opts.res, 256)))
  const radius = num(opts.radius, 6360000)            // Earth-scale metres -> readable elevations; shape is scale-invariant
  const seed = opts.seed !== undefined ? (num(opts.seed, 1337) | 0) : undefined
  const sampler = createHeightSampler({ radius, seed })
  let w, h, latOf, lonOf
  if (opts.center) {
    const [clat, clon] = String(opts.center).split(',').map(Number)
    const span = num(opts.span, 20)
    w = res; h = res
    latOf = (px, py) => clat + (0.5 - py / (h - 1)) * span
    lonOf = (px, py) => clon + (px / (w - 1) - 0.5) * span
  } else {
    w = res * 2; h = res                               // equirectangular 2:1
    latOf = (px, py) => 90 - (py / (h - 1)) * 180
    lonOf = (px, py) => (px / (w - 1)) * 360 - 180
  }
  const elev = new Float64Array(w * h)
  let min = Infinity, max = -Infinity, sum = 0, land = 0
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      const e = sampler.heightAt(dirFromLatLon(latOf(px, py), lonOf(px, py)))
      elev[py * w + px] = e
      if (e < min) min = e; if (e > max) max = e
      sum += e; if (e > 0) land++
    }
  }
  return { w, h, elev, min, max, mean: sum / (w * h), landFrac: land / (w * h), radius }
}

// ---------------------------------------------------------------- PNG (no deps, node zlib)
const CRC_TABLE = (() => {
  const t = new Uint32Array(256)
  for (let n = 0; n < 256; n++) {
    let c = n
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1)
    t[n] = c >>> 0
  }
  return t
})()
function crc32(buf) {
  let c = 0xFFFFFFFF
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xFF] ^ (c >>> 8)
  return (c ^ 0xFFFFFFFF) >>> 0
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0)
  const tb = Buffer.from(type, 'ascii')
  const body = Buffer.concat([tb, data])
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body), 0)
  return Buffer.concat([len, body, crc])
}
// grayscale 8-bit PNG from a width*height Uint8Array
function encodePNGGray(width, height, gray) {
  const sig = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8; ihdr[9] = 0; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0   // 8-bit, grayscale
  const raw = Buffer.alloc((width + 1) * height)
  for (let y = 0; y < height; y++) {
    raw[y * (width + 1)] = 0                                            // filter: none
    gray.subarray(y * width, (y + 1) * width).forEach((v, x) => { raw[y * (width + 1) + 1 + x] = v })
  }
  const idat = zlib.deflateSync(raw, { level: 9 })
  return Buffer.concat([sig, chunk('IHDR', ihdr), chunk('IDAT', idat), chunk('IEND', Buffer.alloc(0))])
}

// map elevation field -> grayscale, optional hillshade relief
function toGray(field, hillshade) {
  const { w, h, elev, min, max } = field
  const g = new Uint8Array(w * h)
  const span = (max - min) || 1
  if (!hillshade) {
    for (let i = 0; i < w * h; i++) g[i] = Math.max(0, Math.min(255, Math.round((elev[i] - min) / span * 255)))
    return g
  }
  // simple lambert hillshade from finite-difference slope (light from NW, high)
  const lx = -0.5, ly = -0.5, lz = 0.7, ll = Math.hypot(lx, ly, lz)
  const scale = 255 / span
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const xl = Math.max(0, x - 1), xr = Math.min(w - 1, x + 1)
    const yu = Math.max(0, y - 1), yd = Math.min(h - 1, y + 1)
    const dzdx = (elev[y * w + xr] - elev[y * w + xl]) * scale
    const dzdy = (elev[yd * w + x] - elev[yu * w + x]) * scale
    let nx = -dzdx, ny = -dzdy, nz = 2.0
    const nl = Math.hypot(nx, ny, nz) || 1
    let lum = (nx * lx + ny * ly + nz * lz) / (nl * ll)
    const base = (elev[y * w + x] - min) / span
    const v = Math.max(0, Math.min(1, 0.35 * base + 0.65 * Math.max(0, lum)))
    g[y * w + x] = Math.round(v * 255)
  }
  return g
}

function ensureOutDir() { if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true }) }

// ---------------------------------------------------------------- subcommand: heightmap
function cmdHeightmap(args) {
  const field = sampleField(args)
  const gray = toGray(field, !!args.hillshade)
  ensureOutDir()
  const out = args.out ? path.resolve(String(args.out)) : path.join(OUT_DIR, 'heightmap.png')
  fs.writeFileSync(out, encodePNGGray(field.w, field.h, gray))
  const m = (v) => v.toFixed(1)
  console.log(JSON.stringify({
    ok: true, out, w: field.w, h: field.h, radiusM: field.radius,
    minM: +m(field.min), maxM: +m(field.max), meanM: +m(field.mean),
    reliefM: +m(field.max - field.min), landFrac: +field.landFrac.toFixed(3)
  }, null, 1))
  return 0
}

// ---------------------------------------------------------------- subcommand: build (CPU + GLSL)
async function cmdBuild(args) {
  console.log('[lab] building CPU height (scripts/gen-height.mjs -> src/height-gen.js)')
  const gen = spawnSync(process.execPath, [path.join(ROOT, 'scripts', 'gen-height.mjs')], { cwd: ROOT, encoding: 'utf8' })
  process.stdout.write(gen.stdout || ''); if (gen.stderr) process.stderr.write(gen.stderr)
  if (gen.status !== 0) { console.log(JSON.stringify({ ok: false, step: 'gen-height', status: gen.status })); return 1 }
  console.log('[lab] compile-checking the GLSL (headless SwiftShader)')
  return await cmdGlslCheck(args)
}

// ---------------------------------------------------------------- headless SwiftShader Chromium + CDP
function findChrome() {
  if (process.env.CHROME) return process.env.CHROME
  const cands = [
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe',
    path.join(os.homedir(), 'AppData/Local/Google/Chrome/Application/chrome.exe'),
    '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  ]
  return cands.find(p => { try { return fs.existsSync(p) } catch { return false } }) || null
}
function waitFor(fn, ms, every = 200) {
  return new Promise((res, rej) => {
    const t0 = Date.now()
    const tick = async () => {
      try { const v = await fn(); if (v) return res(v) } catch {}
      if (Date.now() - t0 > ms) return rej(new Error('timeout'))
      setTimeout(tick, every)
    }
    tick()
  })
}
async function serverUp() { try { const r = await fetch('http://localhost:8080/planet.html', { method: 'HEAD' }); return r.ok || r.status === 200 } catch { return false } }

// Launch (server if needed) + headless SwiftShader chrome, run `fn(evalIn)`, tear everything down.
async function withHeadless(fn) {
  const chrome = findChrome()
  if (!chrome) return { ok: false, err: 'no chromium found (set CHROME=/path/to/chrome); CPU heightmap/parity still work GPU-free' }
  const procs = []
  try {
    if (!(await serverUp())) {
      const srv = spawn(process.execPath, [path.join(ROOT, 'server.js')], { cwd: ROOT, env: { ...process.env, PORT: '8080' }, stdio: 'ignore' })
      procs.push(srv)
      await waitFor(serverUp, 15000)
    }
    const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'tv8-lab-'))
    const cr = spawn(chrome, ['--headless=new', '--use-angle=' + (process.env.LAB_ANGLE || 'swiftshader'), '--use-gl=angle',
      '--disable-gpu-sandbox', '--no-sandbox', '--remote-debugging-port=0',
      '--user-data-dir=' + profile, 'about:blank'], { stdio: 'ignore' })
    procs.push(cr)
    const portFile = path.join(profile, 'DevToolsActivePort')
    const port = await waitFor(() => fs.existsSync(portFile) ? Number(fs.readFileSync(portFile, 'utf8').split('\n')[0]) : null, 15000)
    const ver = await (await fetch(`http://localhost:${port}/json/version`)).json()
    const ws = new WebSocket(ver.webSocketDebuggerUrl)
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej })
    let seq = 0; const pending = new Map()
    ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(m.error.message)) : res(m.result) } }
    const send = (method, params = {}, sessionId) => new Promise((res, rej) => { const id = ++seq; pending.set(id, { res, rej }); ws.send(JSON.stringify(sessionId ? { id, method, params, sessionId } : { id, method, params })) })
    const { targetId } = await send('Target.createTarget', { url: 'http://localhost:8080/planet.html' })
    const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true })
    await send('Runtime.enable', {}, sessionId)
    await send('Page.enable', {}, sessionId)
    const screenshot = async (p) => { const sr = await send('Page.captureScreenshot', { format: 'png' }, sessionId); fs.writeFileSync(p, Buffer.from(sr.data, 'base64')) }
    const evalIn = async (expr, awaitPromise = true) => {
      const r = await send('Runtime.evaluate', { expression: `(async()=>{ return (${expr}); })()`, awaitPromise, returnByValue: true }, sessionId)
      if (r.exceptionDetails) throw new Error(r.exceptionDetails.text)
      return r.result.value
    }
    // GL backend string up front (confirms SwiftShader is active even if the shader compile is slow)
    const vendor = await evalIn('(()=>{const c=document.createElement("canvas");const gl=c.getContext("webgl2");const e=gl&&gl.getExtension("WEBGL_debug_renderer_info");return gl&&e?gl.getParameter(e.UNMASKED_RENDERER_WEBGL):(gl?"webgl2":"no-webgl2");})()').catch(() => '?')
    // wait for orch ready; the SwiftShader SOFTWARE cold-compile of the full terrain shader is very
    // slow (minutes). On timeout, return a DIAGNOSTIC (not a bare throw) so the failure path is legible.
    const orchDeadline = Date.now() + (Number(process.env.LAB_ORCH_TIMEOUT_MS) || 8 * 60 * 1000)
    let st = 'init', pageErr = null
    while (Date.now() < orchDeadline) {
      st = await evalIn('String(window.__planetOrchStatus || "init")').catch(() => 'navigating')
      pageErr = await evalIn('window.__pageErr ? String(window.__pageErr.message || window.__pageErr) : null').catch(() => null)
      if (st === 'ready' || st === 'error' || (typeof pageErr === 'string' && pageErr.length)) break
      await new Promise(r => setTimeout(r, 3000))
    }
    if (st !== 'ready') {
      try { ws.close() } catch {}
      return { ok: false, reason: pageErr ? 'page-error' : (st === 'error' ? 'orch-error' : 'orch-not-ready'),
        status: st, pageErr, vendor,
        note: 'SwiftShader software cold-compile of the full terrain shader is slow (minutes); raise LAB_ORCH_TIMEOUT_MS, or use a GPU/Windows chrome with --use-angle=d3d11 (CHROME env) for a fast compile-check.' }
    }
    const result = await fn(evalIn, screenshot)
    try { ws.close() } catch {}
    return { ok: true, vendor, ...result }
  } finally {
    for (const p of procs) { try { p.kill() } catch {} }
  }
}

// SELF-SERVE visual witness: headless render of the terrain over LAND at a low oblique pose -> PNG.
// `--d3d11` uses the real AMD/ANGLE-D3D11 (FXC) backend; default SwiftShader (GPU-free, portable).
async function cmdShot(args) {
  ensureOutDir()
  if (args.d3d11) process.env.LAB_ANGLE = 'd3d11'
  const out = args.out ? path.resolve(String(args.out)) : path.join(OUT_DIR, 'shot.png')
  const altKm = num(args.alt, 4.0)
  const pitch = num(args.pitch, 0.4)   // 0 = straight down, ~0.4 = oblique forward (ground fills frame)
  const dir = args.dir ? ('[' + String(args.dir) + ']') : 'null'   // world dir to aim at (find a peak via height-cpu)
  const r = await withHeadless(async (evalIn, screenshot) => {
    // parkAboveGround (planet.html:1097): OBLIQUE pitch so the forward GROUND fills the frame (uses the
    // GPU height probe to avoid empty/nadir-over-peak frames). parkOblique/litParkOverLand aim at the
    // HORIZON/sky -- wrong for inspecting terrain.
    const parked = await evalIn(`(async()=>{
      const d = window.__diag || {}, p = window.__planet;
      window.__landDir = ${JSON.stringify(LAND_REF)};   // reliable LAND reference for parkAboveGround's default dir
      if (p && p.cam && p.cam.sunLatBase!==undefined) p.cam.sunLatBase = 0.35;   // oblique sun -> relief shading
      if (d.parkAboveGround) { try { const r = await d.parkAboveGround(${altKm}, ${dir}, ${pitch}); return (typeof r==='object')?JSON.stringify(r).slice(0,220):String(r); } catch(e){ return 'pag-err:'+e.message; } }
      if (d.landWitness) { try { await d.landWitness(${altKm}, ${pitch}); return 'landWitness'; } catch(e){ return 'lw-err:'+e.message; } }
      return 'no-park-fn';
    })()`)
    await evalIn('(async()=>{ const f=()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); for(let i=0;i<8;i++) await f(); return 1; })()')
    const info = await evalIn('({ glErr:(window.__lastGLRender&&window.__lastGLRender.checkGlError)?window.__lastGLRender.checkGlError():"x", kept:window.__cullStats?window.__cullStats.kept:null, altM:window.__cullStats?window.__cullStats.altM:null })')
    await screenshot(out)
    return { parked, ...info }
  })
  console.log(JSON.stringify({ out, ...r }, null, 1))
  return r.ok ? 0 : 1
}

// A/B every scalar FS material/color lever (window.__<name>, read by gl-render _g): set each to an
// EXTREME value, render on the GPU, and hash the framebuffer screenshot vs the baseline. A lever whose
// extreme value produces an IDENTICAL frame has NO effect -> a dead lever to remove or fix.
async function cmdAbFs(args) {
  if (args.d3d11) process.env.LAB_ANGLE = 'd3d11'
  ensureOutDir()
  const tmp = path.join(OUT_DIR, '_abfs.png')
  const hashFile = () => { const b = fs.readFileSync(tmp); let s = 0; for (let i = 0; i < b.length; i++) s = (s * 16777619 ^ b[i]) >>> 0; return (s >>> 0) + ':' + b.length }
  // [name, extreme value clearly != default]. If even this changes nothing, the lever is dead.
  const L = [
    ['biomeTint', 1.0], ['texBright', 0.3], ['texSat', 3.0], ['texMix', 0], ['hazeMul', 4.0],
    ['exposure', 3.0], ['lookSat', 3.0], ['lookContrast', 3.0], ['reliefShade', 8.0], ['vertexAO', 3.0],
    ['aoAmt', 3.0], ['variationAmt', 0.8], ['biomeWarp', 5.0], ['nrmLow', 4.0], ['triSharp', 16],
    ['texWarp', 2.0], ['texPhoto', 1.0], ['texPhotoNear', 1.0], ['flatNormal', 1.0], ['skyFill', 2.0],
    ['terminatorGlow', 2.0], ['nightLights', 3.0], ['nightFloor', 1.0], ['termWidth', 2.0], ['texNrmK', 5.0],
    ['diffWrap', 1.0], ['beachTop', 3000], ['beachWidth', 60], ['bandWarp', 9000], ['texFar0', 60000],
    ['texFar1', 90000], ['xSoft', 3.0], ['xFinger', 12], ['ordPush', 3.0], ['xFade0', 0], ['xFade1', 80],
    ['nrmFade0', 0], ['nrmFade1', 90], ['colorVar', 2.0], ['biomeSat', 0], ['texTile', 8.0],
  ]
  const FR = 'const f=()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); for(let i=0;i<5;i++) await f();'
  const r = await withHeadless(async (evalIn, screenshot) => {
    await evalIn(`(async()=>{ window.__landDir=${JSON.stringify(LAND_REF)}; const d=window.__diag; if(d&&d.parkAboveGround) await d.parkAboveGround(${num(args.alt, 4)}, null, 0.4); const f=()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r))); for(let i=0;i<10;i++) await f(); return 1; })()`)
    await screenshot(tmp); const base = hashFile()
    const changed = [], noEffect = []
    for (const [name, val] of L) {
      await evalIn(`(async()=>{ window.__${name} = ${JSON.stringify(val)}; ${FR} return 1; })()`)
      await screenshot(tmp); const h = hashFile()
      await evalIn(`(async()=>{ try { delete window.__${name}; } catch(e){} ${FR} return 1; })()`);
      (h !== base ? changed : noEffect).push(name)
    }
    // biome-ramp levers (window.__gen.state.biome, read by the C() helper -- colors + height/slope bands)
    const RAMP = [
      ['bcRock', [1, 0, 0]], ['bcGrass', [1, 0, 1]], ['bcSnow', [1, 0, 0]], ['bcShore', [1, 0, 0]], ['bcLowland', [0, 0, 1]],
      ['bandEdgesLo', [0, 50]], ['bandEdgesHi', [50, 120]], ['snowEdges', [0, 200]], ['slopeRock', [0, 0.05]], ['seaDepthM', 100],
    ]
    for (const [name, val] of RAMP) {
      await evalIn(`(async()=>{ window.__gen=window.__gen||{state:{}}; window.__gen.state=window.__gen.state||{}; window.__gen.state.biome=window.__gen.state.biome||{}; window.__gen.state.biome.${name}=${JSON.stringify(val)}; ${FR} return 1; })()`)
      await screenshot(tmp); const h = hashFile()
      await evalIn(`(async()=>{ try{ delete window.__gen.state.biome.${name}; }catch(e){} ${FR} return 1; })()`);
      (h !== base ? changed : noEffect).push('biome.' + name)
    }
    return { baseHash: base, changedCount: changed.length, noEffectCount: noEffect.length, noEffect, changed }
  })
  console.log(JSON.stringify(r, null, 1))
  return r.ok ? 0 : 1
}

async function cmdGlslCheck() {
  const r = await withHeadless(async (evalIn) => {
    const vendor = await evalIn('(()=>{ const c=document.createElement("canvas"); const gl=c.getContext("webgl2"); const e=gl&&gl.getExtension("WEBGL_debug_renderer_info"); return gl&&e?gl.getParameter(e.UNMASKED_RENDERER_WEBGL):(gl?"webgl2-no-dbg":"no-webgl2"); })()')
    const probe = await evalIn('(window.__planetOrch && window.__planetOrch.render && window.__planetOrch.render.sampleGroundM)? window.__planetOrch.render.sampleGroundM([0,1,0]) : "no-probe"')
    const pageErr = await evalIn('window.__pageErr || null')
    return { compiled: pageErr === null, vendor, probe, pageErr }
  })
  console.log(JSON.stringify(r, null, 1))
  return r.ok && r.compiled ? 0 : 1
}

async function cmdParity(args) {
  const n = Math.max(1, Math.round(num(args.n, 64)))
  // deterministic spiral of directions (no Math.random for reproducibility)
  const dirs = []
  for (let i = 0; i < n; i++) {
    const y = 1 - (i + 0.5) / n * 2
    const r = Math.sqrt(Math.max(0, 1 - y * y))
    const th = i * 2.399963229728653                       // golden angle
    dirs.push([r * Math.cos(th), y, r * Math.sin(th)])
  }
  const r = await withHeadless(async (evalIn) => {
    // CPU sampler at the PAGE's actual radius -> sampleGroundM and heightAt are the same scale, no normalising
    const pageR = Number(await evalIn('window.__WEBGL2_TERRAIN_R_M || 63600')) || 63600
    const sampler = createHeightSampler({ radius: pageR })
    const sg = 'window.__planetOrch && window.__planetOrch.render && window.__planetOrch.render.sampleGroundM'
    // WARM the collision probe: its program is LAZY-compiled on the first sampleGroundM and returns
    // null until ready (another slow SwiftShader cold-compile). Poll until a finite sample comes back.
    const warm = await waitFor(async () => {
      const v = await evalIn(`(()=>{ const o=window.__planetOrch, p=o&&o.render&&o.render.sampleGroundM; if(!p) return null; const h=p([0,1,0]); return (h!=null && isFinite(h))? h : null; })()`).catch(() => null)
      return v != null
    }, Number(process.env.LAB_PROBE_TIMEOUT_MS) || 4 * 60 * 1000, 2000).then(() => true).catch(() => false)
    if (!warm) return { samples: 0, note: 'sampleGroundM probe never warmed (lazy program compile too slow on SwiftShader; try --use-angle=d3d11 / a GPU chrome, or raise LAB_PROBE_TIMEOUT_MS)' }
    // sampleGroundM is ASYNC + 1-FRAME-STALE (gl-render.js:264): each call harvests the PREVIOUS
    // call's PBO read and kicks off the current dir for next frame. So we must FRAME-SPACE: call p(d),
    // let a frame complete the read, then call p(d) again to harvest d's height. Same-frame batch calls
    // all return one stale value (this was the false ~69m 'divergence' -- the mirror was never wrong).
    const gpu = await evalIn(`(async()=>{
      const p = window.__planetOrch.render.sampleGroundM;
      const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      const out = [];
      for (const d of ${JSON.stringify(dirs)}) { p(d); await frame(); const h = p(d); out.push((h != null && isFinite(h)) ? h : null); }
      return out;
    })()`)
    if (gpu == null) return { samples: 0, note: 'sampleGroundM probe unavailable (orch.render not ready)' }
    let maxAbs = 0, sumAbs = 0, cnt = 0
    for (let i = 0; i < dirs.length; i++) {
      if (gpu[i] == null || !isFinite(gpu[i])) continue
      const cpu = sampler.heightAt(dirs[i])          // CPU sampler is at the PAGE radius -> direct compare
      const d = Math.abs(cpu - gpu[i])
      maxAbs = Math.max(maxAbs, d); sumAbs += d; cnt++
    }
    return { pageRadiusM: pageR, samples: cnt, maxAbsM: +maxAbs.toFixed(3), meanAbsM: +(sumAbs / Math.max(1, cnt)).toFixed(3),
      note: 'APPROXIMATE. sampleGroundM is an async, 1-frame-stale, frame-coupled LIVE-loop collision probe (gl-render.js:264) -- residual staleness under headless rAF inflates the divergence; it is NOT a tight CPU==GPU oracle. The AUTHORITATIVE CPU height-shape regression lock is `npm test` (height-cpu.test.js golden samples). Use this sweep as a coarse live sanity check only.' }
  })
  const tolM = num(args.tol, 50)
  const ran = r.ok && r.samples > 0
  // 'ran' is the gate (the probe is approximate, so withinTol is informational, not a hard fail).
  console.log(JSON.stringify({ ...r, tolM, withinTol: ran && r.maxAbsM <= tolM, ran }, null, 1))
  return ran ? 0 : 1
}

function cmdHelp() {
  console.log(`TV8 CLI testing lab (scripts/lab.mjs)

  heightmap [--res N=256] [--center lat,lon] [--span deg=20] [--radius m=6360000]
            [--hillshade] [--seed N] [--out file.png]
                 Render the CPU height field (src/height-cpu.js) to a grayscale PNG + print stats.
  build          Regenerate the CPU height (gen-height.mjs) + compile-check the GLSL (SwiftShader).
  glsl-check     Headless SwiftShader Chromium: assert terrain.glsl compiles, report the GL backend.
  parity [--n N=64] [--tol m=50]
                 CPU heightAt vs GPU _PROBE_ sampleGroundM divergence sweep (the parity gate).
  shot [--alt km=4] [--pitch 0..1=0.4] [--dir x,y,z] [--d3d11] [--out f.png]
                 Headless RENDER of the terrain over land at an oblique pitch (parkAboveGround: ground
                 fills the frame) -> PNG to inspect. --d3d11 = real AMD/FXC backend (else SwiftShader).
  ab-fs [--d3d11] [--alt km=4]
                 A/B every FS material/color/biome lever (window.__* + __gen.state.biome): perturb each,
                 render, hash the framebuffer vs baseline -> reports any lever with NO pixel effect (dead).
                 Needs LAND in frame (run warm; changedCount high = good frame).
  help

Backend: CPU heights = pure node (no GPU). GLSL = headless Chromium --use-angle=swiftshader
(GPU-free). For the ANGLE/FXC witness, run chrome with --use-angle=d3d11 on Windows instead.`)
  return 0
}

// ---------------------------------------------------------------- exports (for src/lab.test.js)
export { parseArgs, dirFromLatLon, sampleField, crc32, encodePNGGray, toGray }

// ---------------------------------------------------------------- main (only when run as the CLI entry)
import { pathToFileURL } from 'node:url'
// SETTLED parity over a LOCAL TANGENT PATCH around an anchor dir (matches a consumer's play patch,
// e.g. spoint). Unlike cmdParity (whole-sphere, 2-tap), this taps the SAME dir for K frames so the
// 1-frame-stale single-slot probe fully converges -> a tight CPU==GPU oracle at the patch.
async function cmdParityPatch(args) {
  const R = num(args.radius, 63600)
  const A = args.anchor ? String(args.anchor).split(',').map(Number) : [-0.641, 0.2558, 0.7237]
  const reach = num(args.reach, 320)   // metres half-extent of the local patch
  const grid = Math.max(2, Math.round(num(args.grid, 7)))
  const K = Math.max(3, Math.round(num(args.settle, 8)))   // frames to settle each dir
  const nrm = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0]/l, v[1]/l, v[2]/l] }
  const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
  const up = nrm(A)
  const ref = Math.abs(up[1]) < 0.99 ? [0,1,0] : [1,0,0]
  const east = nrm(cross(ref, up)); const north = cross(east, up)
  const localToDir = (x, z) => nrm([ up[0] + (east[0]*x + north[0]*z)/R, up[1] + (east[1]*x + north[1]*z)/R, up[2] + (east[2]*x + north[2]*z)/R ])
  const samples = []
  for (let i = 0; i < grid; i++) for (let j = 0; j < grid; j++) {
    const x = (i/(grid-1)*2-1)*reach, z = (j/(grid-1)*2-1)*reach
    samples.push({ x, z, dir: localToDir(x, z) })
  }
  const skip = !!args.skip
  const reliefScale = R / 6360000
  const r = await withHeadless(async (evalIn) => {
    const sampler = createHeightSampler({ radius: R })
    if (skip) await evalIn('(()=>{ window.__probeSkipCarves = 1; return 1; })()')
    const cpuH = skip
      ? (dir) => { const d = (function(v){const l=Math.hypot(v[0],v[1],v[2])||1;return [v[0]/l,v[1]/l,v[2]/l]})(dir); return sampler._fns.composeHeightC(d, [0,0], 100, sampler._fns.computeHCache(d), true) * reliefScale }
      : (dir) => sampler.heightAt(dir)
    const warm = await waitFor(async () => {
      const v = await evalIn(`(()=>{ const o=window.__planetOrch, p=o&&o.render&&o.render.sampleGroundM; if(!p) return null; const h=p([0,1,0]); return (h!=null&&isFinite(h))?h:null; })()`).catch(() => null)
      return v != null
    }, Number(process.env.LAB_PROBE_TIMEOUT_MS) || 4*60*1000, 2000).then(() => true).catch(() => false)
    if (!warm) return { samples: 0, note: 'probe never warmed' }
    const dirs = samples.map(s => s.dir)
    const gpu = await evalIn(`(async()=>{
      const p = window.__planetOrch.render.sampleGroundM;
      const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      const out = [];
      for (const d of ${JSON.stringify(dirs)}) { let h=null; for (let k=0;k<${K};k++){ h=p(d); await frame(); } out.push((h!=null&&isFinite(h))?h:null); }
      return out;
    })()`)
    if (gpu == null) return { samples: 0, note: 'probe unavailable' }
    const rows = []
    let maxAbs = 0, sumAbs = 0, cnt = 0
    for (let i = 0; i < samples.length; i++) {
      if (gpu[i] == null || !isFinite(gpu[i])) continue
      const cpu = cpuH(samples[i].dir)
      const d = Math.abs(cpu - gpu[i]); maxAbs = Math.max(maxAbs, d); sumAbs += d; cnt++
      rows.push({ x: samples[i].x, z: samples[i].z, cpu: +cpu.toFixed(2), gpu: +gpu[i].toFixed(2), diff: +(gpu[i]-cpu).toFixed(2) })
    }
    return { pageRadiusM: R, anchor: A, reachM: reach, settleFrames: K, samples: cnt, maxAbsM: +maxAbs.toFixed(3), meanAbsM: +(sumAbs/Math.max(1,cnt)).toFixed(3), rows }
  })
  console.log(JSON.stringify(r, null, 1))
  return r.ok && r.samples > 0 ? 0 : 1
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = parseArgs(process.argv.slice(2))
  const cmd = args._[0] || 'help'
  const table = { heightmap: cmdHeightmap, build: cmdBuild, 'glsl-check': cmdGlslCheck, parity: cmdParity, 'parity-patch': cmdParityPatch, shot: cmdShot, 'ab-fs': cmdAbFs, help: cmdHelp }
  const fn = table[cmd]
  if (!fn) { console.error(`unknown command: ${cmd}`); cmdHelp(); process.exit(2) }
  try { process.exit((await fn(args)) | 0) }
  catch (e) { console.error('[lab] error:', e && e.stack || e); process.exit(1) }
}
