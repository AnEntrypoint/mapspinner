// PatchBaker.js -- bakes terrain height patches off the REAL GPU shader (mapspinner) inside the
// singleplayer/host Web Worker (which has OffscreenCanvas WebGL2), feeding them straight into the
// physics collider. Measured ~2ms per 130^2 tile = ~0.0001ms/sample, ~200-400x faster than the CPU
// fractal AND faster than a JS bilinear lookup -- so the worker bakes patches LIVE on demand and stores
// nothing (the patch IS the GPU's composeHeight, exact). Whole planet reachable: any (face,ox,oy,l)
// bakeable. Runtime-safe (worker). Falls back to null (caller uses the CPU sampler) when GL2 is absent
// (dedicated Node server) or init fails.
//
// Tile->world convention (pinned, gap 0.00/texel vs sampleGroundM): texel (ix,iz) over a tile placed at
// face param corner (ox,oy) spanning l face-metres -> p=(ox+ix/(res-1)*l, oy+iz/(res-1)*l);
// faceLocal=faceWarp(p)=R*tan((p/R)*PI/4); dir=normalize(U*faceLocal_x + V*faceLocal_y + center*R)
// where (U,V,center)=FACE_FRAME[face]; tile.heights[iz*res+ix]=composeHeight(dir) exactly.

const FACE_FRAME = [
  { c: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] }, { c: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
  { c: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] }, { c: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
  { c: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] }, { c: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] },
]
const _dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

// pick the cube face a unit dir falls on + its pre-warp face-UV (ox,oy), matching the renderer's
// quadtree coord (ox = (4/pi)R atan(dot(dir,U)/dot(dir,center))).
export function dirToFace(dir, R) {
  let bf = 0, bd = -Infinity
  for (let i = 0; i < 6; i++) { const d = _dot(dir, FACE_FRAME[i].c); if (d > bd) { bd = d; bf = i } }
  const F = FACE_FRAME[bf]
  const cc = _dot(dir, F.c), cu = _dot(dir, F.u), cv = _dot(dir, F.v)
  const k = (4 / Math.PI) * R
  return { face: bf, ox: k * Math.atan(cu / cc), oy: k * Math.atan(cv / cc) }
}

// Create the patch baker. opts: { radius, reliefScale, seed }. Returns { bakeTile, dirToFace, res } or
// null if no GL2. bakeTile(face,ox,oy,l) -> Float32Array(res*res) of absolute composeHeight, or null.
export async function createPatchBaker(opts = {}) {
  const warn = (m) => { try { console.warn('[PatchBaker] unavailable: ' + m) } catch (_) {} }
  if (typeof OffscreenCanvas === 'undefined') { warn('no OffscreenCanvas (dedicated Node?)'); return null }
  let gl
  try {
    const oc = new OffscreenCanvas(8, 8)
    gl = oc.getContext('webgl2', { antialias: false, depth: false })
    if (!gl) { warn('no webgl2 context'); return null }
    if (!gl.getExtension('EXT_color_buffer_float') || !gl.getExtension('OES_texture_float_linear')) { warn('no float-buffer extensions'); return null }
  } catch (e) { warn('gl init threw: ' + (e && e.message)); return null }
  const _isNode = typeof process !== 'undefined' && process.versions?.node
  let initMapspinnerPlanet
  try { ({ initMapspinnerPlanet } = await import('./planet-orchestrator.js')) }
  catch (e) { warn('orchestrator import threw: ' + (e && e.message)); return null }
  let planet
  try { planet = await initMapspinnerPlanet(gl, { radius: opts.radius, gridMeshSize: 11, reliefScale: opts.reliefScale, hpfSeed: opts.seed }) }
  catch (e) { warn('initMapspinnerPlanet threw: ' + (e && e.message)); return null }
  // bakeTileReadback is exposed on self (window in worker) after init; ensure the bake program is built.
  const g = (typeof self !== 'undefined') ? self : (typeof window !== 'undefined' ? window : globalThis)
  if (typeof g.__thcEnsureBake === 'function') g.__thcEnsureBake()
  // the bake program compiles lazily on the first bakeTileReadback; poll briefly for it to appear.
  for (let k = 0; k < 20 && typeof g.__thcBakeReadback !== 'function'; k++) await new Promise(r => setTimeout(r, 100))
  if (typeof g.__thcBakeReadback !== 'function') { warn('__thcBakeReadback never appeared after init'); return null }
  // give the lazy bake-program build a moment (it compiles the _HEIGHTBAKE_ shader on first call).
  await new Promise(r => setTimeout(r, 200))
  let res = 130
  function bakeTile(face, ox, oy, l, level = 0) {
    for (let k = 0; k < 12; k++) {
      const r = g.__thcBakeReadback(face | 0, ox, oy, l, level)
      if (r && r.heights) { res = r.res; return r.heights }
    }
    return null
  }
  return { bakeTile, dirToFace: (dir) => dirToFace(dir, opts.radius), res, planet }
}

// Build a groundHeightLocal-compatible O(1) patch lookup over a baker, matching the FINEST display LOD
// density (no reduction). Shared by the server collider (TerrainPhysics) AND the client placement frame
// (veg/grass/rock) so BOTH derive height from the identical GPU bake -> byte-identical placement parity,
// and the per-candidate fractal (~0.4ms) becomes a per-chunk-amortized patch lookup. frame supplies
// radius/anchorHeight/localToDir; tcfg supplies maxLevel/offsetY. Returns null if no baker.
// fallbackFn(x,z) is used when a patch bake transiently fails (keeps determinism on a miss).
export function createPatchHeightFn({ baker, frame, maxLevel = 11, offsetY = 0, fallbackFn }) {
  if (!baker) return null
  const R = frame.radius, aH = frame.anchorHeight, res = baker.res, gridMeshSize = 11
  const finestLeaf = 2 * R / Math.pow(2, maxLevel)
  const visualSpacing = finestLeaf / (gridMeshSize - 1)
  const patchSpan = Math.max(8, visualSpacing * (res - 1))
  const cache = new Map(); const MAX = 96
  function patchFor(face, ox, oy) {
    const pi = Math.floor(ox / patchSpan), pj = Math.floor(oy / patchSpan)
    const key = face + ':' + pi + ':' + pj
    let p = cache.get(key)
    if (!p) {
      const heights = baker.bakeTile(face, pi * patchSpan, pj * patchSpan, patchSpan, 0)
      if (!heights) return null
      p = { heights, ox: pi * patchSpan, oy: pj * patchSpan }
      cache.set(key, p); if (cache.size > MAX) cache.delete(cache.keys().next().value)
    }
    return p
  }
  function heightFn(x, z) {
    const d = frame.localToDir(x, z)
    const { face, ox, oy } = baker.dirToFace(d)
    const p = patchFor(face, ox, oy)
    if (!p) return fallbackFn ? fallbackFn(x, z) : frame.groundHeightLocal(x, z)
    const fx = (ox - p.ox) / patchSpan * (res - 1), fy = (oy - p.oy) / patchSpan * (res - 1)
    const ix = Math.max(0, Math.min(res - 2, Math.floor(fx))), iz = Math.max(0, Math.min(res - 2, Math.floor(fy)))
    const tx = fx - ix, tz = fy - iz, h = p.heights
    const h00 = h[iz * res + ix], h10 = h[iz * res + ix + 1], h01 = h[(iz + 1) * res + ix], h11 = h[(iz + 1) * res + ix + 1]
    const abs = (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz
    const r2 = x * x + z * z, s = r2 / (R * R), sq = Math.sqrt(1 + s), drop = r2 / R / ((sq + 1) * sq)
    return (abs - aH) - drop + offsetY
  }
  return { heightFn, patchSpan, res, spacing: visualSpacing, maxLevel }
}
