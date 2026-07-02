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
  // bakeTile: SYNCHRONOUS, blocking (bounded retry). Use for physics-collider-prep callers that need a
  // resolved height NOW and cannot tolerate a stale/fallback value -- e.g. the server/host authoritative
  // collider build, where a wrong or missing patch would desync physics for every connected player.
  function bakeTile(face, ox, oy, l, level = 0) {
    for (let k = 0; k < 12; k++) {
      const r = g.__thcBakeReadback(face | 0, ox, oy, l, level)
      if (r && r.heights) { res = r.res; return r.heights }
    }
    return null
  }
  // bakeTileAsync: TRULY NON-BLOCKING (2026-07-02, evidence-driven -- a live stack-trace CDP profile
  // showed getBufferSubData, the harvest inside __thcBakeReadback, as the #1 measured cost even on THIS
  // caller, because the prior version still called the same synchronous readback -- only the outer retry
  // loop was removed, the inner bounded-spin-then-block harvest was not). Now issues via
  // __thcBakeIssueAsync (draw + PBO read + fence, returns immediately, ZERO wait) and polls via
  // __thcBakePollAsync (non-blocking clientWaitSync timeout 0) on each call -- first call for a tile
  // issues the bake and returns null (caller's fallbackFn covers this frame); a LATER call for the same
  // tile (once the fence has signaled, no CPU stall to check) harvests and returns the real heights.
  // Falls back to the old one-shot synchronous __thcBakeReadback if the async pair isn't exposed (an
  // older gl-render.js build without them) so this stays forward-compatible.
  let _asyncTileKey = null
  const _asyncDone = new Map()   // key -> heights, for a completed bake that belongs to a DIFFERENT call than the one that harvested it
  const _ASYNC_DONE_MAX = 8      // small: bridges the one-call-behind race, not a real cache (createPatchHeightFn owns the real LRU)
  function bakeTileAsync(face, ox, oy, l, level = 0) {
    if (typeof g.__thcBakeIssueAsync !== 'function' || typeof g.__thcBakePollAsync !== 'function') {
      const r = g.__thcBakeReadback(face | 0, ox, oy, l, level)
      if (r && r.heights) { res = r.res; return r.heights }
      return null
    }
    const key = face + ':' + ox + ':' + oy + ':' + l + ':' + level
    if (_asyncDone.has(key)) { const h = _asyncDone.get(key); _asyncDone.delete(key); return h }   // a prior call already harvested this exact tile
    // Harvest first: a completed bake from a prior issue() becomes available here, non-blocking.
    const done = g.__thcBakePollAsync()
    if (done) {
      res = done.res
      const doneKey = done.face + ':' + done.ox + ':' + done.oy + ':' + done.l + ':' + done.level
      if (doneKey === key) return done.heights   // this call's own tile finished -- return it now
      // A DIFFERENT tile finished than the one THIS call wants (the caller moved on since issuing it).
      // Stash it rather than discard the completed GPU work -- the tile that requested it will very
      // likely be re-queried again soon (patch-span-sized cells, revisited on the next lookup in that
      // area) and will hit the stash above instead of re-issuing a redundant bake.
      _asyncDone.set(doneKey, done.heights)
      if (_asyncDone.size > _ASYNC_DONE_MAX) _asyncDone.delete(_asyncDone.keys().next().value)
    }
    if (_asyncTileKey !== key) { g.__thcBakeIssueAsync(face | 0, ox, oy, l, level); _asyncTileKey = key }
    return null
  }
  return { bakeTile, bakeTileAsync, dirToFace: (dir) => dirToFace(dir, opts.radius), res, planet }
}

// Build a groundHeightLocal-compatible O(1) patch lookup over a baker, matching the FINEST display LOD
// density (no reduction). Shared by the server collider (TerrainPhysics) AND the client placement frame
// (veg/grass/rock) so BOTH derive height from the identical GPU bake -> byte-identical placement parity,
// and the per-candidate fractal (~0.4ms) becomes a per-chunk-amortized patch lookup. frame supplies
// radius/anchorHeight/localToDir; tcfg supplies maxLevel/offsetY. Returns null if no baker.
// fallbackFn(x,z) is used when a patch bake transiently fails (keeps determinism on a miss).
//
// blocking (default true): the server/host authoritative collider MUST resolve to the real GPU-baked
// height (never a stale/fallback value) since every connected player's physics derives from it -- keep
// the bounded-retry bakeTile() there. A CLIENT-side placement/render caller should pass blocking:false:
// veg/grass/rock placement is consumed at a distance/rate where a value resolved up to ~100 frames late
// is indistinguishable, so bakeTileAsync's one-shot non-retried attempt (cache hit -> instant, cache miss
// -> fallbackFn now, GPU bake completes in the background and the NEXT lookup at that cell hits cache)
// removes the retry-loop's busy-spin GL cost from the hot placement path entirely.
export function createPatchHeightFn({ baker, frame, maxLevel = 11, offsetY = 0, fallbackFn, blocking = true }) {
  if (!baker) return null
  const R = frame.radius, aH = frame.anchorHeight, res = baker.res, gridMeshSize = 11
  const finestLeaf = 2 * R / Math.pow(2, maxLevel)
  const visualSpacing = finestLeaf / (gridMeshSize - 1)
  const patchSpan = Math.max(8, visualSpacing * (res - 1))
  // MAX 384 (up from 96, ~2026-07-02 fps-drop investigation): a synchronous readPixels-bound cache MISS
  // is the dominant frame cost (profiled ~3.3ms/frame, 48% of the 144Hz budget) -- during sustained
  // movement (not a static profile snapshot) the player continuously enters new patch cells, evicting
  // recently-baked neighbors from a too-small LRU before they're revisited, forcing needless re-bakes.
  // 384 entries * ~67KB/patch (130^2 float32) = ~25MB, trivial against a modern GPU/heap budget, and
  // covers a much larger contiguous streamed area before any eviction -- correctness unchanged (still an
  // exact re-bake of the SAME deterministic GPU shader on a miss, just fewer misses).
  const cache = new Map(); const MAX = 384
  const _bakeFn = blocking ? baker.bakeTile : (baker.bakeTileAsync || baker.bakeTile)   // non-blocking callers fall back to bakeTile if an older baker lacks the async variant
  function patchFor(face, ox, oy) {
    const pi = Math.floor(ox / patchSpan), pj = Math.floor(oy / patchSpan)
    const key = face + ':' + pi + ':' + pj
    let p = cache.get(key)
    if (!p) {
      const heights = _bakeFn(face, pi * patchSpan, pj * patchSpan, patchSpan, 0)
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
  // PREFETCH (2026-07-02, non-blocking-fallback follow-up): heightFn's non-blocking bakeTileAsync path
  // always misses (falls back to the CPU fractal) the FIRST time a patch cell is entered, since the bake
  // is only ISSUED on that call and harvested on a later one. During sustained movement the player keeps
  // entering never-before-seen cells, so misses recur continuously rather than being a one-time cost.
  // prefetchAround(x,z) issues bakes for the 3x3 patch neighborhood around a position (skipping the
  // center cell, which heightFn's own call already covers) BEFORE the player's next placement/render
  // lookup needs them -- call once per frame from the moving entity's (or camera's) position. Each issue
  // is a single bakeTileAsync call (immediate return, no wait); a cell already cached is skipped, and
  // re-issuing an already-in-flight cell just re-marks the same in-flight key (cheap, not wrong -- see
  // bakeTileAsync's single _asyncTileKey slot). blocking:true bakers (server/host collider) never need
  // this -- prefetch only pays off the async miss cost, which only exists on the non-blocking path.
  function prefetchAround(x, z) {
    const d = frame.localToDir(x, z)
    const { face, ox, oy } = baker.dirToFace(d)
    const pi0 = Math.floor(ox / patchSpan), pj0 = Math.floor(oy / patchSpan)
    for (let dj = -1; dj <= 1; dj++) {
      for (let di = -1; di <= 1; di++) {
        if (di === 0 && dj === 0) continue
        const pi = pi0 + di, pj = pj0 + dj
        const key = face + ':' + pi + ':' + pj
        if (cache.has(key)) continue
        if (typeof baker.bakeTileAsync === 'function') baker.bakeTileAsync(face, pi * patchSpan, pj * patchSpan, patchSpan, 0)
      }
    }
  }
  return { heightFn, prefetchAround, patchSpan, res, spacing: visualSpacing, maxLevel }
}
