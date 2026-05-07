# Streaming Vegetation and Rocks Without FPS or Physics Gaps

A field-tested playbook drained from the **spawnpoint** project. mapspinner ships its own terrain system, so this document covers only the vegetation and rock streaming layers that sit on top of any heightfield-style terrain.

This is not a tutorial. It is a list of decisions, with the reasoning for each, ordered roughly from architecture down to micro-pitfalls.

---

## 1. Streaming architecture

Goal: never let world generation share a frame budget with the simulation tick or the renderer.

### Worker pool topology
- **One worker pool per subsystem**, never a global pool. Vegetation placement, vegetation physics bake, and rock convex-hull bake each get their own pool.
- **Cap workers at `min(8, hardwareConcurrency / 2)`** *per pool*, with the SUM across all client pools strictly under `hardwareConcurrency`. Oversubscription is the single biggest source of "stuttering at startup" on 4-core laptops.
- **Shortest-queue dispatch**, not round-robin. Track `_wPending[i]` per worker; pick `argmin`. Round-robin guarantees a stalled worker holds up half the queue.

### Decouple from sim/render tick
- Physics streaming (vegetation body spawn, rock collider creation) runs on **dedicated workers** with results applied on the sim thread as cheap writes.
- Result drain runs off a `setInterval(12ms)` cadence, **not** the physics tick. With Jolt's `mBlockSize=2`, this keeps streaming bursts from tripping the dilation governor (see §5).
- Validated for 39m of continuous sprint over 10s of streaming, zero free-fall events.

### Cheap writes on main, expensive compute off-main
- The main thread is allowed to: assign a `BufferGeometry`, set a `mesh.position`, push an `InstancedMesh` matrix.
- The main thread is **never** allowed to: run placement noise, convex hull, or billboard bake. All of these go to workers and return `Transferable` ArrayBuffers (zero-copy).

---

## 2. LOD strategy

### Vegetation LOD bands as a bitmask
Vegetation has three orthogonal LOD decisions per chunk: `cardNear | trunkNear | foliageNear`. Encode as a bitmask, rebuild the chunk only when bits change. Otherwise the same chunk reassembles every frame the camera moves a meter.

### Atomic LOD swap
When a chunk transitions LOD, **keep the old chunk in-scene** until the new one is fully assembled, then dispose. Naïve "remove old → request new → add new" produces 1–3 empty-frame artifacts at every band crossing. Validated zero empty-chunk frames.

### Adaptive renderDistance, never adaptive bands
When FPS dips, scale `renderDistance` down. **Do not** rebalance LOD band radii at runtime — that triggers cascading rebuilds that cost more than they save. renderDistance must exceed `foliageLodChunks` or you get a visible far-band hole.

### Recompute LOD band at assemble time
A request might be queued at LOD-2 and drained 800ms later when the camera moved into LOD-1 range. Recompute lodBand at *assemble* time, not request time, or you'll keep building stale levels.

---

## 3. Per-chunk work budgeting

### Strict 1-chunk-per-RAF GPU drain
GPU upload of newly-assembled InstancedMesh chunks **must** be capped at one chunk per `requestAnimationFrame`, not time-budgeted. Three.js has no pre-upload API; `gl.bufferData` happens on the first render of a new geometry, and N chunks uploaded in one frame stalls 50–200ms.

### Time-slice drain for compute work
Worker-result drain (CPU-side assembly, no GPU upload) uses a **4ms time-slice** budget per frame. Balances startup speed against frame-pacing jitter.

### Closest-first eviction queue
Sort work by camera distance ascending; never time-out evictions on a clock — only on "wanted" status changes.

### `_pendingEvict` overlap guard
During a chunk's LOD transition, both the old and new representation can briefly render the same area. A `_pendingEvict` flag on the outgoing chunk prevents it from re-emerging once the swap commits, eliminating overlap artifacts.

---

## 4. Caching and stale-result handling

### Placement cache by cell hash
Vegetation placement is deterministic per `(chunkX, chunkZ)`. Cache placements in `_placementCache` Map, keyed by cell hash. LOD band crossings stop spamming `workerPool.request()`.

### Scratch-buffer hoisting in hot loops
Per-vertex / per-instance hot paths called >10K times per chunk allocate `Array`/`Float64Array` per call by default. Hoist scratch buffers (e.g. `_ws`, `_kindAccum`, `_paletteLow`, `_paletteHigh`) to closure scope and reuse across the chunk. Inline tight inner functions to drop allocation+destructure overhead. **General pattern: any per-vertex / per-instance hot path with >10K invocations gets hoisted TypedArrays.**

### Stale-result drop
Worker pool tracks `pending: Map<key, requestedLOD>`. Worker echoes back `_reqLOD`. Drain compares: if `_reqLOD !== pending.get(key)`, the result is stale (the request was upgraded mid-flight) — drop it. Otherwise you double-build and the second build wins, with the first still occupying a frame's GPU upload.

---

## 5. Physics streaming for vegetation and rocks

### Radius cap independent of visual
Physics streaming for vegetation and rocks runs over a **smaller radius than visual** (in spawnpoint, `PHYSICS_RADIUS_CHUNKS = 2` regardless of visual `renderDistance = 12`). Far-LOD trees and rocks don't need colliders.

### Sync inline sampler on spawn
Player spawn under a streaming chunk hits free-fall if vegetation/rock colliders aren't there yet. Fix: **synchronous `_inlineState` sampler** runs on spawn that produces a ground value from the same noise function as the worker. Player lands on inline-sampled ground; real colliders replace within ~50ms. The same pattern applies to anything (vegetation, rocks) the player can land on.

### Tick dilation as a metric, not a feature
Jolt tick dilation (load > 85% → factor decrements 0.05) is a backstop, not part of the design. If streaming bursts trip it, players see "animation plays, position frozen" for seconds. The decouple pattern (§1) keeps streaming off the sim tick precisely so dilation never fires during normal play.

### Drain decoupled at 12ms
Vegetation and rock physics drain on `setInterval(12ms)`, *not* per physics tick. With `mBlockSize = 2`, this lets streaming spread across multiple sim steps without trip.

### Convex hull baking must be sync
Async rock convex-hull bake races worker results: rocks spawn before colliders exist, players walk through them. Bake **inline-synchronously** on the server (it's fast enough — pure-JS hull computation, `ROCK_COLLIDER_RADIUS=32m` gating).

### Player divisor dt accumulation
Performance optimization `PHYSICS_PLAYER_DIVISOR=3` (process player movement every 3rd tick) was passing **single-tick dt** instead of accumulated dt → 3× speed reduction. Fix: `accumDt` Map, accumulate skipped ticks, pass on processed tick.

---

## 6. Rocks

### Integrated into the vegetation pipeline
Rocks ride the same placement/system/physics path as vegetation, **not** a parallel rock subsystem. Same workers, same LOD bands, same eviction queue. Avoids two streamers fighting for the same frame budget.

### Second-pass grid emit
Rocks emit on an **independent second pass** over the height grid after vegetation. This keeps rock density decoupled from tree density and lets each system own its own poisson/jitter rules.

### Convex hull colliders
Pure-JS hull builder; the server bakes hulls inline (see §5). Keep `ROCK_COLLIDER_RADIUS` modest (32m in spawnpoint) — most rocks past that distance never collide with the player.

### Normal alignment and size variation
Use the local ground normal as an axis-angle quaternion to align each rock to the slope it sits on. Per-LOD `ROCK_SIZE_MULT` (e.g. 2.0 / 4.5 / 8.0) gives near→far size variation without re-baking hulls.

---

## 7. Vegetation specifics

### Collapse variant buckets
Original 6 variants per species (3 mature + 3 juvenile by scale<0.5) → 1 variant per species (`VARIANTS_PER_SPECIES = 1`). InstancedMesh count drops 778 → 164. Per-instance scale alone gives size variation.

### Cylindrical billboard impostors
See §7a for the full impostor playbook.

### Frustum cull via bounding sphere
Wrap each chunk's instanced meshes in a `Group` at chunk centre with a `computeBoundingSphere()`. Three.js's per-instance frustum is wrong for InstancedMesh anyway; the group sphere is correct and free.

### Deterministic placement across LOD bands
`cellHash(cx, cz, speciesIndex, slotIndex)` produces the same xyz across all LOD bands. The same tree appears in the same world position whether rendered as full mesh, trunk-only, or billboard. Critical for atomic swap (§2) to look like a swap and not a re-roll.

### Don't compress Y to fake bare trees
Bare-tree variants with `p.branch=0` originally collapsed trunk Y scale to 0 — pancakes on the ground. Use variant geometry or X/Z scale, never Y=0.

### MapSpinner branch polycount
Override `sections`/`segments`/`children` on the branch material — 73% triangle reduction at no perceptual cost on mid-range hardware.

### Material name-based leaf detection
MapSpinner exposes leaves via material name `"leaf"` or `"foliage"`. Detect by `material.name`, split foliage geometry for billboard bake.

### Carry kind/biome through field re-mapping
When re-mapping placements through a terrain field, `kind` and `biome` MUST carry through. Drop them and rocks turn into trees, or snow biomes turn into grass on the next LOD rebuild.

### Capsule colliders need scalar scale
Jolt `CapsuleShape` requires a scalar `p.scale`. Non-uniform scale silently produces wrong colliders.

### Bake billboard with temp parent
During billboard bake, temp-wrap the tree in a parent that applies `baseScale`. Otherwise the bake captures unscaled geometry and the impostor renders at the wrong size.

---

## 7a. Billboard impostors — full playbook

The single biggest LOD win for mapspinner forests. Each species bakes once at startup into a 256×512 RGBA texture; far-LOD chunks render as one InstancedMesh of camera-facing quads. ~3k–8k tris per live tree collapse to 2 tris. Reference impl: `client/VegetationSystem.js` (spawnpoint), one file, ~250 lines including the bake.

### Why cylindrical (Y-only) and not spherical
Spherical (full lookAt) impostors look correct from any angle but break the moment the camera tilts or the player jumps: the tree visibly rolls about its own base. Cylindrical (rotate around world Y only) keeps the trunk vertical for free, costs one cross product in the vertex shader, and matches how players actually look at trees in a survival/exploration camera (mostly horizontal, occasional pitch). Vertical silhouette stays fixed because the `up` axis is hardcoded `(0,1,0)`, never derived from the camera. Tradeoff: looking down on a forest from a peak, billboards read as flat — acceptable because that's also when atmospheric haze hides them.

### The shader (camera-face around Y)
Full vertex/fragment in `client/VegetationSystem.js:165–198`. Core derivation:

```glsl
vec3 instPos     = vec3(instanceMatrix[3][0], instanceMatrix[3][1], instanceMatrix[3][2]);
float instScaleX = length(instanceMatrix[0].xyz);
float instScaleY = length(instanceMatrix[1].xyz);
vec3 worldInst   = (modelMatrix * vec4(instPos, 1.0)).xyz;
vec3 camToInst   = worldInst - cameraPosition;
vec3 right       = normalize(vec3(-camToInst.z, 0.0, camToInst.x)); // perpendicular in XZ plane
vec3 up          = vec3(0.0, 1.0, 0.0);                              // hardcoded — vertical is invariant
vec3 worldOffset = right * (position.x * billboardSize.x * instScaleX)
                 + up    * (position.y * billboardSize.y * instScaleY);
gl_Position = projectionMatrix * viewMatrix * vec4(worldInst + worldOffset, 1.0);
```

Quad geometry is `PlaneGeometry(1,1).translate(0, 0.5, 0)` so `position.y ∈ [0,1]` — the impostor stands on its base, not centered at origin. Per-instance non-uniform scale survives because `instScaleX/Y` are extracted from the instance matrix columns, so size variation written by the placement worker comes through unmodified. `viewMatrix` only — no `modelViewMatrix` — because the world-space right axis is computed against `cameraPosition`, not against the model's local frame. Fragment shader is a hard `discard` at `alpha < 0.4`; no transparency sort needed (`transparent: false, depthWrite: true`).

### Bake pipeline (per species, once at startup)
`bakeBillboard()` at `client/VegetationSystem.js:122–159`. Steps:

1. Wrap the live tree group in a temp parent: `wrapper = new Group(); wrapper.add(treeGroup); wrapper.scale.setScalar(baseScale); tmp.add(wrapper)`. The wrapper applies the species' `baseScale` so the bake captures the same physical size that runtime instances render at — without it, impostors come out too small or too large by `baseScale` and you cannot fix it post-hoc without re-baking.
2. Bright omnidirectional light rig in the temp scene: `HemisphereLight(white, gray, 1.2)` + `AmbientLight(white, 0.6)` + 4 directional lights at corner directions `[1,1,0.5], [-1,1,0.5], [0.5,1,-1], [-0.5,1,-1]`. Reason: the live forest uses sun + sky lighting that swings with time-of-day, but a single baked billboard must look acceptable under any sun angle. Flat-ish lighting is the only neutral answer.
3. Frame the bounding box with an orthographic camera: `w = max(size.x, size.z), h = size.y`, ortho `[-w/2, w/2, h/2, -h/2]`, positioned `center.z + max(w,h)*2` away. Width-based on the larger of X/Z so non-square crowns don't clip.
4. Render to `WebGLRenderTarget(256, 512, RGBAFormat, mipmap=Linear)`. The 1:2 aspect matches typical tree silhouettes. **`rt.texture.colorSpace = NoColorSpace`**, render with `outputColorSpace = SRGBColorSpace` — avoids a double-sRGB-encode that otherwise renders the impostor near-black. This is the single most subtle bake bug.
5. After render, `wrapper.remove(treeGroup)` so the caller still owns the live geometry. The render target's texture (`rt.texture`) becomes the impostor map; geometry is a shared `PlaneGeometry(1,1)` plus the per-species `ShaderMaterial(map, vec2(w,h))`.

### Async bake correctness — texture poll
The trap: `Tree.generate()` returns synchronously, but mapspinner's leaf and bark `textureLoader.load()` returns Texture objects whose `.image` populates **asynchronously** from the network. Baking immediately captures fully transparent leaf material — the impostor renders as a bare trunk on a transparent background, which is worse than no billboard.

Fix at `client/VegetationSystem.js:319–356`:
- After `bakeTreeVariant()` for every species, push entries into `_pendingBillboard[]` instead of baking inline.
- Walk every material on every preview group, collect every `map`/`alphaMap`/`normalMap` whose `.image` is missing or not yet `.complete`.
- `_waitForTextures(textures, 4000)` resolves when every texture either fires its `image.load`/`error` event or, for textures whose `.image` is still null, when a `setInterval(100ms)` poll observes `t.image?.complete || t.image?.width > 0`. 4-second timeout finishes anyway so a single missing asset never blocks startup forever.
- One extra `await new Promise(r => requestAnimationFrame(r))` after textures resolve, to guarantee the GPU upload of the freshly-loaded image has flushed before the bake reads it.
- Then bake all species in one pass, log `[Vegetation] billboards baked: N/M`.

The bake runs in an IIFE'd async block kicked off during `createVegetationSystem()`; first chunk requests fire **immediately** (don't await the bake to gate streaming). Chunks at billboard distance simply skip the billboard branch (`!sp.billboard` → `continue`) until the bake lands. Once `entry.billboard` is set, the next LOD-band rebuild for those chunks will pick it up — that's why you also need cache-by-cell-hash placements (§4) so the rebuild costs nothing on the worker side.

### LOD band integration — bitmask + atomic swap
`_lodBandOf(d2)` at `client/VegetationSystem.js:415–423` packs four orthogonal decisions into one int:

```
1  cardNear      d2 ≤ cardLodChunks²
2  trunkNear     d2 ≤ trunkLodChunks²
4  foliageNear   d2 ≤ foliageLodChunks²
8  billboardBand d2 ≤ billboardLodChunks²
```

Assembly at `client/VegetationSystem.js:512–533` reads the mask: a tree species with `!trunkNear && !foliageNear && billboardBand && sp.billboard` takes the billboard branch and emits one InstancedMesh of quads. Otherwise it falls through to the real geometry path. `_lodBand` is recorded on the chunk record; `update()` at `client/VegetationSystem.js:602–625` only enqueues a rebuild when the band changes — so a chunk holding 5000 billboard instances costs zero per-frame work until the camera crosses a band boundary. The atomic-swap rule from §2 applies here too: keep the old chunk in-scene until the new (billboard or full-mesh) chunk is fully assembled, then dispose.

### Gating: renderDistance must exceed foliageLodChunks
`billboardLodChunks` defaults to `Math.max(foliageLodChunks + 2, renderDistance)` (`client/VegetationSystem.js:410`). If `renderDistance ≤ foliageLodChunks`, every chunk inside `renderDistance` is also inside the foliage band, so the `!trunkNear && !foliageNear && billboardBand` predicate is **never true** and billboards never render — you just get a hard pop to nothing at the edge of the foliage band. Always tune so `renderDistance > foliageLodChunks` by at least 1, ideally ≥ 2. The adaptive-LOD controller scales `renderDistance` only (not band radii) precisely so this invariant holds at every adapted setting.

### Y-scale-zero pancake (the bare-tree bug)
Symptom: bare-tree variants (configured with `p.branch = 0` to disable subbranches for visual variety) rendered as flat green smears on the ground at every LOD. Cause: the trunk Y scale was being multiplied by `p.branch`, which is 0 for bare variants → `scale.y = 0` → vertical degeneracy → trunk geometry pancaked at ground plane and only foliage card-billboards remained, draped over the squashed trunk. Fix: drop `p.branch` from the trunk Y scale entirely (`client/VegetationSystem.js:559`); use variant geometry or X/Z scale to convey "bare" instead. **General rule: never let any per-instance weight feed `scale.y = 0`.** This bug ate two days because the foliage card overlay made it look like a rendering glitch, not a scale issue.

### Material handling — leaf/foliage detection by name
mapspinner emits exactly two meshes per tree: `branchesMesh` and `leavesMesh`. `bakeTreeVariant()` at `client/VegetationSystem.js:88–117` splits them by `material.name.toLowerCase().includes('leaf'|'leav'|'foliage')` or any non-zero `alphaTest`. Trunk → opaque MeshStandardMaterial; foliage → `transparent: true, alphaTest: 0.5, side: DoubleSide, depthWrite: true`. The split survives into the billboard bake because both meshes go into the temp wrapper together; the bake captures their composite. At runtime, the foliage-only LOD band can render foliage without trunk (separate InstancedMesh per part), so never emit the foliage as a billboard fallback when trunk is also visible — the per-part split is mutually exclusive with the per-tree billboard branch.

### Variant-bucket collapse interplay
With `VARIANTS_PER_SPECIES = 1` (down from 6), every species has exactly one billboard texture and one InstancedMesh per chunk per band. Total InstancedMesh count drops 778 → 164. If you keep N variants per species, the billboard branch needs an N-way variant-index dispatch (`(seed >>> 16) % variantCount`), N billboard textures per species, and N InstancedMeshes per chunk in the billboard band. Almost always not worth it: per-instance scale plus the cylindrical rotation already give enough silhouette variety at billboard distance that 6 textures look identical to 1.

### Frustum culling — bounding sphere on the chunk Group
Three.js's per-instance frustum cull on InstancedMesh is wrong: it tests the original geometry bounding sphere, not the instance positions. For billboards the mesh-local bounds are `[-0.5,0,-0.5]..[0.5,1,0.5]`, way smaller than the cloud of instance positions — so individual billboards get culled when they shouldn't. Fix: wrap each chunk's meshes in a `Group` at chunk centre, call `computeBoundingSphere()` on the group bounds (chunk size). One sphere test culls the whole chunk; in-chunk you accept the small overdraw because billboards are 2 tris each. This is also why per-mesh `frustumCulled = true` stays on (the group cull does the real work; per-mesh is a cheap secondary).

### InstancedBufferGeometry shared-geo trap
You **cannot** share one `InstancedBufferGeometry` across multiple `InstancedMesh` instances even though the underlying attributes are read-only. Three.js mutates the wrapper (sets `instanceCount`, modifies `_maxInstanceCount`); two meshes sharing one wrapper corrupt each other's draw range. Pattern at `client/VegetationSystem.js:461–466` (rocks, but identical for billboards if you ever extend): create a fresh `InstancedBufferGeometry()` per mesh, assign `geo.index = sharedGeo.index`, copy attributes by reference, copy `boundingSphere`/`boundingBox`. The shared parts are the BufferAttributes themselves; the wrapper is per-mesh. For billboards specifically, the shared `PlaneGeometry(1,1).translate(0,0.5,0)` plus a fresh `InstancedMesh` per chunk per species is enough — no `InstancedBufferGeometry` wrapper needed unless you add per-instance attributes beyond the matrix.

### Tuning knobs and pitfalls

| Knob | spawnpoint value | Notes |
|------|------------------|-------|
| Bake resolution | 256 × 512 | 1:2 matches typical tree aspect; 128×256 still readable |
| `alphaTest` (frag) | 0.4 | Higher → cleaner edges, more leaf shrinkage |
| `cardLodChunks` | 1 | Real-leaf cards (close) |
| `trunkLodChunks` | 2 | Trunk geometry without foliage |
| `foliageLodChunks` | 3 | Full mesh including foliage |
| `billboardLodChunks` | `max(foliageLodChunks+2, renderDistance)` | Never less than this |
| Bake light rig | 4 dirs + hemi + ambient | Flat-ish; matches under any sun angle |
| Texture wait timeout | 4000ms | Bake proceeds anyway after timeout |
| RT colorSpace | `NoColorSpace` | With `outputColorSpace = SRGBColorSpace` |
| Frustum cull | Group sphere at chunk centre | Per-instance is wrong on InstancedMesh |
| Shared geometry | `PlaneGeometry(1,1).translate(0,0.5,0)` | One global, fresh InstancedMesh per chunk |

Pitfalls in priority order:

1. **Double sRGB encode** — RT defaults to sRGB; without `rt.texture.colorSpace = NoColorSpace` the bake renders near-black. Verify by sampling pixel center after first bake.
2. **Bake before textures load** — capture transparent quads. Always poll `texture.image.complete` (or use the load event) before bake; one extra RAF after textures resolve to ensure GPU upload.
3. **`renderDistance ≤ foliageLodChunks`** — billboards never engage; you see a hard pop at the foliage edge. Enforce `renderDistance > foliageLodChunks + 1`.
4. **`scale.y = 0`** — pancake bug. Audit every code path that touches `scale.y` to make sure no per-instance weight can zero it.
5. **`baseScale` not applied at bake** — impostors render at wrong size. Always temp-wrap the tree group with `wrapper.scale.setScalar(baseScale)` before `setFromObject(wrapper)`.
6. **Spherical lookAt instead of Y-only** — trees roll on camera pitch. Use the world-space right-axis derivation above; `up` is hardcoded.
7. **Per-mesh InstancedBufferGeometry sharing** — corrupts attributes across chunks. One wrapper per mesh.
8. **Billboard band overlap with foliage band** — predicate must be `!trunkNear && !foliageNear && billboardBand`, not `billboardBand` alone, or you double-render at every chunk inside foliage range.

---

## 8. Common pitfalls (the trap list)

1. **BufferGeometry first-frame upload stall** — three.js has no pre-upload API. Cap drains at 1/RAF for newly-assembled InstancedMesh chunks (§3).
2. **Stale result double-build** — without `_reqLOD` echo, an upgraded request rebuilds twice (§4).
3. **`_pendingEvict` overlap** — during LOD transition, both representations can render the same area for a frame. Flag `_pendingEvict` blocks the outgoing chunk from re-emerging (§3).
4. **`window.__debug` overwrite trap** — `client/app.js` clobbers `window.__debug.vegetation` if you assign instead of extending. Add to the return object instead of reassigning.
5. **Node 22 globalThis.Worker async yield** — synchronous test loops never yield, worker postMessage callbacks never fire. Insert `await setTimeout(r, 30)` between updates in `test.js`.
6. **InstancedBufferGeometry shared trap** — wrapping ONE InstancedBufferGeometry across multiple meshes corrupts attributes. Each mesh needs its own wrapper.
7. **Sun shadow ortho recenter** — directional shadow ortho frustum must recenter on player target every frame, with castShadow band gating.
8. **MapSpinner references `document` at import time** — cannot be imported in pure Node without a DOM shim. Lazy-import client-side or guard the worker entry.

---

## 9. Tuning knobs reference

| Knob | spawnpoint value | Notes |
|------|------------------|-------|
| WORKERS_PER_POOL | min(8, hwc/2) | Sum across pools < hwc |
| renderDistance | 12 | Adaptive scales this only; must exceed foliageLodChunks |
| PHYSICS_RADIUS_CHUNKS | 2 | Independent of visual |
| mBlockSize (Jolt) | 2 | Pairs with 12ms drain |
| Drain interval | 12ms setInterval | Off physics tick |
| Frame compute drain | 4ms time-slice | CPU assembly |
| GPU upload drain | 1 chunk/RAF | Hard cap |
| Tick dilation threshold | 0.85 load | Backstop only |
| VARIANTS_PER_SPECIES | 1 | Was 6 |
| ROCK_COLLIDER_RADIUS | 32m | Distance gate |
| ROCK_SIZE_MULT | 2.0/4.5/8.0 | Per LOD band |
| HARD_LAND_FALL_VY | -8.0 m/s | Below = hard landing oneshot |
| Coyote time | 120ms | Grace after leaving ground |
| Jump buffer | 120ms | Queue during airtime |
| PHYSICS_PLAYER_DIVISOR | 3 | Pass accumDt, not dt |
| Defer-evict timeout | 800ms | Was 5s |

---

## TL;DR for mapspinner

If you are bringing mapspinner into a streaming world on top of your own terrain, the three things you will get wrong on the first try are:

1. **GPU upload pacing** — you will queue 20 chunks of trees and drop a 200ms frame. Cap at 1 chunk/RAF.
2. **LOD swap atomicity** — you will dispose the old chunk before the new one renders and get empty frames. Keep both alive during transition.
3. **Free-fall on spawn** — players spawn before vegetation/rock colliders stream in and fall through the world. Provide a synchronous inline ground sampler that matches the worker's noise output.

Everything else here is refinement on top of those three.
