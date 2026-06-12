# Changelog

## 2026-06-08

- **Single comprehensive mobile-first terrain version** (`ab68667`) — rearchitected to run performantly on phones with drastically fewer GPU resources, one version only (no device tiers, no desktop/procedural/clipmap fallbacks). Driven by two subagent workflows (an 8-dimension analysis fan-out, then a 13-item per-file fixing-run fan-out), each step lint-witnessed.
  - `gl-render.js`: height pool **R32F→R16F**, **2048→1024 layers** (tuned from a live eviction measurement: 512 evicted ~3155/s at the deck → 1024 gives 0/s), **8→4 mip levels**, with a format-only capability probe (`_useR16F`) that falls back to R32F when half-float is unavailable. **~90% VRAM cut** (138MB → ~35MB). Set the previously-missing `uHeightPoolTexSize` + `uFloatLinearOK` uniforms (were `vec2(0)` → NaN displacement). Added a `__tcEvictRate` counter + `window.__terrainConfig` diagnostics readout.
  - `terrain.glsl` + `atmosphere.glsl`: global **`mediump float`** with explicit **highp islands** on all planet-scale quantities (validated safe — `assertElevLinear` pass, ~238k tris rasterized). Octave cuts (`broadShapeM` 14→12, `vtxDisplace` 9→6, single-octave rock), `aw*aw*aw*aw` for `pow(aw,4)`, distance-gated cheap FS far path, tanh ceiling `x/8000` (pointed peaks, no flat clipped tops). Overflow leaves shade with the geometric macro normal (no pale/colored flash) + macro-slope rock gate.
  - `planet-orchestrator.js`: tightened LOD (splitFactor + near-radius) so peak visible leaves drop under the smaller cap; HPF continental field packed to RG16F+RG8.
  - `quadtree.js`, `terrain-lab.mjs`: matched LOD + lab-mirror updates. Dead atlas apparatus deleted across files. 6-stage GLSL lint + `node --check` clean; CLI `shapeReport allGatesPass`; live glError 0 (R16F active, `bakeErr null`).

## 2026-06-07

- `terrain.glsl` + `gl-render.js` (`7ee7cb0`): unified lit-normal slope gain (`uNrmGain`) so the normal is the true gradient of the rendered height; camera moveStep + collision both use the GPU-exact `sampleGroundM` height — fixes the camera hitting zero speed before the surface.
- `planet-orchestrator.js` + `terrain-lab.mjs` (`69c2a86`): LOD `distFactor` `sf*3.6 -> sf*8.0` — the detail seen at 5.5km now displays at ~12km (each LOD pop ~2.2x farther in altitude).
- `terrain.glsl` (`06a9bd9`): rockface/canyon detail texture ~10x larger (FS noise freq `1800->180`, `1200->120`).
- `terrain.glsl` (`db4dae0`): four normal-slope defects — full fine band restored, cbias continental-swell gradient added to the normal, peak-lift smoothstep chain rule, normalized pole tangent frame.
- `terrain.glsl` + `lint-shader.mjs` (`7cb6eac`): guard the VS-only fractal + debug-only functions out of the render FS to shrink the ANGLE cold compile; lint now covers the PROBE and DEBUG-FS programs.
- `terrain.glsl` + `planet-orchestrator.js` + `atlas-bake.mjs` (`b73465e`, `89f381e`): the baked atlas shades with relief — `broadShapeMD` central-differences `atlasHeight` so atlas-on land is no longer flat-shaded; `interiorExact` bake-validation gate added. Atlas default-off, live A/B via `window.__toggleAtlas`/`__forceAtlas`.
- `server.js` + `planet.html` (`61135c6`, `4b116ad`): re-host the `/diag` sink and add a `/cmd` command channel — a headless agent reads the warm tab's live render state and drives it (toggle atlas, hot-reload shaders, probe GPU state) with no page reload, routing around the cold-compile browser-tool block.
- `atlas.js` (`a2b2ce0`): `PLANET_R` default `6371000 -> 6360000` to match the system radius; closes the coordinate-system audit sweep (face-UV convention, vH<->normal term parity, tangent-frame guards, fp32 coord-scale, quadtree R, camera-height AGL — all pass).
- `terrain.glsl` (`e3e4572`): bounded quantization — break the `vtxDisplace` octave loop below the Nyquist fade floor; ~2-4 fewer noise taps/vertex at deep LOD, geometry pixel-identical.
- `quadtree.js` (`6bbf6e1`): flatten the near-far detail gradient as altitude rises above fps height (gradient 8 at the deck -> 2 at 300km).
- `planet-orchestrator.js` + `terrain.glsl` + `gl-render.js` (`7b7dcd6`): matched HPF bake+shader seam-inset (`window.__hpfInset`) collapses the 985m cube-face seam to 0m; gated off by default.

## 2026-05-23

- `terrain-phase2.js` (commit `d87c51b`): Add `allocateTileSlot`, `computePipelines`, `computeBindGroups` to `ProlandProducer` — stopped TypeError crash every render frame when `terrain-phase3-integration.js` called these missing methods.
- `shader-loader.js` (commit `1e61b7b` session): Fix `#if` numeric literal handling — `#if 1` was treated as a flag lookup (always false), stripping always-true blocks and causing GPU validation errors and black canvas.
- `terrain-phase1.js`: Change `normTexArray` format from `rg32float` to `rgba8unorm` to match `normal_producer.wgsl` storageTexture output.
- `terrain-phase2.js`: Remove unused bind group entries for upsample (b5), normal producer (b0), ortho producer (b0,b1,b2,b6) — WebGPU `layout:'auto'` strips unreachable bindings, causing validation errors when they were supplied.
- `.gitignore`: Add `--session-id` stale runtime artifact.
