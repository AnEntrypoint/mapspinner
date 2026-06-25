export const meta = {
  name: 'fps-2x',
  description: 'Exhaustive mapspinner FPS-doubling sweep: fan out per-surface finders for VISUAL-NEUTRAL frame-time cuts, adversarially verify each preserves the render byte-for-byte, arbitrate into one ranked cut plan toward a ~2x target.',
  whenToUse: 'When the user wants every possible FPS win with ZERO visual change (the "double the fps, do not touch the look" ask). Broader + more adversarial than fps-perf: covers VS-vertex-throughput, FS, CPU-frame, draw-state, LOD/quad-count, memory/bandwidth, AND dead-code, and every candidate must carry a same-pose visual-neutrality witness (reliefSD/albedoSD/shadingSD + screenshot) or it is dropped. Needs a headed browser for __diag.gpuTimer (EXT_disjoint_timer_query_webgl2 is absent in headless node-gl); the chain feeds measured numbers in via args.measured.',
  phases: [
    { title: 'Map',       detail: 'parallel readers map every per-frame runtime cost across 7 surfaces' },
    { title: 'Rank',      detail: 'arbitrate candidates against the LIVE-MEASURED doctrine (throughput-bound; ALU/THC/octave dead; FS ~17%)' },
    { title: 'Propose',   detail: 'concrete edit per ranked candidate + the exact visual-neutrality + gpuTimer witness' },
    { title: 'Verify',    detail: 'adversarial refute-by-default: does it change the render, seam, break LOD, or fail compile?' },
    { title: 'Plan',      detail: 'synthesize surviving cuts into one ordered apply-and-measure plan' },
  ],
}

// ----------------------------------------------------------------------------
// mapspinner fps-2x. The DOCTRINE this workflow is built on (all LIVE-MEASURED, ANGLE AMD D3D11,
// oblique 6km deck, 537 quads, GRID 11):
//   - fullMs 21.1 = vsRasterMs 17.2 (81%) + fsMs 3.9 (19%). VS+raster is the budget.
//   - The bottleneck is VERTEX/TRIANGLE THROUGHPUT (vertex count x raster), NOT per-vertex ALU:
//       octMax 12->3 (4x less fractal ALU) left fullMs FLAT (20.4->20.4).
//   - THC height-pool (O(1) baked fetch replacing composeHeight) is a DEAD net lever: it cuts
//       vsRaster 16.9->13.2 but pushes fsMs 3.6->7.6, fullMs flat. Do NOT re-chase ALU-class cuts.
//   - So the ONLY live levers are: (a) fewer vertices (GRID down / fewer quads via splitFactor+cull),
//       (b) FS per-pixel cost (now 19%, no longer 1%), (c) CPU-frame + draw-state (instbuf cache,
//       redundant uniform/bind uploads, getError sync), (d) dead-code removal.
//   - GRID 11->8 is the headline vertex-count lever but is BLOCKED by jagged biome-crossover lines
//       (climate varying interpolated across coarse tris). The unblock = per-PIXEL FS biome sampling.
//   - REFUTED, do not re-attempt: 3-tap forward normal (biases slope -> rock-everywhere up close),
//       analytic-derivative normal (broadShapeMD, node-rejected), baked-octa normal (flattens shading).
// The VISUAL GATE is absolute: every candidate must prove the render is unchanged at the same pose
// (reliefSD/albedoSD/shadingSD within tolerance + screenshot identical) or it is dropped.
// ----------------------------------------------------------------------------

const SURFACES = [
  { key: 'vs-throughput', path: 'src/shaders/terrain.glsl + src/gl-render.js', lens: 'per-vertex VS cost that scales with VERTEX COUNT: GRID mesh size, skirt ring verts, the lit-normal FD tap count, vtxDisplace. Vertex-count reductions only (ALU is proven off the critical path). Any GRID drop must name how the biome-crossover jaggies are prevented.' },
  { key: 'lod-quad-count', path: 'src/planet-orchestrator.js', lens: 'leaf/quad count per frame: splitFactor, altSplitMul PEAK + high-alt holdoff/popBoost, maxLevel, near-radius, behind-limb cull + forward-cone rescue, frustum/horizon cull. Any quad drawn but sub-pixel or off-screen = wasted throughput.' },
  { key: 'fs-shade', path: 'src/shaders/terrain.glsl FS + atmosphere.glsl', lens: 'per-pixel FS cost (now ~19% of frame): biome/strata/river/canyon masks, biplanar+2-scale RNM rock detail-normal, slope/gorge AO, analytic aerial perspective, snoise3 tap count, branches always-taken vs always-skipped at runtime defaults, taps multiplied by 0.' },
  { key: 'cpu-frame', path: 'src/planet-orchestrator.js + src/gl-render.js', lens: 'CPU per-frame: quadtree split/update, instance-buffer Float32Array rebuild + bufferData upload, draw-state setup, sampleGroundM collision readback, per-frame getError/finish sync stalls, work repeated on non-moved frames.' },
  { key: 'draw-state', path: 'src/gl-render.js', lens: 'redundant GL state per frame: useProgram/bind*/uniform* uploaded every frame though invariant, redundant VAO/FBO/texture binds, getUniformLocation in the hot path, getError sync points.' },
  { key: 'memory-bandwidth', path: 'src/gl-render.js + src/planet-orchestrator.js', lens: 'vertex-fetch + texture bandwidth: instance/vertex attr widths (FLOAT vs SHORT/HALF_FLOAT, packed normals), HPF/pool texture formats + filtering, overdraw. Bounded quantization that keeps planet-scale positions exact (highp islands).' },
  { key: 'dead-code', path: 'src/ + scripts/ + shaders', lens: 'unused exports, unreferenced fns/vars, dead window.__ levers, always-zero uniforms whose JS setter is gone, #ifdef branches never compiled in the render program, inlined-to-0 noise sites, orphan files. Removal is visual-neutral by definition; each needs a zero-live-reference witness.' },
]

const COST_SCHEMA = {
  type: 'object',
  properties: {
    costs: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' },
      surface: { type: 'string' },
      stage: { type: 'string', description: 'VS | FS | CPU-frame | draw-state | LOD-quad-count | memory | dead-code' },
      perWhat: { type: 'string', description: 'per-vertex | per-pixel | per-quad | per-frame | static' },
      cutIdea: { type: 'string' },
      visualNeutral: { type: 'boolean', description: 'true if the cut provably does NOT change the rendered image' },
      visualWitness: { type: 'string', description: 'the exact metric/screenshot test that proves no visual change' },
      expectedSaving: { type: 'string' },
      regressionRisk: { type: 'string', description: 'low|medium|high + why' },
    }, required: ['name', 'surface', 'stage', 'cutIdea', 'visualNeutral', 'expectedSaving'] } },
  },
  required: ['costs'],
}

phase('Map')
const maps = await parallel(SURFACES.map(s => () =>
  agent(
    `Read the relevant mapspinner files for the "${s.key}" surface (${s.path}) and map every per-frame RUNTIME cost it carries. Lens: ${s.lens}\n\n` +
    `HARD CONSTRAINT: the goal is ~2x FPS with ZERO visual change. For each cost give name, surface="${s.key}", stage, perWhat, a concrete cutIdea, visualNeutral (true ONLY if it cannot change the rendered image), the exact visualWitness that proves it, expectedSaving, regressionRisk.\n` +
    `DOCTRINE (live-measured, do not contradict without measurement): bottleneck is vertex/triangle THROUGHPUT not ALU; octave/THC/ALU cuts are DEAD (fullMs flat); FS is ~19%; GRID 11->8 needs per-pixel biome first; forward-normal + analytic-derivative + baked-octa normals are REFUTED. Do NOT propose those. Do NOT edit anything; read+map only.`,
    { label: `map:${s.key}`, phase: 'Map', schema: COST_SCHEMA }
  )
))
const costs = maps.filter(Boolean).flatMap(m => m.costs)

phase('Rank')
const measured = (typeof args === 'object' && args && args.measured) ? args.measured : null
log(measured ? `measured gpuTimer split: ${JSON.stringify(measured)}` : 'no args.measured passed; ranking from the live-measured doctrine baseline (full 21.1 / vs 17.2 / fs 3.9 @ deck)')
const ranking = await agent(
  `Arbitrate these mapspinner per-frame costs into a ranked FPS-doubling plan (target ~2x, ZERO visual change). ` +
  (measured ? `MEASURED split (authoritative): ${JSON.stringify(measured)}. ` : `Baseline doctrine: full 21.1ms = vs 17.2 (81%) + fs 3.9 (19%) @ oblique 6km deck. `) +
  `Drop any candidate that is not visualNeutral or contradicts the dead-lever record (ALU/octave/THC/forward-normal/analytic-derivative). ` +
  `Weight by expectedSaving x confidence. The headline lever (per-pixel biome -> GRID 11->8) and FS-deadwork + CPU/draw cuts likely dominate. Costs:\n${JSON.stringify(costs, null, 2)}`,
  { label: 'rank', phase: 'Rank', schema: { type: 'object', properties: {
    summary: { type: 'string' },
    reachableSpeedup: { type: 'string', description: 'honest estimate of total achievable speedup with zero visual change, e.g. ~1.4x' },
    ranked: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, rank: { type: 'number' }, surface: { type: 'string' },
      expectedSaving: { type: 'string' }, visualWitness: { type: 'string' } },
      required: ['name', 'rank', 'expectedSaving', 'visualWitness'] } },
  }, required: ['summary', 'reachableSpeedup', 'ranked'] } }
)

phase('Propose')
const cuts = await pipeline(
  ranking.ranked.sort((a, b) => a.rank - b.rank),
  sec => agent(
    `Propose the concrete code edit for mapspinner FPS lever "${sec.name}" (${sec.surface}, expected ${sec.expectedSaving}). ` +
    `Give file, exact oldSnippet, newSnippet, the re-measured __diag.gpuTimer fullMs delta to expect, and the VISUAL-NEUTRALITY witness (${sec.visualWitness}): same-pose reliefSD/albedoSD/shadingSD within tolerance + screenshot identical + glError 0. If the edit could change ANY pixel, say safe=false.`,
    { label: `propose:${sec.name}`, phase: 'Propose', schema: { type: 'object', properties: {
      file: { type: 'string' }, oldSnippet: { type: 'string' }, newSnippet: { type: 'string' },
      gpuWitness: { type: 'string' }, visualWitness: { type: 'string' }, safe: { type: 'boolean' }, notes: { type: 'string' },
    }, required: ['file', 'visualWitness', 'safe'] } }
  ),
  (proposal, sec) => agent(
    `Adversarially REFUTE this mapspinner FPS cut for "${sec.name}". Could it change the rendered image (any pixel), reintroduce a tile-edge seam, break LOD invariance, alter biome/lighting/shape, break a uniform, regress a refuted lever, or fail to compile? Default real=false/safe=false if ANY doubt. Proposal:\n${JSON.stringify(proposal, null, 2)}`,
    { label: `verify:${sec.name}`, phase: 'Verify', schema: { type: 'object', properties: {
      real: { type: 'boolean' }, safe: { type: 'boolean' }, visualRisk: { type: 'string' }, reason: { type: 'string' },
    }, required: ['real', 'safe', 'reason'] } }
  ).then(v => ({ section: sec.name, surface: sec.surface, proposal, verdict: v }))
)

phase('Plan')
const safe = cuts.filter(Boolean).filter(c => c.proposal.safe && c.verdict.safe && c.verdict.real)
const gated = cuts.filter(Boolean).filter(c => !(c.proposal.safe && c.verdict.safe && c.verdict.real))
return {
  summary: ranking.summary,
  reachableSpeedup: ranking.reachableSpeedup,
  measured,
  applyOrder: safe.map(c => ({ section: c.section, surface: c.surface, file: c.proposal.file, gpuWitness: c.proposal.gpuWitness, visualWitness: c.proposal.visualWitness })),
  gated: gated.map(c => ({ section: c.section, reason: c.verdict.reason, visualRisk: c.verdict.visualRisk })),
  note: 'Apply ONE cut at a time on a headed browser; re-measure __diag.gpuTimer AND assert the same-pose visual metrics + screenshot are unchanged before keeping it. Any perceptible change = revert that cut first, investigate second. Batch terrain.glsl edits into one reloadShaders (each shader edit = a cold compile). If the honest reachable speedup is below 2x, report the measured ceiling with per-lever attribution rather than claiming the target.',
}
