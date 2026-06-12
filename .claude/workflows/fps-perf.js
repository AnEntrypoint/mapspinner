export const meta = {
  name: 'fps-perf',
  description: 'Find + attack TV8 runtime FPS bottlenecks: measure the GPU VS/FS split on a headed browser, rank the lever, cut, re-measure, verify no visual regression',
  whenToUse: 'When the live frame rate is low (close-approach / low-alt especially). Runs the measure-first loop: __diag.gpuTimer attributes frame ms to VS (per-vertex 14-oct broadShapeM x3/vertex) vs FS (per-pixel shade) vs CPU emit, ranks the most-performant lever, proposes the cut, and re-measures. Needs a headed browser (EXT_disjoint_timer_query_webgl2 is absent in headless node-gl).',
  phases: [
    { title: 'Map',     detail: 'parallel readers over the runtime frame path: gl-render.js draw/emit, planet-orchestrator.js LOD/cull, terrain.glsl VS displace + FS shade' },
    { title: 'Measure', detail: 'headed-browser __diag.gpuTimer baseline: fullMs (VS+FS), vsRasterMs (VS+raster), fsMs delta, per rung (orbit/lowalt/closeup/deck)' },
    { title: 'Rank',    detail: 'pick the dominant cost (VS triple-fractal / GRID tessellation / deck maxLevel / over-subdivision / FS per-pixel / CPU emit) as the single most-performant lever' },
    { title: 'Cut',     detail: 'one reduction for the ranked lever, with the LOD-invariance + no-seam constraint stated' },
    { title: 'Verify',  detail: 'per cut: re-measure gpuTimer delta + continuous-normals/no-seam visual witness + lab gates; each shader edit = one cold compile so batch reloads' },
  ],
}

// ----------------------------------------------------------------------------
// TV8 fps-perf workflow. The BEHIND-LIMB cull (planet-orchestrator.js limb cull + forward-cone
// rescue) already removes ~50-80% of generated leaves, and frustum cull is intentionally OFF (it
// mis-culled bulged on-screen quads). So the remaining runtime cost is the KEPT VISIBLE quads x the
// per-vertex work. The known prime lever: terrain.glsl's lit-normal FD runs TWO extra full broadShapeM
// (14-octave) calls per vertex on top of the displacement call = 3x fractal/vertex over GRID=24 ->
// (24+2)^2=676 verts/quad. broadShapeM MUST stay a pure LOD-invariant world-dir function (a per-tile /
// LOD octave fade was TRIED + REFUTED: adjacent tiles diverge at the shared edge = seams/popping), so
// any cheaper gradient must not reintroduce that seam. The measurement gap this workflow closes: node
// headless gl has no EXT_disjoint_timer_query_webgl2, so VS-vs-FS attribution was impossible -- a HEADED
// browser + __diag.gpuTimer (full frame vs __fsCheap short-circuit frame) finally names the split, so
// the lever is PICKED from a number, not guessed.
// ----------------------------------------------------------------------------

const FRAME_FILES = [
  { path: 'src/gl-render.js',           lens: 'per-frame instBuf Float32Array(n*5)+bufferData upload, drawElementsInstanced, GRID mesh size, uniform set, _lastInstQuads reuse guard' },
  { path: 'src/planet-orchestrator.js', lens: 'per-frame quadtree split (splitFactor/altSplitMul low-alt PEAK/distFactor), maxLevel + altitude-gated deck cap, behind-limb cull + forward-cone rescue, leaf emit count. NOTE the altSplitMul PEAK is the highest-payoff LOD-quad-count lever measured (2026-06-04): a LIVE window.__splitFactor sweep at the worst rung names the quads-vs-pxPerPoly knee with NO edit/cold-compile.' },
  { path: 'src/shaders/terrain.glsl',   lens: 'VS vtxDisplace + the lit-normal FD (2 extra full broadShapeM/vertex), broadShapeM octave count, FS per-pixel shade (riverMask/strata/biome/detail-normal taps)' },
]

const COST_SCHEMA = {
  type: 'object',
  properties: {
    costs: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name:        { type: 'string' },
          file:        { type: 'string' },
          stage:       { type: 'string', description: 'VS | FS | CPU-emit | LOD-quad-count' },
          perWhat:     { type: 'string', description: 'per-vertex | per-pixel | per-quad | per-frame' },
          cutIdea:     { type: 'string' },
          lodInvariant:{ type: 'boolean', description: 'true if the cut keeps broadShapeM a pure LOD-invariant world-dir fn (no seam)' },
          regressionRisk: { type: 'string', description: 'low | medium | high, and the witness that proves no-regression (continuous normals / no seam / lab gate)' },
        },
        required: ['name', 'file', 'stage', 'cutIdea', 'regressionRisk'],
      },
    },
  },
  required: ['costs'],
}

phase('Map')
const maps = await parallel(FRAME_FILES.map(f => () =>
  agent(
    `Read ${f.path} in the TV8 repo and map every per-frame RUNTIME cost it carries (focus: ${f.lens}). ` +
    `For each cost return name, file, stage (VS|FS|CPU-emit|LOD-quad-count), perWhat, a concrete cutIdea, ` +
    `whether it keeps broadShapeM LOD-invariant (no seam), and regressionRisk with the witness that proves no-regression. ` +
    `Do NOT edit anything; this is a read+map pass.`,
    { label: `map:${f.path.split('/').pop()}`, phase: 'Map', schema: COST_SCHEMA }
  )
))
const costs = maps.filter(Boolean).flatMap(m => m.costs)

// MEASURE is a HEADED-browser dispatch the human (or the gm chain) runs OUTSIDE this workflow, because
// the workflow's subagents have no browser surface -- they reason over the map + the gpuTimer numbers
// the chain captured. The chain passes the measured split in via args.measured (fullMs/vsRasterMs/fsMs
// per rung). If absent, the Rank agent reasons from the static map + the known triple-fractal prior.
// MEASURE health-check FIRST (2026-06-04 lesson): before trusting any gpuTimer number, read
// window.__pageErr and confirm dbg.quads()>0 with __altM tracking dbg.gotoDown(km). A FROZEN __altM +
// zero quads + the loading overlay still up is a CRASHED render loop (e.g. a ReferenceError thrown
// per-frame), NOT a slow one -- gpuTimer will read ~0.002ms on an empty frame and mislead the rank.
// LIVEST lever-finding: sweep window.__splitFactor live ([2.0,1.4,1.0,0.7]) + orch.clearCache() at the
// worst rung, reading gpuTimer + pxPerPoly each step; the quads/vsRaster/pxPerPoly knee names the cut
// with no source edit and no ~190s cold compile. Then bake the winning value into altSplitMul's PEAK.
phase('Measure')
const measured = (typeof args === 'object' && args && args.measured) ? args.measured : null
log(measured
  ? `gpuTimer split provided: ${JSON.stringify(measured)}`
  : 'no measured split passed (args.measured); ranking from the static map + the triple-fractal VS prior')

phase('Rank')
const ranking = await agent(
  `Rank these TV8 per-frame runtime costs to pick the SINGLE most-performant FPS lever. ` +
  `Context: behind-limb cull already removes ~50-80% of leaves; frustum cull is intentionally OFF; the ` +
  `prime known cost is the VS running 3x full broadShapeM/vertex (displace + 2 FD taps) over 676 verts/quad. ` +
  (measured ? `MEASURED gpuTimer split (authoritative): ${JSON.stringify(measured, null, 2)}. ` : `No live measurement; weight the VS triple-fractal prior. `) +
  `Pick the lever with the highest payoff that keeps broadShapeM LOD-invariant (no seam). Costs:\n${JSON.stringify(costs, null, 2)}`,
  { label: 'rank', phase: 'Rank', schema: {
      type: 'object',
      properties: {
        lever: { type: 'string', description: 'the single chosen lever' },
        stage: { type: 'string', description: 'VS | FS | CPU-emit | LOD-quad-count' },
        rationale: { type: 'string' },
        ranked: { type: 'array', items: { type: 'object', properties: {
          name: { type: 'string' }, rank: { type: 'number' }, expectedSaving: { type: 'string' },
          gate: { type: 'string' } }, required: ['name', 'rank', 'expectedSaving', 'gate'] } },
      }, required: ['lever', 'stage', 'ranked'] } }
)

phase('Cut')
// Propose the concrete edit for the chosen lever + the witness that proves it (re-measured gpuTimer
// delta + continuous-normals/no-seam visual witness). Adversarially verify each before it is applied.
const cuts = await pipeline(
  ranking.ranked.sort((a, b) => a.rank - b.rank),
  sec => agent(
    `Propose the concrete code edit for TV8 FPS lever "${sec.name}" (expected saving ${sec.expectedSaving}). ` +
    `Give the exact file, the old snippet, the new snippet, and the WITNESS that proves it: the re-measured ` +
    `__diag.gpuTimer fullMs/fsMs delta AND the visual invariant (continuous normals across a tile boundary, ` +
    `no seam/popping, glError 0). Gate: ${sec.gate}. broadShapeM MUST stay a pure LOD-invariant world-dir fn ` +
    `(a per-tile / screen-space gradient that diverges at shared edges = the refuted seam). If risk is high, say so.`,
    { label: `cut:${sec.name}`, phase: 'Cut', schema: {
        type: 'object',
        properties: {
          file: { type: 'string' }, oldSnippet: { type: 'string' }, newSnippet: { type: 'string' },
          witness: { type: 'string' }, safe: { type: 'boolean' }, notes: { type: 'string' },
        }, required: ['file', 'witness', 'safe'] } }
  ),
  (proposal, sec) => agent(
    `Adversarially verify this TV8 FPS cut for "${sec.name}". Could it reintroduce a tile-edge seam, break ` +
    `LOD invariance, change the rendered terrain (shape/biome/lighting), break a uniform, or fail to compile? ` +
    `Default to safe=false if uncertain. Proposal:\n${JSON.stringify(proposal, null, 2)}`,
    { label: `verify:${sec.name}`, phase: 'Verify', schema: {
        type: 'object',
        properties: { real: { type: 'boolean' }, safe: { type: 'boolean' }, reason: { type: 'string' } },
        required: ['real', 'safe', 'reason'] } }
  ).then(v => ({ section: sec.name, proposal, verdict: v }))
)

const safe = cuts.filter(Boolean).filter(c => c.proposal.safe && c.verdict.safe)
const risky = cuts.filter(Boolean).filter(c => !(c.proposal.safe && c.verdict.safe))
return {
  lever: ranking.lever,
  stage: ranking.stage,
  rationale: ranking.rationale,
  measured,
  ranked: ranking.ranked,
  safeCuts: safe.map(c => ({ section: c.section, file: c.proposal.file, witness: c.proposal.witness })),
  gatedCuts: risky.map(c => ({ section: c.section, reason: c.verdict.reason })),
  note: 'Re-measure __diag.gpuTimer after each cut on a headed browser; broadShapeM stays LOD-invariant ' +
        '(no per-tile/screen-space gradient that seams). Batch terrain.glsl edits into ONE reload ' +
        '(each shader edit = a fresh ~188s cold compile). Witness continuous normals + glError 0 before COMPLETE.',
}
