export const meta = {
  name: 'speed-dna',
  description: 'Optimize every part of mapspinner speed (cold compile + FPS) through the 12-principle synthesized-engineering-DNA lens: measure-first, subtract before adding, physics-first constraints, empirical cross-axis arbitration, adversarial verify. Fans out subagents; composes the fps-perf + startup-perf workflows as sub-steps.',
  whenToUse: 'When the user wants a holistic speed pass over BOTH cold-compile time AND runtime FPS, ranked and arbitrated as one budget rather than two independent passes. Honours the measured-dead-lever record (FS source-shrink is dead; atlas activation is the only real compile lever) so it does not re-chase refuted work.',
  phases: [
    { title: 'Constraints', detail: 'P5 physics-first: enumerate the hard speed constraints (ANGLE cold-HLSL translation cost is intrinsic + driver-cached; no getProgramBinary; VS 14-oct broadShapeM x3/vertex; behind-limb cull on, frustum off; measured-dead levers from recall) before any agent proposes a cut' },
    { title: 'Measure',     detail: 'P7 empirical: gather the live numbers that exist (args.measured gpuTimer VS/FS split per rung; args.compile cold/warm ms + translated-HLSL size). Absent numbers are flagged UNMEASURED, never guessed past.' },
    { title: 'Map',         detail: 'fan-out readers map every per-frame (FPS) and per-init (compile) cost across gl-render.js / planet-orchestrator.js / terrain.glsl, each tagged with the DNA principle that judges it' },
    { title: 'Arbitrate',   detail: 'P2 subtractive + P9 worst-case + P12 human-value: rank ALL levers across both axes into ONE budget; resolve cross-axis conflicts (a compile cut that costs FPS or vice versa); drop measured-dead levers; pick the single highest-payoff worst-case-respecting move first' },
    { title: 'Cut',         detail: 'P3 evolutionary + P6 adversarial: propose the concrete edit for each surviving lever with its no-regression witness; adversarially verify (could it seam / break LOD-invariance / change terrain / fail compile / regress the OTHER axis)' },
    { title: 'Plan',        detail: 'P10 honest interface: return the DNA-justified, witness-bearing, cross-axis-arbitrated cut plan; cold edits BATCHED into one reload (each shader edit = one ~110s cold compile)' },
  ],
}

// ----------------------------------------------------------------------------
// speed-dna -- the unifying speed workflow. The repo already carries two single-axis
// workflows (fps-perf, startup-perf); this one does NOT duplicate them -- it composes them
// (P4 composition spine: each adds one capability, this one adds cross-axis arbitration on
// top) and overlays the 12 DNA principles as the explicit ranking lens.
//
// The load-bearing prior the DNA demands we respect (P7 measure-don't-assume, P3 revert-and-
// don't-rechase): COLD COMPILE FS SOURCE-SHRINK IS MEASURED-DEAD (recall
// tv8-compile-time-every-approach-evaluated-2026-06-05: guarding VS-only funcs out of the FS
// changed cold 109124->118939ms = driver noise; ANGLE already DCEs unused funcs; the cost is
// INTRINSIC ANGLE->HLSL translation of the FS main(), not char/branch/loop count). The ONLY
// real compile lever is ATLAS ACTIVATION (removes the per-vertex fractal from BOTH stages) --
// see the atlas-pivot workflow. So this workflow's compile axis routes to atlas, NOT to FS
// trimming, and any agent that proposes FS source-shrink for cold compile is contradicting a
// measured result. Warm reload is already 41ms (driver program cache persists across reloads;
// cold recurs only on SOURCE CHANGE = dev edits, or first-ever cold driver cache).
//
// The FPS prime lever (also measured): the VS runs 3x full 14-oct broadShapeM/vertex (displace
// + 2 central-diff taps) over the GRID mesh; over-tessellation was sub-pixel (GRID 24->16 cut
// verts/quad 676->324). broadShapeM MUST stay a pure LOD-invariant world-dir fn -- a per-tile /
// screen-space octave fade was TRIED + REFUTED (adjacent tiles diverge at the shared edge =
// seams/popping), so no FPS cut may reintroduce that seam (P6 adversarial: the refuted misuse
// will recur unless structurally blocked).
//
// args (all optional): { measured: <gpuTimer split per rung {fullMs,vsRasterMs,fsMs}>,
//                        compile: <{coldMs, warmMs, translatedFsChars, translatedVsChars}>,
//                        runFps: <bool, run the fps-perf child>, runStartup: <bool, run startup-perf child> }
// ----------------------------------------------------------------------------

const A = (typeof args === 'object' && args) ? args : {}
const measured = A.measured || null
const compile  = A.compile  || null

// The DNA principles, as the ranking lens every agent is handed. Kept as data (P1 data-first):
// the workflow does not branch on principle names, it passes them as the judgement frame.
const DNA = [
  'P1 data-first: is the cost a state/data-model problem (re-bake, cache shape) or genuinely compute?',
  'P2 subtractive: can the cost be REMOVED entirely rather than made faster? (kill an octave, a tap, a debug branch)',
  'P5 physics-first: what hardware/driver constraint bounds this? (ANGLE translation, ALU count, bandwidth, vert count)',
  'P7 empirical: is there a MEASURED number, or is this a guess? Refuse measured-dead levers (FS source-shrink for cold compile).',
  'P9 worst-case: does the cut help the WORST rung (close-approach / cold first-load), or only the average?',
  'P12 human-value: does a real user FEEL this? (a 110s once-per-cold-cache compile vs a per-frame stutter)',
]

const SPEED_FILES = [
  { path: 'src/gl-render.js',           lens: 'COMPILE: terrain.glsl compile/link, KHR_parallel_shader_compile non-blocking path, probe+debug program split, GRID mesh size. FPS: per-frame instBuf upload, drawElementsInstanced, verts/quad.' },
  { path: 'src/planet-orchestrator.js', lens: 'COMPILE: HPF bake (HPF_RES^2*6 sampleUV), anchor field, init timings. FPS: per-frame quadtree split (splitFactor/altSplitMul/maxLevel), behind-limb cull + forward-cone rescue, leaf emit count, deck cap.' },
  { path: 'src/shaders/terrain.glsl',   lens: 'COMPILE: FS main() intrinsic ANGLE-HLSL cost (NOT source-trimmable -- measured dead), debug displayMode blocks. FPS: VS vtxDisplace + lit-normal central-diff (2 extra full broadShapeM/vertex), broadShapeM octave count, FS per-pixel shade taps.' },
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
          axis:        { type: 'string', description: 'compile | fps | both' },
          stage:       { type: 'string', description: 'VS | FS | CPU-emit | LOD-quad-count | js-init | cold-compile' },
          perWhat:     { type: 'string', description: 'per-vertex | per-pixel | per-quad | per-frame | per-init | once-cold' },
          cutIdea:     { type: 'string' },
          dnaPrinciple:{ type: 'string', description: 'which DNA principle most justifies attacking this (e.g. P2 subtractive)' },
          measuredDead:{ type: 'boolean', description: 'true if this is a known measured-dead lever (e.g. FS source-shrink for cold compile)' },
          crossAxisRisk:{ type: 'string', description: 'does cutting this regress the OTHER axis? (e.g. removing an octave helps FPS+compile but loses relief; an atlas helps compile but adds a sampler tap to FPS)' },
          lodInvariant:{ type: 'boolean', description: 'true if an FPS cut keeps broadShapeM a pure LOD-invariant world-dir fn (no seam); n/a-> true for compile-only' },
          regressionRisk: { type: 'string', description: 'low | medium | high + the witness that proves no-regression' },
        },
        required: ['name', 'file', 'axis', 'stage', 'cutIdea', 'dnaPrinciple', 'regressionRisk'],
      },
    },
  },
  required: ['costs'],
}

// --- Constraints (P5 physics-first) -- stated as data, before any agent reasons -------------
phase('Constraints')
const CONSTRAINTS = [
  'ANGLE cold HLSL translation of the terrain FS+VS main() is INTRINSIC (~110-120s on AMD); not reducible by source char/branch/loop count (MEASURED dead 2026-06-05). Driver-cached: warm reload 41ms; cold recurs only on source change (dev) or first-ever cache.',
  'No getProgramBinary on ANGLE-AMD -> the browser ANGLE blob cache is the only persistence; cannot ship a precompiled binary.',
  'The ONLY real compile lever is atlas activation (removes the per-vertex fractal from BOTH VS+FS translation units) -- routes to the atlas-pivot workflow, not FS trimming.',
  'FPS: VS runs 3x full 14-oct broadShapeM/vertex (displace + 2 central-diff taps) over the GRID mesh; behind-limb cull removes ~50-80% leaves; frustum cull intentionally OFF (mis-culled bulged on-screen quads).',
  'broadShapeM MUST stay a pure LOD-invariant world-dir fn: per-tile/screen-space octave fade TRIED + REFUTED (seams at shared tile edges).',
  'Each terrain.glsl edit = one fresh ~110s cold compile -> BATCH all shader edits into one reload.',
]
log('Physics-first constraints fixed (' + CONSTRAINTS.length + '); measured-dead levers will be refused, not re-chased.')

// --- Measure (P7 empirical) -----------------------------------------------------------------
phase('Measure')
log(measured ? ('gpuTimer split provided: ' + JSON.stringify(measured))
             : 'NO gpuTimer split passed (args.measured) -> FPS ranking flags every lever UNMEASURED; weight the VS triple-fractal prior, do not assert a number.')
log(compile  ? ('compile numbers provided: ' + JSON.stringify(compile))
             : 'NO compile numbers passed (args.compile) -> compile ranking uses the cold-HLSL-intrinsic prior; FS source-shrink stays measured-dead.')

// --- Map (fan-out) --------------------------------------------------------------------------
phase('Map')
const maps = await parallel(SPEED_FILES.map(f => () =>
  agent(
    'Read ' + f.path + ' in the mapspinner repo and map every SPEED cost it carries across BOTH axes (focus: ' + f.lens + '). ' +
    'Judge each cost through the synthesized-engineering-DNA lens:\n' + DNA.map(d => '  - ' + d).join('\n') + '\n' +
    'For each cost return name, file, axis (compile|fps|both), stage, perWhat, a concrete cutIdea, the dnaPrinciple that most justifies attacking it, ' +
    'measuredDead (true for FS source-shrink as a cold-compile lever and any other refuted lever), crossAxisRisk (does cutting it regress the other axis?), ' +
    'lodInvariant (does an FPS cut keep broadShapeM seam-free?), and regressionRisk + the witness. ' +
    'HARD CONSTRAINTS you must respect (do not propose anything that violates them):\n' + CONSTRAINTS.map(c => '  - ' + c).join('\n') + '\n' +
    'Do NOT edit anything; this is a read+map pass.',
    { label: 'map:' + f.path.split('/').pop(), phase: 'Map', schema: COST_SCHEMA }
  )
))
const costs = maps.filter(Boolean).flatMap(m => m.costs)
const live  = costs.filter(c => !c.measuredDead)
const dead  = costs.filter(c => c.measuredDead)
log('mapped ' + costs.length + ' speed costs (' + live.length + ' live, ' + dead.length + ' measured-dead and dropped)')

// --- Arbitrate (P2 + P9 + P12) -- one budget across both axes, cross-axis conflicts resolved -
phase('Arbitrate')
const ranking = await agent(
  'You are arbitrating the mapspinner speed budget across BOTH axes (cold-compile time AND runtime FPS) as ONE ranked plan, ' +
  'through the synthesized-engineering-DNA lens. Rank the LIVE levers below into a single ordered list, highest-payoff first, ' +
  'applying:\n' +
  '  - P2 subtractive: a lever that REMOVES a cost outright outranks one that merely speeds it.\n' +
  '  - P9 worst-case: a lever that helps the WORST rung (close-approach FPS / first-load cold compile) outranks an average-case win.\n' +
  '  - P12 human-value: weight by what a real user FEELS (a once-per-cold-cache compile is felt once; a per-frame deck stutter is felt continuously).\n' +
  '  - P7 empirical: levers WITHOUT a measured number are ranked but flagged unmeasured; never rank a measured-dead lever (already dropped).\n' +
  'CRITICALLY, resolve every CROSS-AXIS conflict explicitly: if a compile cut costs FPS (e.g. an atlas sampler tap) or an FPS cut costs compile/quality (e.g. dropping octaves loses relief), state the trade and which axis wins and WHY (cite the principle). ' +
  (measured ? 'MEASURED gpuTimer split (authoritative for FPS): ' + JSON.stringify(measured) + '. ' : 'No FPS measurement -> weight the VS triple-fractal prior. ') +
  (compile ? 'MEASURED compile numbers: ' + JSON.stringify(compile) + '. ' : 'No compile measurement -> cold-HLSL-intrinsic prior; atlas is the only real compile lever. ') +
  'Live levers:\n' + JSON.stringify(live, null, 2),
  { label: 'arbitrate', phase: 'Arbitrate', schema: {
      type: 'object',
      properties: {
        topLever: { type: 'string', description: 'the single highest-payoff move to do first' },
        topAxis:  { type: 'string', description: 'compile | fps | both' },
        rationale:{ type: 'string', description: 'why this one first, in DNA-principle terms' },
        crossAxisConflicts: { type: 'array', items: { type: 'object', properties: {
          conflict: { type: 'string' }, winningAxis: { type: 'string' }, principle: { type: 'string' } },
          required: ['conflict', 'winningAxis', 'principle'] } },
        ranked: { type: 'array', items: { type: 'object', properties: {
          name: { type: 'string' }, axis: { type: 'string' }, rank: { type: 'number' },
          expectedSaving: { type: 'string' }, dnaPrinciple: { type: 'string' }, gate: { type: 'string' } },
          required: ['name', 'axis', 'rank', 'expectedSaving', 'gate'] } },
      }, required: ['topLever', 'topAxis', 'ranked'] } }
)

// --- Cut (P3 evolutionary propose + P6 adversarial verify) ----------------------------------
phase('Cut')
const cuts = await pipeline(
  ranking.ranked.sort((a, b) => a.rank - b.rank),
  sec => agent(
    'Propose the concrete code edit for mapspinner speed lever "' + sec.name + '" (axis ' + sec.axis + ', expected saving ' + sec.expectedSaving + ', principle ' + (sec.dnaPrinciple || 'n/a') + '). ' +
    'Give the exact file, the old snippet, the new snippet, and the WITNESS that proves it with no regression on EITHER axis: ' +
    'for FPS -> re-measured __diag.gpuTimer fullMs/fsMs delta AND continuous normals across a tile boundary (no seam/popping) AND glError 0; ' +
    'for compile -> translated-HLSL size delta via WEBGL_debug_shaders OR a lab gate (maxElev/landFrac/hypsometry unchanged) OR node --check, AND a note that it does NOT regress FPS. ' +
    'Gate: ' + sec.gate + '. broadShapeM MUST stay a pure LOD-invariant world-dir fn (the refuted per-tile/screen-space gradient = seam). ' +
    'If this lever is FS source-shrink for cold compile, STOP and return safe=false (measured-dead). If risk is high, say so.',
    { label: 'cut:' + sec.name, phase: 'Cut', schema: {
        type: 'object',
        properties: {
          file: { type: 'string' }, oldSnippet: { type: 'string' }, newSnippet: { type: 'string' },
          witness: { type: 'string' }, safe: { type: 'boolean' }, notes: { type: 'string' },
        }, required: ['file', 'witness', 'safe'] } }
  ),
  (proposal, sec) => agent(
    'Adversarially verify this mapspinner speed cut for "' + sec.name + '" (axis ' + sec.axis + '). Could it: reintroduce a tile-edge seam, break LOD invariance, ' +
    'change the rendered terrain (shape/biome/lighting), break a uniform, fail to compile, OR REGRESS THE OTHER SPEED AXIS (an FPS cut that bloats compile, or a compile cut that adds per-pixel/per-vertex work)? ' +
    'Is it a measured-dead lever (FS source-shrink for cold compile) dressed up? Default to safe=false if uncertain. Proposal:\n' + JSON.stringify(proposal, null, 2),
    { label: 'verify:' + sec.name, phase: 'Cut', schema: {
        type: 'object',
        properties: { real: { type: 'boolean' }, safe: { type: 'boolean' }, regressesOtherAxis: { type: 'boolean' }, reason: { type: 'string' } },
        required: ['real', 'safe', 'reason'] } }
  ).then(v => ({ section: sec.name, axis: sec.axis, proposal, verdict: v }))
)

// --- Plan (P10 honest interface) ------------------------------------------------------------
phase('Plan')
const ok    = cuts.filter(Boolean).filter(c => c.proposal.safe && c.verdict.safe && !c.verdict.regressesOtherAxis)
const gated = cuts.filter(Boolean).filter(c => !(c.proposal.safe && c.verdict.safe && !c.verdict.regressesOtherAxis))

return {
  workflow: 'speed-dna',
  constraints: CONSTRAINTS,
  measuredDeadDropped: dead.map(c => ({ name: c.name, axis: c.axis, why: c.cutIdea })),
  topLever: ranking.topLever,
  topAxis: ranking.topAxis,
  rationale: ranking.rationale,
  crossAxisConflicts: ranking.crossAxisConflicts || [],
  ranked: ranking.ranked,
  safeCuts: ok.map(c => ({ section: c.section, axis: c.axis, file: c.proposal.file, witness: c.proposal.witness })),
  gatedCuts: gated.map(c => ({ section: c.section, axis: c.axis, reason: c.verdict.reason })),
  note: 'Apply safeCuts highest-rank-first. BATCH all terrain.glsl edits into ONE reload (each = a fresh ~110s cold compile). ' +
        'Re-measure __diag.gpuTimer (FPS) + translated-HLSL size (compile) after each batch on a HEADED browser. ' +
        'broadShapeM stays LOD-invariant (no per-tile/screen-space gradient that seams). FS source-shrink for cold compile is measured-dead -- do not apply. ' +
        'The real compile lever is atlas activation (atlas-pivot workflow). Witness continuous normals + glError 0 before COMPLETE.',
}
