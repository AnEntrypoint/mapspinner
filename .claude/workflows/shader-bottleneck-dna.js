export const meta = {
  name: 'shader-bottleneck-dna',
  description: 'Profile + eliminate every possible bottleneck in the TV8 shaders (terrain.glsl VS+FS, water path, probe variant) through the synthesized-engineering-DNA lens: physics-first constraints, measure-first, subtract-before-add, adversarial refute-by-default verify. Fans out per-surface finders + web-research agents; outputs a ranked, witness-bearing cut plan.',
  whenToUse: 'When the user wants an exhaustive shader-only bottleneck sweep (runtime ALU/varying/branch cost AND FXC/ANGLE compile shape), honouring the refuted-lever record (VS no-loss frontier, FS source-shrink dead, atlas faceting) so no dead lever is re-chased.',
  phases: [
    { title: 'Constraints', detail: 'P5 physics-first: fix the hard priors as data -- VS-bound (96%), FD 3-tap irreducible at no-loss, FXC uOctMax/fdIters invariants, refuted-lever list' },
    { title: 'Research',    detail: 'P7: web agents gather external evidence (FXC codegen pitfalls, ANGLE translation cost, snoise cost models, varying/interpolator pressure) -- evidence feeds finders, never ships unverified' },
    { title: 'Find',        detail: 'fan-out finders per shader surface, each blind to the others, each handed the constraints + research digest' },
    { title: 'Verify',      detail: 'P6 adversarial: every candidate refuted by 3 lenses (correctness/fidelity, FXC-invariant, actually-measurable); 2-of-3 survive threshold' },
    { title: 'Rank',        detail: 'P2+P9+P12 arbitration into ONE ranked plan with per-cut witness' },
  ],
}

// ---------------------------------------------------------------------------
// The refuted record (P7: measured-dead, finders must NOT propose these):
//  - VS octave reduction below 12 / wider FD step / analytic-derivative normal / full-height
//    atlas: ALL refuted 2026-06-10 (tv8-VS-perf-frontier record) -- fidelity loss measured.
//  - FS source-shrink for cold compile: measured-dead (cost = FXC optimize, not chars).
//  - per-tile/screen-space octave fade: seams at shared tile edges.
//  - constant-bound fractal loops / differencing composeHeight across call sites: FXC
//    mis-translation roots (uOctMax + fdIters fixes, commits f062365 + d56a202) -- INVARIANT.
//  - CPU-frame alloc/upload churn: 96% GPU-bound, refuted wb08pmga5.
// What HAS CHANGED since those records (fresh ground the finders target):
//  - 4770801 moved the domain warp to the VS (vTexWarp varying, -9 snoise3/px) and halved freqs.
//  - d56a202 rewrote the FD normal as a runtime-bounded fdIters loop (3x height in ONE loop).
//  - The FS still carries ~17 snoise3/px in places (rockface fBm, mottle, overlay) -- FS was
//    "not the bottleneck" pre-warp-move; the VS/FS split MUST be re-measured post-change.
// ---------------------------------------------------------------------------

const A = (typeof args === 'object' && args) ? args : {}
const measured = A.measured || null   // fresh gpuTimer split {fullMs, vsRasterMs, fsMs} per rung

const DNA = [
  'P2 subtractive: prefer REMOVING a tap/branch/varying outright over speeding it',
  'P5 physics-first: name the bound (ALU, interpolator count, register pressure, FXC optimize time)',
  'P7 empirical: every claim needs a measurable witness; UNMEASURED is a flag, not a guess',
  'P9 worst-case: the deck (close-approach) rung and the d3d11 cold path outrank averages',
  'P12 human-value: per-frame stutter > once-per-machine compile',
]

const CONSTRAINTS = [
  'VS-bound: ~96% of frame GPU is VS+raster at the deck (measured). The lit-normal FD 3-tap through the fdIters loop is the irreducible VS cost at zero fidelity loss (frontier record 2026-06-10) -- do not re-propose octave cuts, wider FD, analytic normals, or full-height atlas.',
  'FXC invariants (d3d11 default Chrome): uOctMax + fdIters loops stay runtime-bounded; never difference composeHeight across call sites; any NEW VS fractal loop gets a runtime bound.',
  'highp islands on every planet-scale quantity stay; global mediump elsewhere.',
  'broadShapeM stays a pure LOD-invariant world-dir fn (per-tile octave fade = seams, refuted).',
  'FS source-shrink is dead as a compile lever; cold d3d11 compile is 31s (corrected figure) and acceptable.',
  'Each terrain.glsl edit = one cold FXC compile for d3d11 users; BATCH shader edits into one reload; dev loop is vulkan (140ms).',
  'Verification surface: warm-tab /cmd + __diag (gpuTimer, bisect, groundTruth, seamProbe, verify.mjs suite). gpuTimer through the spool-browser verb TIMES OUT -- use /cmd.',
]

// Shader surfaces, each a finder lens. Fresh-ground emphasis per the changed-since record.
const SURFACES = [
  { key: 'vs-height',     lens: 'terrain.glsl VS height path: vtxDisplace, broadShapeM call count, the fdIters FD loop shape (is any height work duplicated outside it? is the water uIsWater gate still dead-code-free?), carve fns, anchor cbias. Count snoise3 evaluations per vertex on the worst path.' },
  { key: 'vs-varyings',   lens: 'VS->FS interface: count every varying (incl. new vTexWarp), their precision, packing opportunities (vec2 pairs into vec4), interpolator pressure on mobile-class GPUs, and any varying computed but consumed only on a gated FS path.' },
  { key: 'fs-material',   lens: 'terrain.glsl FS material/splat: per-pixel snoise3 census post-4770801 (rockface 3-oct fBm, mottle, detail overlay, strata), texture tap count per layer (4 layers x albedo+normal A/B), texIdent path, slope/rock gates -- which taps are live on FLAT ground (the common case, P9)?' },
  { key: 'fs-lighting',   lens: 'terrain.glsl FS lighting: detail-normal biplanar RNM taps, slope/gorge AO, aerial perspective single-scatter, haze/overlay fades -- distance/pxWorld gating completeness: does every expensive term fade out where invisible?' },
  { key: 'fs-water',      lens: 'water shading path: per-pixel cost, whether water pixels pay terrain-material costs before the water branch, ocean/lake/river branch structure under d3d11 (branch flattening = both sides paid).' },
  { key: 'compile-shape', lens: 'FXC compile-time shape: loop bounds, function inlining sites, anything that could regress the 31s cold figure or re-trigger per-callsite divergence; probe _PROBE_ variant sharing.' },
  { key: 'branch-divergence', lens: 'branch + precision audit: mediump leaks onto hot math, highp where mediump suffices on non-planet-scale values, branches FXC will flatten (both sides executed), redundant normalize/pow/exp on hot paths.' },
]

const FINDING_SCHEMA = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: { type: 'object', properties: {
      name: { type: 'string' }, file: { type: 'string' }, line: { type: 'string' },
      stage: { type: 'string', description: 'VS | FS | varying | compile' },
      perWhat: { type: 'string', description: 'per-vertex | per-pixel | per-frame | once-cold' },
      mechanism: { type: 'string', description: 'the physical cost mechanism (ALU taps, interpolators, branch flatten, FXC inline)' },
      cutIdea: { type: 'string' },
      estSaving: { type: 'string' },
      refutedLever: { type: 'boolean', description: 'true if this matches the refuted record -- finder must self-flag' },
      fidelityRisk: { type: 'string', description: 'none | low | medium | high + what could visibly change' },
      witness: { type: 'string', description: 'the measurable probe that proves the cut worked with no regression' },
    }, required: ['name', 'file', 'stage', 'mechanism', 'cutIdea', 'fidelityRisk', 'witness'] } },
  },
  required: ['findings'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    real: { type: 'boolean' }, safe: { type: 'boolean' }, reason: { type: 'string' },
  },
  required: ['real', 'safe', 'reason'],
}

// --- Constraints ---------------------------------------------------------------------------
phase('Constraints')
log('Constraints fixed (' + CONSTRAINTS.length + '); refuted levers will be self-flagged and dropped, not re-chased.')

// --- Research (web evidence, P7) -------------------------------------------------------------
phase('Research')
const RESEARCH_TOPICS = [
  'ANGLE D3D11 FXC HLSL codegen performance pitfalls for WebGL2 GLSL: loop unrolling, branch flattening, interpolator limits, dynamic indexing penalties. Cite sources.',
  'GPU vertex-shader optimization for procedural terrain: simplex noise ALU cost models, fewest-instruction snoise/simplex variants, vertex cache and varying-count costs on AMD RDNA and mobile GPUs. Cite sources.',
  'WebGL2 fragment shader cost on flattened branches and mediump vs highp throughput on AMD and mobile; best practice for distance-gated shading terms. Cite sources.',
]
const research = await parallel(RESEARCH_TOPICS.map((t, i) => () =>
  agent('Research via web search (WebSearch/WebFetch): ' + t + ' Return a compact digest of CLAIMS with source URLs; mark each claim verified-by-source vs speculative.',
    { label: 'research:' + i, phase: 'Research' })
))
const researchDigest = research.filter(Boolean).join('\n---\n').slice(0, 8000)
log('research digests gathered: ' + research.filter(Boolean).length + '/' + RESEARCH_TOPICS.length)

// --- Find (fan-out per surface) --------------------------------------------------------------
phase('Find')
const found = await pipeline(
  SURFACES,
  s => agent(
    'TV8 shader bottleneck finder, surface "' + s.key + '". Read src/shaders/terrain.glsl (and src/gl-render.js for uniform/varying wiring) and enumerate EVERY cost on this lens: ' + s.lens + '\n' +
    'Judge through the DNA lens:\n' + DNA.map(d => '  - ' + d).join('\n') + '\n' +
    'HARD CONSTRAINTS (violating proposals are auto-dead):\n' + CONSTRAINTS.map(c => '  - ' + c).join('\n') + '\n' +
    (measured ? 'MEASURED gpuTimer split: ' + JSON.stringify(measured) + '\n' : 'No fresh measurement passed -- flag every estSaving UNMEASURED.\n') +
    'External research digest (claims with sources; treat as hypotheses to check against the actual code):\n' + researchDigest + '\n' +
    'Return findings with file:line, the cost MECHANISM, a concrete cutIdea, fidelityRisk, and the witness probe. Self-flag refutedLever=true for anything matching the refuted record. Read-only pass; do not edit.',
    { label: 'find:' + s.key, phase: 'Find', schema: FINDING_SCHEMA }
  ),
  (res, s) => {
    const live = (res && res.findings || []).filter(f => !f.refutedLever)
    return parallel(live.map(f => () =>
      parallel(['fidelity-correctness', 'fxc-invariant', 'measurability'].map(lens => () =>
        agent(
          'Adversarially REFUTE this TV8 shader cut candidate via the ' + lens + ' lens. ' +
          (lens === 'fidelity-correctness' ? 'Could it visibly change terrain shape/material/lighting, seam at tile edges, or break LOD invariance? Read the actual code at the cited location.' :
           lens === 'fxc-invariant' ? 'Does it reintroduce a constant-bound fractal loop, difference composeHeight across call sites, drop a highp island, or otherwise risk the FXC d3d11 mis-translation class? Read the cited code.' :
           'Is the claimed saving actually measurable with the named witness, and plausibly non-noise given the VS-bound 96% prior? An FS-only cut at the deck is likely noise.') +
          ' Default real=false/safe=false if uncertain. Candidate:\n' + JSON.stringify(f, null, 2),
          { label: 'verify:' + f.name + ':' + lens, phase: 'Verify', schema: VERDICT_SCHEMA }
        )
      )).then(vs => {
        const ok = vs.filter(Boolean).filter(v => v.real && v.safe).length >= 2
        return { surface: s.key, finding: f, confirmed: ok, verdicts: vs }
      })
    ))
  }
)

const all = found.filter(Boolean).flat().filter(Boolean)
const confirmed = all.filter(x => x.confirmed)
const refuted = all.filter(x => !x.confirmed)
log('candidates: ' + all.length + ' -> confirmed ' + confirmed.length + ', refuted ' + refuted.length)

// --- Rank (single budget) --------------------------------------------------------------------
phase('Rank')
const ranking = confirmed.length ? await agent(
  'Rank these CONFIRMED TV8 shader cuts into ONE ordered plan (highest payoff first) through the DNA lens: ' +
  'P2 removal>speedup, P9 worst-rung (deck FPS, d3d11 path) first, P12 per-frame>once-cold, P7 measured>estimated. ' +
  'Batch note: all terrain.glsl edits ship as ONE reload. Confirmed cuts:\n' + JSON.stringify(confirmed.map(c => c.finding), null, 2),
  { label: 'rank', phase: 'Rank', schema: { type: 'object', properties: {
      ranked: { type: 'array', items: { type: 'object', properties: {
        rank: { type: 'number' }, name: { type: 'string' }, file: { type: 'string' },
        expectedSaving: { type: 'string' }, witness: { type: 'string' } },
        required: ['rank', 'name', 'file', 'witness'] } },
      rationale: { type: 'string' } }, required: ['ranked'] } }
) : { ranked: [], rationale: 'no confirmed cuts survived adversarial verify' }

return {
  workflow: 'shader-bottleneck-dna',
  constraints: CONSTRAINTS,
  researchSources: researchDigest.slice(0, 2000),
  candidates: all.length,
  confirmed: confirmed.map(c => ({ surface: c.surface, ...c.finding })),
  refuted: refuted.map(c => ({ surface: c.surface, name: c.finding.name, why: (c.verdicts || []).filter(v => v && !(v.real && v.safe)).map(v => v.reason).join(' | ') })),
  ranked: ranking.ranked,
  rationale: ranking.rationale,
  note: 'Apply ranked cuts as ONE batched terrain.glsl reload; witness each via /cmd warm-tab (compile clean, glError 0, gpuTimer delta, bisect/seamProbe no-regression, backend-ab d3d11 parity); then verify.mjs full suite.',
}
