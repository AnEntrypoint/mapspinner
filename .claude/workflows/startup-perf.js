export const meta = {
  name: 'startup-perf',
  description: 'Find + attack mapspinner startup bottlenecks: map init cost, attribute by cold-compile proxy, cut, verify',
  whenToUse: 'When page startup/load time regresses, or before stepping up shader detail. Re-runs the measure->attribute->cut->verify loop.',
  phases: [
    { title: 'Map',       detail: 'parallel readers over gl-render.js / planet-orchestrator.js / terrain.glsl init paths' },
    { title: 'Attribute', detail: 'rank init sections by cold-compile cost proxy (translated-HLSL size, tap count, JS bake samples)' },
    { title: 'Cut',       detail: 'one reduction per ranked section (debug-split, probe-slim, octave-cut, bake-background)' },
    { title: 'Verify',    detail: 'per cut: lab gates + translated-size delta + node --check; render-run glError deferred to user reload' },
  ],
}

// ----------------------------------------------------------------------------
// mapspinner startup-perf workflow. The cold bottleneck is the ANGLE/HLSL translation of the giant terrain
// FS (~188s first-load on AMD; warm <0.1s, driver-cached). The driver cache is the only persistence
// (no getProgramBinary on ANGLE-AMD), so the only lever is the unrolled INSTRUCTION COUNT, proxied by
// WEBGL_debug_shaders getTranslatedShaderSource().length. This workflow maps init, ranks sections by
// that proxy, applies one cut per section, and verifies each. Cold seconds are unmeasurable while the
// driver cache is warm -> validation is the translated-size delta + the lab gates + node --check, with
// the live render glError + initTimings deferred to a user hard-reload (each shader edit = one fresh
// cold compile, so cuts are BATCHED into as few reloads as possible).
// ----------------------------------------------------------------------------

const INIT_FILES = [
  { path: 'src/gl-render.js',            lens: 'shader compile/link, program build, probe, parallel-compile, debug-program split' },
  { path: 'src/planet-orchestrator.js',  lens: 'HPF bake (HPF_RES^2*6 sampleUV), anchor field, init timings, background bake' },
  { path: 'src/shaders/terrain.glsl',    lens: 'broadShapeM octave count, debug displayMode blocks, biome/strata/water branches, _PROBE_ block' },
]

const SECTION_SCHEMA = {
  type: 'object',
  properties: {
    sections: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name:        { type: 'string' },
          file:        { type: 'string' },
          costKind:    { type: 'string', description: 'cold-compile | js-init | both' },
          proxyMetric: { type: 'string', description: 'translated-HLSL chars, snoise3 tap count, or sampleUV call count' },
          cutIdea:     { type: 'string' },
          regressionRisk: { type: 'string', description: 'low | medium | high, and what gate proves no-regression' },
        },
        required: ['name', 'file', 'costKind', 'cutIdea', 'regressionRisk'],
      },
    },
  },
  required: ['sections'],
}

phase('Map')
const maps = await parallel(INIT_FILES.map(f => () =>
  agent(
    `Read ${f.path} in the mapspinner repo and map every startup-init cost it carries (focus: ${f.lens}). ` +
    `For each cost section return name, file, costKind (cold-compile|js-init|both), proxyMetric, a concrete cutIdea, ` +
    `and regressionRisk with the gate that proves no-regression. Do NOT edit anything; this is a read+map pass.`,
    { label: `map:${f.path.split('/').pop()}`, phase: 'Map', schema: SECTION_SCHEMA }
  )
))
const sections = maps.filter(Boolean).flatMap(m => m.sections)

phase('Attribute')
const ranking = await agent(
  `Rank these mapspinner startup-init sections by how much they cost the COLD shader compile + JS init, using the ` +
  `cold-compile proxy = translated-HLSL instruction count (scales with unrolled snoise3 taps + branch count) ` +
  `and, for JS, the sampleUV call count. Cheapest-to-cut-with-highest-payoff first; flag any whose cut has ` +
  `medium/high regression risk so it is gated behind the lab. Sections:\n${JSON.stringify(sections, null, 2)}`,
  { label: 'attribute', phase: 'Attribute', schema: {
      type: 'object',
      properties: {
        ranked: { type: 'array', items: { type: 'object', properties: {
          name: { type: 'string' }, rank: { type: 'number' }, expectedSaving: { type: 'string' },
          gate: { type: 'string' } }, required: ['name', 'rank', 'expectedSaving', 'gate'] } },
        summary: { type: 'string' },
      }, required: ['ranked', 'summary'] } }
)

phase('Cut')
// Per ranked section, propose the concrete edit + the witness that proves it (translated-size delta /
// lab gate). Worktree isolation so parallel cut proposals do not collide. This workflow PROPOSES +
// verifies cuts; the human applies + batches the terrain.glsl edits into one reload (cold-compile cost).
const cuts = await pipeline(
  ranking.ranked.sort((a, b) => a.rank - b.rank),
  sec => agent(
    `Propose the concrete code edit for mapspinner startup section "${sec.name}" (expected saving ${sec.expectedSaving}). ` +
    `Give the exact file, the old snippet, the new snippet, and the WITNESS that proves it (translated-HLSL ` +
    `size delta via WEBGL_debug_shaders, OR a lab gate maxElev/landFrac/hypsometry unchanged, OR node --check). ` +
    `Gate: ${sec.gate}. Do not break the lit/shape path; if risk is high, say so and require the lab gate.`,
    { label: `cut:${sec.name}`, phase: 'Cut', schema: {
        type: 'object',
        properties: {
          file: { type: 'string' }, oldSnippet: { type: 'string' }, newSnippet: { type: 'string' },
          witness: { type: 'string' }, safe: { type: 'boolean' }, notes: { type: 'string' },
        }, required: ['file', 'witness', 'safe'] } }
  ),
  (proposal, sec) => agent(
    `Adversarially verify this mapspinner startup cut proposal for "${sec.name}". Could it change the rendered terrain ` +
    `(shape/biome/lighting), break a uniform, or fail to compile? Default to safe=false if uncertain. ` +
    `Proposal:\n${JSON.stringify(proposal, null, 2)}`,
    { label: `verify:${sec.name}`, phase: 'Verify', schema: {
        type: 'object',
        properties: { real: { type: 'boolean' }, safe: { type: 'boolean' }, reason: { type: 'string' } },
        required: ['real', 'safe', 'reason'] } }
  ).then(v => ({ section: sec.name, proposal, verdict: v }))
)

const safe = cuts.filter(Boolean).filter(c => c.proposal.safe && c.verdict.safe)
const risky = cuts.filter(Boolean).filter(c => !(c.proposal.safe && c.verdict.safe))
return {
  attribution: ranking.summary,
  ranked: ranking.ranked,
  safeCuts: safe.map(c => ({ section: c.section, file: c.proposal.file, witness: c.proposal.witness })),
  gatedCuts: risky.map(c => ({ section: c.section, reason: c.verdict.reason })),
  note: 'Batch all terrain.glsl edits into ONE reload (each shader edit = a fresh cold compile). ' +
        'Validate via translated-size delta + lab gates; live glError/initTimings via a user hard-reload.',
}
