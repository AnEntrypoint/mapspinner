export const meta = {
  name: 'optimize-dna',
  description: 'Maximize whole-project optimization through the 12-principle synthesized-engineering-DNA lens. Fans out per-axis finders (correctness, speed, simplification/maintenance-burden, architecture-pliability, jank/tell-tale-AI), adversarially verifies each finding (is it real? does fixing it regress another axis?), then arbitrates ALL findings across axes into ONE DNA-ranked plan. Composes the speed-dna workflow for the speed axis rather than re-implementing it.',
  whenToUse: 'When the user wants a holistic optimization pass over the WHOLE project (not just speed) -- correctness/bugs, runtime/compile speed, simplification + maintenance-burden reduction, architecture pliability, and machine-shaped jank -- ranked and arbitrated as one budget through the compound DNA lens. The cross-axis arbitration is the capability no single skill has: code-review finds bugs, simplify finds cleanups, speed-dna finds speed levers, but nothing weighs a simplification that costs perf against a perf cut that adds risk.',
  phases: [
    { title: 'Constraints', detail: 'P5 physics-first: fix the hard project constraints + the measured-dead record (FS source-shrink dead, atlas is the only compile lever, broadShapeM LOD-invariant/no-seam, 141s/shader-edit) so no finder proposes refuted or physics-fighting work' },
    { title: 'Map',         detail: 'one reader per optimization AXIS maps that axis surface (correctness, speed, simplification, architecture, jank) tagged with the DNA principle that judges it' },
    { title: 'Find',        detail: 'parallel finders per axis surface findings (a bug, a perf lever, a dedupe/simplification, a bespoke->native swap that NET-SHRINKS, a tell-tale-AI shape)' },
    { title: 'Verify',      detail: 'P6 adversarial: each finding gets refuted -- is it REAL, and does fixing it REGRESS another axis (a simplification that costs perf, a perf cut that adds risk, a dep that net-grows surface)?' },
    { title: 'Arbitrate',   detail: 'P11 crucible / P2+P9+P12: rank ALL surviving cross-axis findings into ONE budget by DNA-principle order weighted by felt human-value; resolve every cross-axis conflict explicitly' },
    { title: 'Plan',        detail: 'P10 honest interface: return the DNA-justified, witness-bearing, cross-axis-arbitrated optimization plan, each item with the no-regression witness and the principle that ranked it' },
  ],
}

// ----------------------------------------------------------------------------
// optimize-dna -- the whole-project optimization workflow. It does NOT duplicate the existing
// single-concern tooling (P2 subtractive / P4 composition spine): the SPEED axis is delegated to
// the speed-dna workflow (which already arbitrates compile-vs-fps); code-review / simplify already
// own bug-hunting and cleanup. What this workflow ADDS is the layer above them: a CROSS-AXIS
// arbitration that ranks a bug fix, a perf lever, a simplification, an architecture swap, and a
// jank cleanup against EACH OTHER in one budget, by DNA-principle order, weighted by felt impact.
//
// The load-bearing priors every finder must respect (P7 measure-don't-assume, P3 revert-and-don't-
// rechase):
//   - COLD-COMPILE: FS source-shrink is MEASURED-DEAD; the only real compile lever is atlas/THC
//     height-bake (removes the per-vertex fractal from the VS). FPS prime lever = the VS 3x-fractal
//     /vertex + LOD quad count. broadShapeM MUST stay a pure LOD-invariant world-dir fn (per-tile/
//     screen-space octave fade was TRIED + REFUTED = seams).
//   - Each terrain.glsl edit = one ~141s cold ANGLE compile -> a finder that proposes a shader micro-
//     edit must justify it against that cost (P5/P12: a change a user never feels but costs 141s to
//     ship is negative value).
//   - Net-smaller-surface rule (P2): a bespoke->library/native swap is only a win if it NET-SHRINKS
//     the shipped+maintained surface; adding a heavy dep to delete a few lines is the failure mode.
//
// args (all optional): { axes: <string[] subset of the 5 axis keys to run; default all>,
//                        runSpeedChild: <bool, run the speed-dna child for the speed axis; default false
//                          since it needs headed-browser numbers -- when false the speed finder reasons
//                          from the static map + the measured priors>,
//                        speedArgs: <forwarded to the speed-dna child: {measured, compile}> }
// ----------------------------------------------------------------------------

const A = (typeof args === 'object' && args) ? args : {}

// The 12 principles as the ranking lens handed to every finder + the arbiter (P1 data-first: kept as
// data, never branched on).
const DNA = [
  'P1 data-first: is this a state/data-model problem (fix the model, not the control flow around it)?',
  'P2 subtractive: can the thing be REMOVED rather than improved? does a fix NET-SHRINK the surface?',
  'P3 evolutionary: ship the simplest thing that works; revert-first on regressions.',
  'P4 composition-spine: does each module do one thing, understandable at its call site?',
  'P5 physics-first: what hardware/driver constraint bounds this (ANGLE compile, ALU, bandwidth, vert count)?',
  'P6 adversarial: is misuse structurally possible? make the wrong thing hard.',
  'P7 empirical: is there a MEASURED number, or a guess? refuse measured-dead levers.',
  'P8 automated-correctness: can a guardrail (lint/type/pure-fn/gate) prevent this class of bug?',
  'P9 worst-case: does it help the WORST case (close-approach fps / cold first-load), not just average?',
  'P10 honest-interface: does any public contract / default behaviour lie about what it guarantees?',
  'P11 crucible: does the hardest integration (close-approach, cold load, real input) still hold?',
  'P12 human-value: does a real user FEEL this? trace the impact chain to a human outcome.',
]

// The optimization AXES. Each is one finder surface; speed is delegated to the speed-dna child.
const AXES = [
  { key: 'correctness', principles: 'P6/P8/P9/P11',
    prompt: 'Hunt CORRECTNESS issues: real bugs, unguarded invalid states, worst-case failure modes (close-approach, cold load, degenerate input), silent-catastrophic paths, missing guardrails. For each: the bug, the file:line, the failure trigger, and the structural guardrail (P8) that would prevent the class.' },
  { key: 'speed', principles: 'P5/P7/P9', delegate: 'speed-dna',
    prompt: 'Map the SPEED levers (cold compile + runtime fps) WITHOUT re-deriving the speed-dna analysis. Respect the measured-dead record (FS source-shrink dead; atlas is the only compile lever; VS triple-fractal + LOD-quad-count are the fps levers). Surface only the top levers + their expected saving + the no-seam/LOD-invariance gate.' },
  { key: 'simplification', principles: 'P2/P4',
    prompt: 'Hunt SIMPLIFICATION + maintenance-burden cuts: duplicated logic (drift-prone reimplementations like a second copy of a mirror), dead code, abstractions that cost more than they deliver, config options that could be a default, kitchen-sink modules. For each: what to REMOVE/merge, the net-lines-saved, and the risk witness (lint/gate that proves no behaviour change).' },
  { key: 'architecture', principles: 'P1/P4/P10',
    prompt: 'Hunt ARCHITECTURE-pliability improvements: a bad data model the code works around (P1), a bespoke reimplementation a popular well-maintained library/native API replaces with a NET-SMALLER surface (P2 -- reject if the dep net-grows surface), a layer boundary that is bypassed, a dishonest default/interface (P10). For each: the change, the net-surface delta, and why it is clearly outstanding.' },
  { key: 'jank', principles: 'P12 + gm-discipline',
    prompt: 'Hunt JANK + tell-tale-AI shapes: unfinished edges, half-wired paths, immaturity across gui/ux/client-state/server-state and the client/server boundary; AND machine-authored tells (boilerplate flourishes, over-hedged comments, generic scaffold names, decorative non-ASCII glyphs that should be ASCII). For each: the surface, what unfinished/machine-shaped thing it is, and the polish/convert fix.' },
]

const FINDING_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title:        { type: 'string' },
          axis:         { type: 'string', description: 'correctness | speed | simplification | architecture | jank' },
          file:         { type: 'string', description: 'file:line if known' },
          what:         { type: 'string', description: 'the concrete issue + the concrete fix' },
          dnaPrinciple: { type: 'string', description: 'the principle that most justifies acting (e.g. P2 subtractive)' },
          surfaceDelta: { type: 'string', description: 'net lines/files/deps added or removed by the fix (P2 net-shrink test)' },
          crossAxisRisk:{ type: 'string', description: 'does the fix regress another axis? (simplify costs perf, perf cut adds risk, dep net-grows surface)' },
          measuredDead: { type: 'boolean', description: 'true if this is a known refuted/measured-dead lever (e.g. FS source-shrink for cold compile)' },
          felt:         { type: 'string', description: 'P12: the human outcome a real user feels, or "internal only"' },
        },
        required: ['title', 'axis', 'what', 'dnaPrinciple'],
      },
    },
  },
  required: ['findings'],
}

// --- Constraints (P5 physics-first) ---------------------------------------------------------
phase('Constraints')
const CONSTRAINTS = [
  'COLD COMPILE: FS source-shrink is MEASURED-DEAD (ANGLE DCEs unused funcs; cost is intrinsic HLSL translation of the FS/VS main). The only real compile lever is the atlas/THC height-bake (removes the per-vertex broadShapeM fractal from the VS). Driver-cached: warm 41ms, cold ~141s, recurs on shader SOURCE change.',
  'FPS: the VS runs 3x full 12-oct broadShapeM/vertex (displace + 2 central-diff taps); behind-limb cull removes ~50-80% leaves; frustum cull intentionally OFF.',
  'broadShapeM MUST stay a pure LOD-invariant world-dir fn: a per-tile/screen-space octave fade was TRIED + REFUTED (seams at shared tile edges).',
  'Each terrain.glsl edit = one ~141s cold compile -> a shader micro-edit a user never feels is negative value (P5/P12). BATCH shader edits.',
  'NET-SMALLER-SURFACE (P2): a bespoke->library/native swap is a win ONLY if it net-shrinks the shipped+maintained surface; a heavy dep to delete a few lines net-grows it.',
  'There is ONE canonical broadShapeM Node mirror (terrain-lab-shape.js, 2026-06-09 consolidation); a finder proposing a second copy is reintroducing refuted drift.',
]
log('Constraints fixed (' + CONSTRAINTS.length + '); measured-dead + refuted levers will be refused.')

// --- Map + Find (fan-out, one finder per axis) ----------------------------------------------
phase('Map')
const selected = Array.isArray(A.axes) && A.axes.length ? AXES.filter(x => A.axes.includes(x.key)) : AXES
log('running ' + selected.length + ' optimization axes: ' + selected.map(x => x.key).join(', '))

phase('Find')
const perAxis = await parallel(selected.map(ax => async () => {
  // The SPEED axis is delegated to the speed-dna workflow (compose, do not re-implement).
  if (ax.delegate === 'speed-dna' && A.runSpeedChild) {
    try {
      const speed = await workflow('speed-dna', A.speedArgs || {})
      const ranked = (speed && speed.ranked) || []
      return { findings: ranked.map(r => ({
        title: 'speed: ' + (r.name || r.lever || 'lever'), axis: 'speed', file: speed.topAxis || '',
        what: (r.expectedSaving ? ('expected ' + r.expectedSaving + '. ') : '') + (speed.rationale || ''),
        dnaPrinciple: r.dnaPrinciple || 'P5', surfaceDelta: '', crossAxisRisk: (r.gate || ''),
        measuredDead: false, felt: 'frame rate / load time',
      })) }
    } catch (e) {
      log('speed-dna child failed (' + String(e.message || e) + '); falling back to static speed map')
    }
  }
  return agent(
    ax.prompt + '\n' +
    'Judge each finding through the synthesized-engineering-DNA lens (this axis leans on ' + ax.principles + '):\n' +
    DNA.map(d => '  - ' + d).join('\n') + '\n' +
    'HARD CONSTRAINTS (a finding that violates one is invalid -- flag measuredDead=true):\n' +
    CONSTRAINTS.map(c => '  - ' + c).join('\n') + '\n' +
    'Return findings with title, axis="' + ax.key + '", file:line, what (issue + concrete fix), dnaPrinciple, surfaceDelta (P2 net-shrink), crossAxisRisk, measuredDead, felt (P12). Read-only; do NOT edit.',
    { label: 'find:' + ax.key, phase: 'Find', schema: FINDING_SCHEMA }
  )
}))
const found = perAxis.filter(Boolean).flatMap(r => (r && r.findings) || [])
const live  = found.filter(f => !f.measuredDead)
const dead  = found.filter(f => f.measuredDead)
log('found ' + found.length + ' (' + live.length + ' live, ' + dead.length + ' measured-dead dropped)')

// --- Verify (P6 adversarial: is it real, does it regress another axis?) ----------------------
phase('Verify')
const verified = await pipeline(
  live,
  f => agent(
    'Adversarially verify this mapspinner optimization finding. (1) Is it REAL -- not a false positive, not already handled, not a measured-dead lever (FS source-shrink for cold compile, a per-tile broadShapeM fade, a second mirror copy)? (2) Does fixing it REGRESS another optimization axis -- a simplification that costs perf, a perf cut that adds correctness risk or a seam, a dep that net-GROWS the surface, an architecture change that breaks a public default? Default real=false / regresses=true if uncertain. Finding:\n' + JSON.stringify(f, null, 2),
    { label: 'verify:' + (f.axis || '?') + ':' + (f.title || '').slice(0, 24), phase: 'Verify', schema: {
        type: 'object',
        properties: { real: { type: 'boolean' }, regressesOtherAxis: { type: 'boolean' }, reason: { type: 'string' } },
        required: ['real', 'regressesOtherAxis', 'reason'] } }
  ).then(v => ({ ...f, verdict: v }))
)
const confirmed = verified.filter(Boolean).filter(f => f.verdict && f.verdict.real && !f.verdict.regressesOtherAxis)
const gated     = verified.filter(Boolean).filter(f => !(f.verdict && f.verdict.real && !f.verdict.regressesOtherAxis))
log('confirmed ' + confirmed.length + '/' + live.length + ' after adversarial verify')

// --- Arbitrate (P11 crucible / P2+P9+P12: ONE budget across all axes) ------------------------
phase('Arbitrate')
const ranking = confirmed.length ? await agent(
  'You are arbitrating the mapspinner whole-project optimization budget across ALL axes (correctness, speed, simplification, architecture, jank) as ONE ranked plan, through the synthesized-engineering-DNA lens. Rank the CONFIRMED findings below highest-payoff first, applying the principle order P1>P2>...>P12 (earlier principles win conflicts) weighted by P12 human-value (a thing a user FEELS continuously outranks an internal nicety). ' +
  'Resolve every CROSS-AXIS conflict explicitly: when two findings touch the same code or trade against each other (a simplification vs a perf lever, a correctness guardrail vs subtractive removal), state which wins and cite the principle. Prefer the change that REMOVES surface (P2) and helps the WORST case (P9). Confirmed findings:\n' + JSON.stringify(confirmed, null, 2),
  { label: 'arbitrate', phase: 'Arbitrate', schema: {
      type: 'object',
      properties: {
        topItem: { type: 'string', description: 'the single highest-payoff optimization to do first' },
        rationale: { type: 'string', description: 'why first, in DNA-principle terms' },
        crossAxisConflicts: { type: 'array', items: { type: 'object', properties: {
          conflict: { type: 'string' }, winner: { type: 'string' }, principle: { type: 'string' } },
          required: ['conflict', 'winner', 'principle'] } },
        ranked: { type: 'array', items: { type: 'object', properties: {
          title: { type: 'string' }, axis: { type: 'string' }, rank: { type: 'number' },
          dnaPrinciple: { type: 'string' }, expectedValue: { type: 'string' }, gate: { type: 'string' } },
          required: ['title', 'axis', 'rank', 'expectedValue'] } },
      }, required: ['topItem', 'ranked'] } }
) : { topItem: null, rationale: 'no confirmed findings survived adversarial verify', crossAxisConflicts: [], ranked: [] }

// --- Plan (P10 honest interface) ------------------------------------------------------------
phase('Plan')
return {
  workflow: 'optimize-dna',
  constraints: CONSTRAINTS,
  axesRun: selected.map(x => x.key),
  measuredDeadDropped: dead.map(f => ({ title: f.title, axis: f.axis, why: f.what })),
  topItem: ranking.topItem,
  rationale: ranking.rationale,
  crossAxisConflicts: ranking.crossAxisConflicts || [],
  ranked: ranking.ranked,
  gatedFindings: gated.map(f => ({ title: f.title, axis: f.axis, reason: f.verdict && f.verdict.reason })),
  note: 'Apply ranked items highest-first. Each is cross-axis-verified (real + does not regress another axis). ' +
        'Speed items: re-measure on a HEADED browser; broadShapeM stays LOD-invariant; FS source-shrink for cold compile is measured-dead. ' +
        'Architecture/dep swaps must NET-SHRINK the surface. BATCH terrain.glsl edits (each = ~141s cold compile). ' +
        'Run the speed-dna workflow directly for a deep speed pass; this workflow is the cross-axis layer above it.',
}
