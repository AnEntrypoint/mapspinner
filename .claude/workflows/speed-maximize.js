export const meta = {
  name: 'speed-maximize',
  description: 'Comprehensive speed optimization: baseline → hotspot detection → parallel finder agents → ranked lever application → final report',
  phases: [
    { title: 'Baseline & Hotspot', detail: 'Measure perf, locate dominant bottleneck' },
    { title: 'Parallel Finders', detail: '7 subagents scan VS/FS/memory/JS/shader/LOD/startup in parallel' },
    { title: 'Apply & Verify', detail: 'Apply top levers, remeasure, halt on diminishing returns' },
    { title: 'Final Report', detail: 'Ranked action plan: what to ship vs defer' }
  ]
}

phase('Baseline & Hotspot')

const baseline = await agent(
  'Measure baseline: launch planet.html, capture shaderCompileMs cold + warm, gpuTimer VS vs FS at a fixed pose (deck, 1km altitude), quad count, and memory peak. Return {shaderCompileMs_cold, shaderCompileMs_warm, frameMs, vsMs, fsMs, quads, memoryMB, pose}.',
  {
    label: 'baseline-measure',
    phase: 'Baseline & Hotspot',
    schema: {
      type: 'object',
      properties: {
        shaderCompileMs_cold: { type: 'number', description: 'Cold compile time (shader cache cleared)' },
        shaderCompileMs_warm: { type: 'number', description: 'Warm compile time (cached)' },
        frameMs: { type: 'number', description: 'Total frame time at baseline pose' },
        vsMs: { type: 'number', description: 'Vertex shader time via gpuTimer' },
        fsMs: { type: 'number', description: 'Fragment shader time via gpuTimer' },
        quads: { type: 'number', description: 'On-screen quad count' },
        memoryMB: { type: 'number', description: 'GPU memory peak' },
        pose: { type: 'string', description: 'Camera position (deck/altitude km, facing direction)' }
      },
      required: ['shaderCompileMs_cold', 'frameMs', 'vsMs', 'fsMs', 'quads']
    }
  }
)

log(`Baseline: ${baseline.frameMs.toFixed(1)}ms frame (VS ${baseline.vsMs.toFixed(1)}ms ${(100*baseline.vsMs/baseline.frameMs).toFixed(0)}%, FS ${baseline.fsMs.toFixed(1)}ms), compile ${baseline.shaderCompileMs_cold}ms cold`)

const hotspot = await agent(
  `Diagnose dominant bottleneck. Given baseline: VS=${baseline.vsMs}ms (${(100*baseline.vsMs/baseline.frameMs).toFixed(0)}%), FS=${baseline.fsMs}ms, compile=${baseline.shaderCompileMs_cold}ms cold, quads=${baseline.quads}. Return {bottleneck: 'VS'|'FS'|'compile'|'memory', recommendation: string, rationale: string}. If VS>80% frame, bottleneck='VS' + recommend GRID/splitFactor/octave cuts. If FS>30% + VS<70%, bottleneck='FS' + recommend detail-normal/snoise3 cuts. If compile>30s, bottleneck='compile' + recommend profiling. If memory>500MB, bottleneck='memory'.`,
  {
    label: 'hotspot-locate',
    phase: 'Baseline & Hotspot',
    schema: {
      type: 'object',
      properties: {
        bottleneck: { enum: ['VS', 'FS', 'compile', 'memory'] },
        recommendation: { type: 'string' },
        rationale: { type: 'string' }
      },
      required: ['bottleneck', 'recommendation']
    }
  }
)

log(`Hotspot: ${hotspot.bottleneck} (${hotspot.recommendation})`)

phase('Parallel Finders')

const findings = await parallel([
  () => agent(
    'VS-reduction finder: If VS is dominant, measure GRID 16→12, splitFactor 1.4→1.2, broadShapeFD octave 12→10, vtxDisplace 6→4. For each candidate: apply, measure frame time + quad count, revert. Return [{name, vsMs_before, vsMs_after, quads_before, quads_after, speedup_pct, tradeoff}]. Order by speedup/quality ratio.',
    { label: 'vs-levers', phase: 'Parallel Finders', effort: 'high' }
  ),
  () => agent(
    'FS-reduction finder: If FS is significant, profile detail-normal snoise3 taps, test gating to near-field only, evaluate AO reduction. For each: measure uFsCheap on/off. Return [{name, fsMs_before, fsMs_after, speedup_pct, visual_impact}].',
    { label: 'fs-levers', phase: 'Parallel Finders', effort: 'high' }
  ),
  () => agent(
    'Memory profiler: Measure GPU memory for height-pool (if enabled), HPF quadtree baked state, geometry cache, anchorField. For each: quantify MB, propose reduction (e.g. halve pool res, tighten HPF octaves). Return [{component, memoryMB, reduction_MB, method}].',
    { label: 'memory-levers', phase: 'Parallel Finders', effort: 'medium' }
  ),
  () => agent(
    'JS hot-loop profiler: Run node --prof planet-orchestrator.js with a full camera pan, identify top 3 JS bottlenecks (quadtree walk, mesh gen, camera update), propose caching/lowering. Return [{function, selfMs, cumulativeMs, proposal, estimated_speedup_pct}].',
    { label: 'js-levers', phase: 'Parallel Finders', effort: 'high' }
  ),
  () => agent(
    'Shader micro-opt finder: Review terrain.glsl for dead branches, unroll traps (AGENTS.md line 203-224), expensive ops outside loops, texture-fetch order. For each: test compile time + perf impact. Return [{optimization, compile_benefit_pct, runtime_benefit_pct, risk}].',
    { label: 'shader-levers', phase: 'Parallel Finders', effort: 'high' }
  ),
  () => agent(
    'LOD tightening finder: Test splitFactor peak 1.4→1.2, near-field radius tightening, screen-cull margins. For each altitude (deck/1km/10km): measure quads, verify detail monotone growth on descent. Return [{lever, quads_reduction_pct, visual_impact, altitude_effects}].',
    { label: 'lod-levers', phase: 'Parallel Finders', effort: 'high' }
  ),
  () => agent(
    'Startup perf finder: Profile shader compile + module load + initialization. Use insights from /startup-perf skill if invoked, else measure cold-compile under cache-clear. Return [{stage, ms, bottleneck, proposal}].',
    { label: 'startup-levers', phase: 'Parallel Finders', effort: 'medium' }
  )
])

const allFindings = findings.filter(Boolean).flat()
log(`${allFindings.length} findings collected across 7 agents`)

phase('Apply & Verify')

const applied = []
let iteration = 0
const maxIterations = 5
const diminishingReturnThreshold = 0.05

while (iteration < maxIterations && allFindings.length > applied.length) {
  iteration++
  const remaining = allFindings.filter(f => !applied.find(a => a.name === f.name))
  if (!remaining.length) break

  const topFinding = remaining.sort((a, b) => (b.speedup_pct || 0) - (a.speedup_pct || 0))[0]
  if (!topFinding || (topFinding.speedup_pct && topFinding.speedup_pct < diminishingReturnThreshold * 100)) {
    log(`Diminishing returns: top finding is ${topFinding?.speedup_pct?.toFixed(1)}%, stopping`)
    break
  }

  log(`Iteration ${iteration}: applying ${topFinding.name}`)

  const verified = await agent(
    `Apply lever: ${topFinding.name}. Measure before/after. Verify visual equivalence: check mesh wireframe (quads sane), relief SD (loss <10%), normal continuity, screenshot at oblique. If regression: return {applied: false, reason, recommendation}. Else: return {applied: true, frameMs_after, vsMs_after, fsMs_after, visual_check: 'pass'}.`,
    {
      label: `apply-${iteration}`,
      phase: 'Apply & Verify',
      schema: {
        type: 'object',
        properties: {
          applied: { type: 'boolean' },
          frameMs_after: { type: 'number' },
          speedup_pct: { type: 'number' },
          visual_check: { enum: ['pass', 'regress'] },
          reason: { type: 'string' }
        }
      }
    }
  )

  if (verified.applied) {
    applied.push({ ...topFinding, verified })
    log(`✓ Applied: speedup ${verified.speedup_pct?.toFixed(1)}%`)
  } else {
    log(`✗ Rejected: ${verified.reason}`)
  }
}

phase('Final Report')

const report = await agent(
  `Synthesize speed audit. Inputs: baseline frame ${baseline.frameMs}ms (VS ${baseline.vsMs}ms, FS ${baseline.fsMs}ms, compile ${baseline.shaderCompileMs_cold}ms), hotspot=${hotspot.bottleneck}, applied=${applied.length} levers (${applied.map(a => a.name).join(', ')}). Cumulative speedup from baseline: ${applied.reduce((acc, a) => acc * (1 - (a.verified?.speedup_pct || 0) / 100), 1)}. Return one-page report: {baseline_metrics: {...}, applied_levers: [{name, speedup_pct, tradeoffs}], cumulative_speedup_pct, top_5_recommendations, next_tier_opportunities, ship_recommendation: 'immediate'|'defer'}.`,
  {
    label: 'final-report',
    phase: 'Final Report',
    schema: {
      type: 'object',
      properties: {
        baseline_metrics: { type: 'object' },
        applied_levers: { type: 'array' },
        cumulative_speedup_pct: { type: 'number' },
        top_5_recommendations: { type: 'array' },
        ship_recommendation: { enum: ['immediate', 'defer'] }
      },
      required: ['cumulative_speedup_pct', 'ship_recommendation']
    }
  }
)

return {
  baseline,
  hotspot,
  findings: allFindings,
  applied,
  report
}
