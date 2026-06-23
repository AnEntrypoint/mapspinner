export const meta = {
  name: 'cleanup',
  description: 'Project cleanup: remove dead code, test consolidation, stale scripts and artifacts',
  phases: [
    { title: 'Scan', detail: 'discover dead code, test structure, artifact locations' },
    { title: 'Fix', detail: 'remove/consolidate and apply fixes in parallel' },
    { title: 'Verify', detail: 'validate cleanup did not break builds/tests' },
  ],
}

// Phase 1: Scan for cleanup opportunities in parallel
phase('Scan')

const SCANS = [
  { label: 'dead-code', prompt: 'Search src/ for commented-out code, // TODO/FIXME blocks, and unused function definitions. Return a list of files and line ranges with dead code.' },
  { label: 'test-files', prompt: 'Analyze src/*.test.js and tests/run.js files. Identify redundant test files, duplicate test cases, and consolidation opportunities.' },
  { label: 'scripts', prompt: 'Check scripts/ directory for outdated or unused scripts (verify.mjs, backend-ab.mjs, ab-elevation.mjs, etc). Determine which are referenced in package.json and which are dead.' },
  { label: 'artifacts', prompt: 'Identify removable artifacts: planet.zip, bash.exe.stackdump, lab-out/ directory, and other build byproducts. Check .gitignore coverage.' },
]

const scanResults = await parallel(SCANS.map(s => agent(s.prompt, { label: `scan:${s.label}`, phase: 'Scan' })))

log(`Scan discovered ${scanResults.filter(Boolean).length} cleanup surfaces`)

// Phase 2: Propose fixes for each scan surface
phase('Fix')

const validScans = scanResults.filter(Boolean)
const fixes = validScans.map((result, idx) => {
  const scan = SCANS[idx % SCANS.length]
  return agent(
    `Given this scan result:\n${result}\n\nPropose concrete fixes (file edits, deletions, or consolidations). Be specific about what to change and why. Format: list of actions with file:line references.`,
    { label: `fix:${scan.label}`, phase: 'Fix' }
  )
})

const fixResults = await parallel(fixes)

log(`Fixes proposed for all surfaces`)

// Phase 3: Verify proposal feasibility
phase('Verify')

const verify = await agent(
  `Review the proposed cleanup fixes. Identify any that are:
   - High-risk (could break imports or build)
   - Already applied (don't duplicate)
   - Out of scope or contradictory

   Return a feasibility report: which fixes are safe to apply, which need caution, which should be skipped.`,
  { label: 'verify:feasibility', phase: 'Verify' }
)

return {
  summary: 'Cleanup analysis: discovered cleanup opportunities, proposed fixes, reviewed feasibility',
  scanCount: scanResults.filter(Boolean).length,
  fixCount: fixResults.filter(Boolean).length,
  feasibilityReport: verify,
}
