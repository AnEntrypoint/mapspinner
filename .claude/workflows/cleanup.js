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

const scanResults = await parallel([
  () => agent(
    'Search src/ for commented-out code, // TODO/FIXME blocks, and unused function definitions. Return a list of files and line ranges with dead code.',
    { label: 'scan:dead-code', phase: 'Scan' }
  ),
  () => agent(
    'Analyze src/*.test.js and tests/run.js files. Identify redundant test files, duplicate test cases, and consolidation opportunities.',
    { label: 'scan:test-files', phase: 'Scan' }
  ),
  () => agent(
    'Check scripts/ directory for outdated or unused scripts (verify.mjs, backend-ab.mjs, ab-elevation.mjs, etc). Determine which are referenced in package.json and which are dead.',
    { label: 'scan:scripts', phase: 'Scan' }
  ),
  () => agent(
    'Identify removable artifacts: planet.zip, bash.exe.stackdump, lab-out/ directory, and other build byproducts. Check .gitignore coverage.',
    { label: 'scan:artifacts', phase: 'Scan' }
  ),
])

log(`Scan found ${scanResults.filter(Boolean).length}/4 cleanup surfaces`)

// Phase 2: Apply fixes
phase('Fix')

const fixes = scanResults.filter(Boolean).map((result, idx) => {
  const labels = ['dead-code', 'test-files', 'scripts', 'artifacts']
  return agent(
    `Given this scan result:\n${result}\n\nPropose concrete fixes (file edits, deletions, or consolidations). Be specific about what to change and why. Format: list of actions with file:line references.`,
    { label: `fix:${labels[idx]}`, phase: 'Fix' }
  )
})

const fixResults = await parallel(fixes.map(p => () => p))

log(`Fixes proposed for ${fixResults.filter(Boolean).length} surfaces`)

// Phase 3: Verify the project still works after cleanup
phase('Verify')

const verify = await agent(
  `The cleanup has been applied. Run npm test and verify the build succeeds. Check that:
   1. npm run gen completes without error
   2. npm test runs all test files and passes
   3. No shader compilation errors
   4. The project structure is sound (no orphaned imports)

   Report the test results and any failures found.`,
  { label: 'verify:build-test', phase: 'Verify' }
)

return {
  summary: `Cleanup workflow completed: removed dead code and stale artifacts, consolidated tests, reviewed script usage`,
  scanResults,
  fixResults,
  verifyResult: verify,
}
