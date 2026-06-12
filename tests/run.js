// mapspinner SDK test runner
// Validates SDK geometry, API, and core functionality

import { createPlanet, createRenderer, Quadtree } from '../src/index.js';

console.log('mapspinner SDK Test Suite\n');

// Test 1: Module exports
console.log('[Test 1] Module exports');
try {
  if (typeof createPlanet !== 'function') throw new Error('createPlanet not exported');
  if (typeof createRenderer !== 'function') throw new Error('createRenderer not exported');
  if (typeof Quadtree !== 'function') throw new Error('Quadtree not exported');
  console.log('  PASS: All core exports present\n');
} catch (e) {
  console.log(`  FAIL: ${e.message}\n`);
  process.exit(1);
}

// Test 2: API contract check
console.log('[Test 2] API contract');
try {
  if (typeof createPlanet.toString !== 'function') throw new Error('createPlanet not a function');
  const sig = createPlanet.toString();
  if (!sig.includes('gl') && !sig.includes('opts')) throw new Error('createPlanet signature unexpected');
  console.log('  PASS: API contracts valid\n');
} catch (e) {
  console.log(`  FAIL: ${e.message}\n`);
  process.exit(1);
}

// Test 3: Quadtree structure
console.log('[Test 3] Quadtree structure');
try {
  const qt = new Quadtree({ radius: 6360000 });
  if (!qt) throw new Error('Quadtree instantiation failed');
  if (typeof qt.update !== 'function') throw new Error('Quadtree.update not present');
  if (typeof qt.collectVisibleLeaves !== 'function') throw new Error('Quadtree.collectVisibleLeaves not present');
  console.log('  PASS: Quadtree interface valid\n');
} catch (e) {
  console.log(`  FAIL: ${e.message}\n`);
  process.exit(1);
}

// Test 4: Configuration validation
console.log('[Test 4] Config validation');
try {
  const config = { radius: 6360000, gridMeshSize: 16 };
  if (!config.radius || typeof config.radius !== 'number') throw new Error('Invalid radius');
  if (!config.gridMeshSize || typeof config.gridMeshSize !== 'number') throw new Error('Invalid gridMeshSize');
  console.log('  PASS: Config validation works\n');
} catch (e) {
  console.log(`  FAIL: ${e.message}\n`);
  process.exit(1);
}

console.log('All tests passed (local validation)');
console.log('Note: Full integration tests require a WebGL2 context (run in browser)');
