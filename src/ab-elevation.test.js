// src/ab-elevation.test.js -- A/B isolation: assert every ELEVATION term actually affects the height
// field (toggle WITH vs WITHOUT, require a real delta). Answers "do canyons affect elevation?" as a
// standing check, and catches any term that silently goes no-op. (Full report: node scripts/ab-elevation.mjs)

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createHeightSampler } from './height-cpu.js'

const base = createHeightSampler({ radius: 6360000 })
// max |heightAt(default) - heightAt(override)| over a fibonacci sphere -- the term's elevation footprint.
function maxDelta(override, N = 2500) {
  const alt = createHeightSampler({ radius: 6360000, uniforms: override })
  let m = 0
  for (let i = 0; i < N; i++) {
    const y = 1 - (i + 0.5) / N * 2, r = Math.sqrt(Math.max(0, 1 - y * y)), th = i * 2.399963229728653
    const d = [r * Math.cos(th), y, r * Math.sin(th)]
    m = Math.max(m, Math.abs(base.heightAt(d) - alt.heightAt(d)))
  }
  return m
}

test('CANYONS actually affect elevation', () => {
  // The SDK default canyonDepthMul is now 1.0 (terrain-defaults.js; the demo's blessed look). The
  // shader floors <=0 to 1.0, so toggling to 0 would be a no-op vs the default -- compare against a
  // DEEPER depth (3x) instead: a real delta proves canyons carve the mesh.
  assert.ok(maxDelta({ canyonDepthMul: 3 }) > 50, 'canyonDepthMul must change elevation -- canyons carve the mesh')
})

test('every elevation term has a real, isolatable effect', () => {
  assert.ok(maxDelta({ uDetailOverlay: 0 }) > 50, 'detail overlay + flat-area valleys')
  assert.ok(maxDelta({ cliffAmt: 0.01 }) > 50, 'cliff/mesa terracing')
  assert.ok(maxDelta({ uPeakOcts: 1 }) > 50, 'broadShapeM peak crest octaves')
  assert.ok(maxDelta({ uInciseRidgeOcts: 1 }) > 50, 'incise/valley ridge octaves')
  assert.ok(maxDelta({ uDetailFbmOcts: 1 }) > 5, 'detail fbm octaves')
  assert.ok(maxDelta({ uOctMax: 1 }) > 50, 'broadShapeM master octave bound')
})

test('broadShapeLowM high octaves are elevation-IRRELEVANT (mesa slope gate only -- documented)', () => {
  // uBroadLowOcts feeds ONLY the 2400m-step mesa-flatness SLOPE gate (gl-render.js:467); the high octaves
  // average out over that step -> 0 elevation effect. This locks that contract (and the CPU mirror's 2-octave
  // default). If this ever changes, broadShapeLowM became an elevation term and the dedup/perf notes are stale.
  assert.ok(maxDelta({ uBroadLowOcts: 8 }) < 1, 'uBroadLowOcts must NOT change elevation (slope-gate-only)')
})
