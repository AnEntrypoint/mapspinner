// src/ab-elevation.test.js -- A/B isolation: assert every ELEVATION term actually affects the height
// field (toggle WITH vs WITHOUT, require a real delta). Answers "do uniforms affect elevation?" as a
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

test('uLandBias shifts elevation everywhere', () => {
  // landBias is the primary lever in composeHeight (h = fractalTerrainH * 750000 + uLandBias).
  // Shifting it by 50000 must change elevation by at least that much somewhere.
  assert.ok(maxDelta({ uLandBias: -100000 + 50000 }) > 1000, 'uLandBias must shift elevation')
})

test('uBeachShelfM affects low-land elevation', () => {
  // beach shelf smoothing applies to h in [0, bShelf]; toggling shelf size must change heights.
  assert.ok(maxDelta({ uBeachShelfM: 50000 }) > 1, 'uBeachShelfM must affect coastal/land heights')
})

test('broadShapeLowM high octaves are elevation-IRRELEVANT (mesa slope gate only -- documented)', () => {
  // uBroadLowOcts feeds ONLY the 2400m-step mesa-flatness SLOPE gate (gl-render.js:467); the high octaves
  // average out over that step -> 0 elevation effect. This locks that contract (and the CPU mirror's 2-octave
  // default). If this ever changes, broadShapeLowM became an elevation term and the dedup/perf notes are stale.
  assert.ok(maxDelta({ uBroadLowOcts: 8 }) < 1, 'uBroadLowOcts must NOT change elevation (slope-gate-only)')
})
