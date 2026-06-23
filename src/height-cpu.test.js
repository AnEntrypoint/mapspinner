import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createHeightSampler } from './height-cpu.js'
import * as g from './glsl-rt.js'

// Locks the transpiled CPU terrain height (height-gen.js, generated from
// terrain.glsl). Regression guard for the transpiler + anchor-field wiring.
// CORRECTNESS vs the live GPU surface is proven separately by pl-parity-test
// (browser, sampleGroundM). These tests lock determinism, finiteness, an
// Earth-like hypsometry, surfacePoint geometry, and golden samples.

const hs = createHeightSampler()

test('heightAt is finite + deterministic over a sphere grid; Earth-like range', () => {
  let minH = Infinity, maxH = -Infinity, land = 0, n = 0
  for (let i = 0; i < 400; i++) {
    const z = 1 - 2 * (i + 0.5) / 400, r = Math.sqrt(Math.max(0, 1 - z * z)), th = i * 2.399963
    const dir = [r * Math.cos(th), z, r * Math.sin(th)]
    const h = hs.heightAt(dir)
    assert.ok(Number.isFinite(h), `non-finite at ${dir}: ${h}`)
    assert.equal(hs.heightAt(dir), h, 'deterministic')
    if (h > 0) land++; n++
    if (h < minH) minH = h; if (h > maxH) maxH = h
  }
  assert.ok(minH < -100 && minH > -12000, `min ocean depth plausible: ${minH}`)   // deep ocean, above Mariana cap
  assert.ok(maxH > 1000 && maxH < 20000, `max peak plausible: ${maxH}`)            // real mountains, under ceiling
  assert.ok(land > 0 && land < n, `mixed land/sea (land=${land}/${n})`)
})

test('surfacePoint(dir) = normalize(dir) * (radius + heightAt)', () => {
  for (const dir of [[1, 0, 0], [0.3, 0.8, 0.5], [-0.5, 0.2, -0.84]]) {
    const d = g.normalize(dir)
    const h = hs.heightAt(d)
    const sp = hs.surfacePoint(dir)
    const expected = hs.radius + h
    assert.ok(Math.abs(Math.hypot(...sp) - expected) < 1e-3, `|surfacePoint| ${Math.hypot(...sp)} != R+h ${expected}`)
    // points along the same unit direction
    const u = g.normalize(sp)
    assert.ok(Math.abs(u[0] - d[0]) < 1e-9 && Math.abs(u[1] - d[1]) < 1e-9 && Math.abs(u[2] - d[2]) < 1e-9, 'colinear with dir')
  }
})

test('golden samples (regression lock; update only with an intended terrain.glsl change)', () => {
  // Captured from the current transpiled field. If terrain.glsl height changes,
  // re-run gen-height.mjs then update these to the new (parity-verified) values.
  // RE-BAKED 2026-06-18: HEIGHT_UNIFORM_DEFAULTS now carries the blessed SDK shape defaults
  // (terrain-defaults.js SHAPE_UNIFORM_DEFAULTS: landBias -800, detailOverlay 50, hiFreqCut 0.95,
  // canyonDepthMul 1.0, cliffAmt 3.0, mtnBandWide 0.1, climateRelief 0.65, isleWide 1.0) -- the
  // values the demo used to force via window.__ now live SDK-side, so the default CPU field shifted.
  // RE-BAKED 2026-06-23e: heightCurve=3, applied pre-scale on proland normalized output
  // using HPEAK=0.014 (measured land max). pow(clamp(h/HPEAK),curve)*HPEAK before *750000.
  const golden = [
    [[0.9, 0.1, 0.4],   98.7345],
    [[1, 0, 0],        -410.8732],
    [[0.577, 0.577, 0.577], -3200.5408],
  ]
  for (const [dir, exp] of golden) {
    const h = hs.heightAt(dir)
    assert.ok(Math.abs(h - exp) < 0.5, `heightAt(${dir}) = ${h.toFixed(2)}, golden ${exp}`)
  }
})
