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
  assert.ok(minH > -360000, `min depth above cap: ${minH}`)   // above -350000 raw seabed floor (= -350m after reliefScale)
  assert.ok(maxH < 500000, `max height under ceiling: ${maxH}`)  // under ceiling (fractal * 750000 scale)
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
  // RE-BAKED 2026-06-24: seabed = h*1.25 capped at -350000 raw (=-350m after reliefScale).
  //
  // The golden locks the RAW transpiled field (the thing gen-height.mjs produces from
  // terrain.glsl) -- so it must sample with reliefScale=1, NOT the runtime default
  // (radius/63600000), which is a per-consumer DISPLAY scale, not a property of the
  // transpiled height. The default-reliefScale sampler `hs` multiplies heightAt by ~1e-4
  // (radius 6360 / 63600000), which would make this transpiler-regression lock track a
  // render knob instead of terrain.glsl (it caused a false 10x/10000x golden break when
  // commit 1ce6ea3 changed the default reliefScale without an intended terrain.glsl edit).
  const goldenSampler = createHeightSampler({ reliefScale: 1 })
  const golden = [
    [[0.9, 0.1, 0.4],  45022.357],
    [[1, 0, 0],        -38974.597],
    [[0.577, 0.577, 0.577], -132804.879],
  ]
  for (const [dir, exp] of golden) {
    const h = goldenSampler.heightAt(dir)
    assert.ok(Math.abs(h - exp) < 0.5, `heightAt(${dir}) = ${h.toFixed(2)}, golden ${exp}`)
  }
})
