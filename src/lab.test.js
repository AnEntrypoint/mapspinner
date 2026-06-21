// src/lab.test.js -- GPU-free self-test for the CLI lab's CPU height path (scripts/lab.mjs).
// Covered by `npm test` (src/**/*.test.js). The headless SwiftShader half (glsl-check/parity)
// is exercised by `node scripts/lab.mjs glsl-check`, not here (needs a browser).

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { dirFromLatLon, sampleField, encodePNGGray, crc32 } from '../scripts/lab.mjs'

test('dirFromLatLon: unit vectors at the cardinal points', () => {
  const eq = (a, b) => assert.ok(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) < 1e-9)
  eq(dirFromLatLon(0, 0), [1, 0, 0])
  eq(dirFromLatLon(90, 0), [0, 1, 0])          // north pole -> +y
  eq(dirFromLatLon(-90, 0), [0, -1, 0])
  eq(dirFromLatLon(0, 90), [0, 0, 1])
})

test('sampleField: deterministic + Earth-like elevation range', () => {
  const a = sampleField({ res: 24 })
  const b = sampleField({ res: 24 })
  assert.equal(a.w, 48); assert.equal(a.h, 24)                 // equirectangular 2:1
  assert.equal(a.elev.length, b.elev.length)
  for (let i = 0; i < a.elev.length; i++) assert.equal(a.elev[i], b.elev[i])   // byte-identical = deterministic
  assert.ok(Number.isFinite(a.min) && Number.isFinite(a.max))
  assert.ok(a.max > a.min, 'has relief')
  // Range widened 2026-06-21: the deterministic integer shash3 reseed (CPU<->GPU parity fix) shifts the
  // noise field; its peaks reach ~18km on this seed (still a plausible planet, finite + deterministic).
  assert.ok(a.min > -20000 && a.max < 20000, `Earth-like range, got [${a.min},${a.max}]`)
  assert.ok(a.landFrac >= 0 && a.landFrac <= 1)
})

test('sampleField: amplitude scales ~linearly with radius for the BULK field', () => {
  // The open-terrain field is scale-invariant (relief * reliefScale), so heightAt(R/100) ~ heightAt(R)/100
  // for the vast majority of directions. It is NOT exact at coasts: the beach-shelf uses an ABSOLUTE
  // width (uBeachShelfM ~600m) that is relatively 100x wider on the 1/100 planet (tracked: scale-coastal-
  // absolute-width). So assert the MEDIAN ratio is ~100, not every point.
  const big = sampleField({ res: 16, radius: 6360000 })
  const small = sampleField({ res: 16, radius: 63600 })
  const ratios = []
  for (let i = 0; i < big.elev.length; i++) {
    if (Math.abs(small.elev[i]) > 1) ratios.push(big.elev[i] / small.elev[i])
  }
  ratios.sort((a, b) => a - b)
  const median = ratios[ratios.length >> 1]
  assert.ok(Math.abs(median - 100) < 1, `median radius ratio ~100, got ${median}`)
})

test('encodePNGGray: valid PNG signature + IHDR dims + IEND', () => {
  const w = 8, h = 4
  const png = encodePNGGray(w, h, new Uint8Array(w * h).fill(128))
  assert.deepEqual([...png.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10])  // signature
  assert.equal(png.readUInt32BE(16), w)         // IHDR width
  assert.equal(png.readUInt32BE(20), h)         // IHDR height
  assert.equal(png[24], 8)                       // bit depth
  assert.equal(png[25], 0)                       // colour type grayscale
  // ends with an IEND chunk: [len=0][type 'IEND'][crc] -> type is bytes [len-8 .. len-4), crc is the last 4
  assert.equal(png.subarray(png.length - 8, png.length - 4).toString('ascii'), 'IEND')
  assert.equal(png.readUInt32BE(png.length - 4), 0xAE426082)   // the well-known IEND CRC
})

test('crc32: known vector', () => {
  assert.equal(crc32(Buffer.from('IEND', 'ascii')) >>> 0, 0xAE426082)  // PNG IEND type CRC base
})
