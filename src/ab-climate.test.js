// src/ab-climate.test.js -- the BIOME driver: assert the anchor-field climate (temp/humid) actually
// VARIES across the sphere and resolves into multiple distinct classes (so biomes/rivers/canyons/sand
// have something to gate on). A constant climate = one biome everywhere = the failure this catches.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createHeightSampler } from './height-cpu.js'

const af = createHeightSampler({ radius: 6360000 }).anchorField

test('climate temp + humidity span a wide range (biome driver is not constant)', () => {
  let tmin = 9, tmax = -9, hmin = 9, hmax = -9
  const N = 4000
  const classes = new Set()
  for (let i = 0; i < N; i++) {
    const y = 1 - (i + 0.5) / N * 2, r = Math.sqrt(Math.max(0, 1 - y * y)), th = i * 2.399963229728653
    const d = [r * Math.cos(th), y, r * Math.sin(th)]
    const { face, fu, fv } = af.dirToFaceUV(d)
    const c = af.sampleUV(face, fu, fv, 7)
    tmin = Math.min(tmin, c.temp); tmax = Math.max(tmax, c.temp)
    hmin = Math.min(hmin, c.humidity); hmax = Math.max(hmax, c.humidity)
    const cl = (c.temp < 0.3 ? 'cold' : c.temp > 0.6 ? 'hot' : 'temp') + '/' + (c.humidity < 0.33 ? 'dry' : c.humidity > 0.66 ? 'wet' : 'mid')
    classes.add(cl)
  }
  assert.ok(tmax - tmin > 0.7, `temp must span a wide range, got ${tmin.toFixed(2)}..${tmax.toFixed(2)}`)
  assert.ok(hmax - hmin > 0.7, `humidity must span a wide range, got ${hmin.toFixed(2)}..${hmax.toFixed(2)}`)
  assert.ok(classes.size >= 6, `climate must resolve into many distinct biome classes, got ${classes.size}`)
})
