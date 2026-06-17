// height-cpu.js -- pure-JS terrain HEIGHT sampler for headless consumers (e.g. a
// game server building a physics collider). It runs the GLSL composeHeight subset
// transpiled to JS (height-gen.js, generated from terrain.glsl -- the single
// source of truth) over the pure-JS anchor-field (the HPF), so the CPU surface
// matches the GPU-rendered surface. THREE-FREE, no DOM/GL/Node globals.
//
// Parity contract: the uniform DEFAULTS below mirror gl-render.js
// setComposeHeightUniforms / the render uniform set; the HPF is sampled with the
// SAME maxBandLevel the renderer bakes (log2(hpfTexRes)). scripts/gen-height.mjs
// must be re-run after any terrain.glsl height edit; pl-parity-test validates CPU
// == GPU sampleGroundM and fails loudly on drift.

import { makeHeight } from './height-gen.js'
import { createAnchorField } from './anchor-field.js'
import * as g from './glsl-rt.js'

// Renderer uniform defaults -- MUST mirror gl-render.js (setComposeHeightUniforms
// line ~450 + the render set line ~1089). Override per-field via opts.uniforms to
// match a host that dials window.__* levers.
export const HEIGHT_UNIFORM_DEFAULTS = {
  hasHpf: 1,
  uLandBias: 0.0,
  uBeachShelfM: 0.0,        // 0 -> the GLSL guard uses 600m (wide beach)
  canyonDepthMul: 2.0,
  uDetailOverlay: 6.0,
  uHiFreqCut: 0.25,
  uCarveWide: 0.0, uMtnBandWide: 0.0, uClimateRelief: 0.0, uIsleWide: 0.0,
  uVsCheap: 0.0,
  cliffAmt: 1.0,
  vtxDetail: 1.0,           // vtxDisplace returns 0 regardless; kept for completeness
  uOctMax: 12, uInciseRidgeOcts: 4, uBroadLowOcts: 8, uPeakOcts: 3,
  uDetailFbmOcts: 3, uVtxBaseOcts: 6, uVtxErodeOcts: 4,
}

export function createHeightSampler(opts = {}) {
  const radius = opts.radius || 6360000
  const hpfTexRes = opts.hpfTexRes || 128
  const BAKE_MAX_LEVEL = Math.round(Math.log2(hpfTexRes))   // matches planet-orchestrator bake
  const af = opts.anchorField || createAnchorField({ seed: opts.seed })
  const U = { ...HEIGHT_UNIFORM_DEFAULTS, defRadius: radius, ...(opts.uniforms || {}) }

  // --- HPF bake (CPU mirror of planet-orchestrator bakeFaceRows + the terrain.glsl
  // hpfSample bilinear). The GPU samples a per-face RES^2 baked texture with a
  // QUINTIC-C2 bilinear, NOT the continuous field; sampling the continuous field
  // diverged from the rendered surface by up to ~640m (HPF texel interpolation).
  // We bake the SAME grid (inset map fu=x/(RES-1), the renderer default) into a flat
  // [seaBias,elevAmp,temp,humid] per texel, then replicate the inset quintic bilinear
  // -> the CPU continental bias matches the GPU to float precision. Lazy per face
  // (only faces the caller actually samples are baked) so a local play patch pays for
  // ~1 face, not all 6.
  const RES = hpfTexRes
  const _faceBuf = new Array(6).fill(null)
  function _bakeFace(face) {
    const buf = new Float32Array(RES * RES * 4)
    for (let y = 0; y < RES; y++) {
      const fv = y / (RES - 1)                          // inset (matches renderer default _hpfInset=true)
      for (let x = 0; x < RES; x++) {
        const fu = x / (RES - 1)
        const s = af.sampleUV(face, fu, fv, BAKE_MAX_LEVEL)
        const o = (y * RES + x) * 4
        buf[o] = s.seaBias; buf[o + 1] = s.elevAmp; buf[o + 2] = s.temp; buf[o + 3] = s.humidity
      }
    }
    _faceBuf[face] = buf
    return buf
  }
  const _quintic = (t) => t * t * t * (t * (t * 6 - 15) + 10)
  // hpfSample(dir) = the GPU hpfSample: inset quintic-bilinear of the baked face grid.
  const hpfSample = (dir) => {
    const d = g.normalize(dir)
    const { face, fu, fv } = af.dirToFaceUV(d)
    const buf = _faceBuf[face] || _bakeFace(face)
    const denom = RES - 1
    const tx = fu * denom, ty = fv * denom
    let x0 = Math.floor(tx), y0 = Math.floor(ty)
    const wx = _quintic(tx - x0), wy = _quintic(ty - y0)
    if (x0 < 0) x0 = 0; else if (x0 > denom) x0 = denom
    if (y0 < 0) y0 = 0; else if (y0 > denom) y0 = denom
    const x1 = x0 < denom ? x0 + 1 : denom, y1 = y0 < denom ? y0 + 1 : denom
    const out = [0, 0, 0, 0]
    for (let c = 0; c < 4; c++) {
      const o00 = ((y0 * RES + x0) * 4) + c, o10 = ((y0 * RES + x1) * 4) + c
      const o01 = ((y1 * RES + x0) * 4) + c, o11 = ((y1 * RES + x1) * 4) + c
      const a = buf[o00] + wx * (buf[o10] - buf[o00])
      const b = buf[o01] + wx * (buf[o11] - buf[o01])
      out[c] = a + wy * (b - a)
    }
    return out   // [seaBias, elevAmp, temp, humid]
  }
  const H = makeHeight(U, hpfSample)

  // Rendered terrain elevation (metres, signed) at a world DIRECTION (need not be unit).
  // faceLocal/tileM are inert (vtxDisplace returns 0), passed as dummies.
  function heightAt(dir) {
    const d = g.normalize(dir)
    const C = H.computeHCache(d)
    return H.composeHeightC(d, [0, 0], 100, C, false)
  }
  // World-space surface point for a direction: dir * (radius + height).
  function surfacePoint(dir) { const d = g.normalize(dir); return g.mul(d, radius + heightAt(d)) }

  return { heightAt, surfacePoint, radius, anchorField: af, uniforms: U, _fns: H }
}
