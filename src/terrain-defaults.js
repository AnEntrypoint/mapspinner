// terrain-defaults.js -- the SINGLE SOURCE OF TRUTH for every terrain look/shape/lod default.
//
// WHY THIS EXISTS: the calibrated "blessed" look used to live in the DEMO (tweak-panel.js
// applyBaked() force-set ~50 window.__ globals on boot; terrain-gen-controls.js force-applied
// biome/look/lod; planet.html pinned __splitFactor). A bare SDK consumer that did NOT load those
// demo panels got gl-render's old fallback defaults instead -> a DIFFERENT planet. Per the SDK
// Validation Policy + the user directive "none of our setup is in the demo, all of it sits on the
// SDK side so the demo just uses the SDK with default settings", the calibrated values now live
// HERE and the SDK render layers (gl-render.js, planet-orchestrator.js, height-cpu.js) read them as
// their fallback defaults. The demo panels are pure LIVE overlays: they import these for slider
// display + reset and only write a window.__ global when the user moves a control.
//
// KEYS = the window.__<key> suffix the gl-render g()/_g()/o3()/C() helpers look up (so a call site
// becomes `_g('exposure', TERRAIN_DEFAULTS.exposure)`). Vec3 colour/ocean keys carry the values as
// 3-arrays. SHAPE_UNIFORM_DEFAULTS re-keys the geometry levers by their GLSL uniform name for the
// height-cpu.js CPU mirror (which must match the GPU _PROBE_ exactly or the parity gate breaks).

export const TERRAIN_DEFAULTS = {
  // ---- LOD ----
  splitFactor: 0.30,        // blessed mesh density

  // ---- SHAPE (composeHeight; mirrored in SHAPE_UNIFORM_DEFAULTS for the CPU height path) ----
  landBias: -126450.0,      // bias for ~40% land at scale 750000 with the domain-warp FBM
  detailOverlay: 53.0,      // perlin-everywhere relief term
  hiFreqCut: 1.0,           // fine-octave amplitude
  canyonDepth: 40.0,        // canyon-depth multiplier
  cliffAmt: 5.0,            // cliff/mesa terrace strength
  beachShelf: 0.0,          // land coastal shelf (m); 0 -> shader uses 600m guard
  nrmStepM: 200.0,          // lit-normal FD step (m)
  mtnBandWide: 1.0,         // mountain-belt width anchor step
  climateRelief: 1.0,       // climate-relief width
  isleWide: 0.55,           // island-zone width
  carveWide: 0.0,           // carve-climate width

  // ---- LOOK / LIGHTING ----
  exposure: 0.85,
  reliefShade: 6.0,
  skyFill: 0.3,
  variationAmt: 0.05,
  hazeMul: 0.4,
  vertexAO: 0.0,
  aoAmt: 0.0,
  biomeWarp: 1.1,
  nightFloor: 0.18,
  termWidth: 0.45,
  terminatorGlow: 0.6,
  nightLights: 0.8,
  lookSat: 1.0,
  lookContrast: 1.0,
  flatNormal: 0.0,          // diagnostic only

  // ---- TEXTURE / SPLAT ----
  texMix: 1.0,
  texNrmK: 0.4,
  nrmLow: 0.4,
  triSharp: 3.0,
  texSat: 1.0,
  texBright: 1.0,
  biomeTint: 0.62,
  texWarp: 1.0,
  xSoft: 0.3,
  xFinger: 1.6,
  ordPush: 0.4,
  texPhoto: 0.0,
  texPhotoNear: 0.0,
  xFade0: 100.0,
  xFade1: 340.0,
  texFar0: 0.0,
  texFar1: 10000.0,
  nrmFade0: 100.0,
  nrmFade1: 1000.0,
  octFar0: 5.0,
  octFar1: 50.0,

  // ---- BEACH / COAST ----
  beachTop: 15.0,
  beachWidth: 1.0,
  bandWarp: 20.0,

  // ---- BIOME RAMP (read by gl-render C() from __gen.state.biome, else here) ----
  bcDeepSea:  [0.02, 0.06, 0.22],
  bcSea:      [0.08, 0.18, 0.38],
  bcShore:    [0.82, 0.76, 0.52],
  bcLowland:  [0.18, 0.38, 0.10],
  bcGrass:    [0.30, 0.46, 0.12],
  bcRock:     [0.48, 0.40, 0.30],   // fallback until the rock-photo mean loads (window.__surfRockMean)
  bcSnow:     [0.94, 0.96, 1.00],
  bandEdgesLo: [8.0, 20.0],
  bandEdgesHi: [60.0, 100.0],
  snowEdges:   [120.0, 180.0],
  seaDepthM:   580.0,
  slopeRock:   [0.25, 0.55],

  // ---- OCEAN (Beer-Lambert; gl-render o3 reads window.__uOcean*, else here) ----
  uOceanDeep:    [0.008, 0.025, 0.06],
  uOceanShallow: [0.07, 0.22, 0.26],
  uOceanK:       [0.009, 0.004, 0.0018],

};

// SHAPE levers re-keyed by GLSL uniform name for the height-cpu.js CPU mirror. canyonDepth maps to
// canyonDepthMul; the demo's __canyonDepth=0 floors to 1.0 in the shader, so the CPU default is 1.0
// directly. nrmStepM is omitted (not a height term). beachShelf 0 -> shader 600m guard (same on CPU).
export const SHAPE_UNIFORM_DEFAULTS = {
  uLandBias:      TERRAIN_DEFAULTS.landBias,
  uBeachShelfM:   TERRAIN_DEFAULTS.beachShelf,
  canyonDepthMul: TERRAIN_DEFAULTS.canyonDepth,
  uDetailOverlay: TERRAIN_DEFAULTS.detailOverlay,
  uHiFreqCut:     TERRAIN_DEFAULTS.hiFreqCut,
  uCarveWide:     TERRAIN_DEFAULTS.carveWide,
  uMtnBandWide:   TERRAIN_DEFAULTS.mtnBandWide,
  uClimateRelief: TERRAIN_DEFAULTS.climateRelief,
  uIsleWide:      TERRAIN_DEFAULTS.isleWide,
  cliffAmt:       TERRAIN_DEFAULTS.cliffAmt,
};

export default TERRAIN_DEFAULTS;
