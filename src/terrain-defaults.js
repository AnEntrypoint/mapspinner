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
  landBias: -252900.0,      // bias for ~40% land at scale 1500000 with the domain-warp FBM
  detailOverlay: 53.0,      // perlin-everywhere relief term
  hiFreqCut: 1.0,           // fine-octave amplitude
  canyonDepth: 40.0,        // canyon-depth multiplier
  cliffAmt: 5.0,            // cliff/mesa terrace strength
  beachShelf: 0.0,          // land coastal shelf (m); 0 -> shader uses 600m guard
  nrmStepM: 700.0,          // lit-normal FD step (m)
  mtnBandWide: 1.0,         // mountain-belt width anchor step
  climateRelief: 1.0,       // climate-relief width
  isleWide: 0.55,           // island-zone width
  carveWide: 0.0,           // carve-climate width

  // ---- LOOK / LIGHTING ----
  exposure: 0.85,
  reliefShade: 2.5,
  skyFill: 0.5,
  variationAmt: 0.3,
  hazeMul: 0.15,
  diffWrap: 0.45,
  vertexAO: 0.0,
  aoAmt: 0.0,
  biomeWarp: 1.1,
  nightFloor: 0.18,
  termWidth: 0.45,
  terminatorGlow: 0.6,
  nightLights: 0.8,
  lookSat: 0.7,
  lookContrast: 1.12,
  flatNormal: 0.0,          // diagnostic only

  // ---- TEXTURE / SPLAT ----
  texMix: 0.65,
  texNrmK: 0.7,
  nrmLow: 1.9,
  triSharp: 1.0,
  texSat: 0.8,
  texBright: 0.64,
  biomeTint: 0.56,
  texWarp: 0.32,
  xSoft: 0.3,
  xFinger: 1.6,
  ordPush: 0.4,
  texPhoto: 0.0,
  texPhotoNear: 0.0,
  xFade0: 100.0,
  xFade1: 340.0,
  texFar0: 0.0,
  texFar1: 700.0,
  nrmFade0: 100000.0,
  nrmFade1: 2000000.0,

  // ---- BEACH / COAST ----
  beachTop: 29000.0,
  beachWidth: 2000.0,
  bandWarp: 200000.0,

  // ---- BIOME RAMP (read by gl-render C() from __gen.state.biome, else here) ----
  bcDeepSea:  [0.04, 0.10, 0.28],
  bcSea:      [0.10, 0.22, 0.42],
  bcShore:    [0.78, 0.72, 0.50],
  bcLowland:  [0.22, 0.34, 0.12],
  bcGrass:    [0.412, 0.416, 0.145],
  bcRock:     [0.52, 0.43, 0.34],   // fallback until the rock-photo mean loads (window.__surfRockMean)
  bcSnow:     [0.92, 0.94, 0.97],
  bandEdgesLo: [2000.0, 25000.0],
  bandEdgesHi: [350000.0, 650000.0],
  snowEdges:   [600000.0, 850000.0],
  seaDepthM:   300000.0,
  slopeRock:   [0.0, 0.22],

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
