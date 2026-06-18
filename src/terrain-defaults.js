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
  splitFactor: 0.30,        // blessed mesh density (was demo-pinned 0.28 then gen/tweak 0.30; 0.30 wins post-double-rAF)

  // ---- SHAPE (composeHeight; mirrored in SHAPE_UNIFORM_DEFAULTS for the CPU height path) ----
  landBias: -800.0,         // hypsometry bias (m); demo baked -800 (was SDK 0)
  detailOverlay: 50.0,      // perlin-everywhere relief term; demo baked 50 (was SDK 6)
  hiFreqCut: 0.95,          // fine-octave amplitude; demo baked 0.95 (was SDK 0.25)
  canyonDepth: 1.0,         // canyon-depth multiplier; demo baked 0 -> shader floors to 1.0 (was SDK 2.0)
  cliffAmt: 3.0,            // cliff/mesa terrace strength; demo baked 3.0 (was SDK 1.0)
  beachShelf: 0.0,          // land coastal shelf (m); 0 -> shader uses 600m guard (matches SDK)
  nrmStepM: 0.0,            // lit-normal FD step (m); 0 is the "use built-in 300m" sentinel (terrain.glsl:1043/1090)
  mtnBandWide: 0.1,         // mountain-belt width anchor step; demo baked 0.1 (was SDK 0)
  climateRelief: 0.65,      // climate-relief width; demo baked 0.65 (was SDK 0)
  isleWide: 1.0,            // island-zone width; demo baked 1.0 (was SDK 0)
  carveWide: 0.0,           // carve-climate width; matches SDK 0

  // ---- LOOK / LIGHTING ----
  exposure: 0.75,
  reliefShade: 1.8,
  skyFill: 0.45,
  variationAmt: 0.04,
  hazeMul: 0.65,
  diffWrap: 0.5,
  vertexAO: 1.0,
  aoAmt: 1.0,
  biomeWarp: 1.8,
  nightFloor: 0.4,
  termWidth: 0.05,
  terminatorGlow: 1.0,
  nightLights: 1.2,
  lookSat: 1.15,
  lookContrast: 1.12,
  flatNormal: 0.0,          // diagnostic only

  // ---- TEXTURE / SPLAT ----
  texMix: 1.0,
  texNrmK: 1.6,
  nrmLow: 0.5,
  triSharp: 10.0,
  texSat: 2.0,
  texBright: 0.8,
  biomeTint: 0.3,
  texWarp: 0.22,
  xSoft: 0.5,
  xFinger: 2.5,
  ordPush: 0.4,
  texPhoto: 0.0,
  texPhotoNear: 0.0,
  xFade0: 15000.0,
  xFade1: 40000.0,
  texFar0: 0.0,
  texFar1: 12000.0,
  nrmFade0: 45000.0,
  nrmFade1: 100000.0,

  // ---- BEACH / COAST ----
  beachTop: 210.0,
  beachWidth: 5.0,
  bandWarp: 1800.0,

  // ---- BIOME RAMP (read by gl-render C() from __gen.state.biome, else here) ----
  bcDeepSea:  [0.04, 0.10, 0.28],
  bcSea:      [0.10, 0.22, 0.42],
  bcShore:    [0.78, 0.72, 0.50],
  bcLowland:  [0.22, 0.34, 0.12],
  bcGrass:    [0.412, 0.416, 0.145],
  bcRock:     [0.52, 0.43, 0.34],   // fallback until the rock-photo mean loads (window.__surfRockMean)
  bcSnow:     [0.92, 0.94, 0.97],
  bandEdgesLo: [20.0, 250.0],
  bandEdgesHi: [3500.0, 6500.0],
  snowEdges:   [6000.0, 8500.0],
  seaDepthM:   3000.0,
  slopeRock:   [0.0, 0.22],

  // ---- OCEAN (Beer-Lambert; gl-render o3 reads window.__uOcean*, else here) ----
  uOceanDeep:    [0.008, 0.025, 0.06],
  uOceanShallow: [0.07, 0.22, 0.26],
  uOceanK:       [0.009, 0.004, 0.0018],

  // ---- NORMAL / GEOMETRY ----
  pvNormal: 1,
  fsNormal: 0,
  elevEdgeInset: 0.5,
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
