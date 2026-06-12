// terrain-gen-controls.js -- the ONE live control surface for terrain generation params.
// Imported by planet.html; installs window.__gen. No rebuild needed: shader params are window.*
// globals the render reads per-frame; HPF band scales go through the anchor field; biome colors
// are read live by gl-render.
//
// After the C++/wasm + cascade/producer deletion (GPU one-fractal), the old elevation noiseAmp
// cascade + material ortho-noise tables are GONE: they fed the deleted wasm PL.setNoiseAmp/
// setSeaBias/setOrthoNoiseAmp setters, which no longer exist. Only the params below still drive
// anything, so the panel shows only live knobs.
//
// __gen.defaults  - canonical default for every field (exact reset).
// __gen.state     - current values (mutated by sliders / set()).
// __gen.set(path,v)- set one field by dot-path, re-apply.
// __gen.apply()   - push the whole state to shader globals + HPF band scales.
// __gen.get()     - read back the live shader globals.
// __gen.reset()   - restore defaults and apply.
// __gen.serialize()/load(json) - persist/restore.

const ORCH = () => window.__planetOrch || null;
const HPF = () => window.__hpf || null;

// ---- canonical defaults (only the params that still drive the GPU one-fractal) ----
const DEFAULTS = {
  normal: {
    pvNormal: 1,           // window.__pvNormal (per-vertex seamless normal, default on)
    fsNormal: 0,           // window.__fsNormal (cross-dFdx FS normal, diagnostic, default off)
  },
  lod: {
    splitFactor: null,     // window.__splitFactor (null = orchestrator altitude ramp)
  },
  geometry: {
    elevEdgeInset: 0.5,    // window.__elevEdgeInset (mesh-edge elevation sample inset; gl-render)
  },
  // biome ramp colors + band edges (gl-render reads window.__gen.state.biome live).
  biome: {
    bcDeepSea: [0.04,0.10,0.28], bcSea: [0.10,0.22,0.42], bcShore: [0.78,0.72,0.50],
    bcLowland: [0.20,0.34,0.15], bcGrass: [0.26,0.40,0.17], bcRock: [0.52,0.43,0.34],   // bcRock -> a clearly WARMER TAN-GREY (R>G>B) so it reads as ROCK when lit, not olive-green (the [0.46,0.42,0.37] near-grey read green next to the biome). state.biome OVERRIDES gl-render defaults via window.__gen.
    bcSnow: [0.92,0.94,0.97],
    bandEdgesLo: [150.0,1200.0], bandEdgesHi: [3500.0,6500.0], snowEdges: [6000.0,8500.0],   // 8000/10500->6000/8500 (user 2026-06-11 'snowy mountains disappeared' -- see gl-render snowEdges note)   // snowEdges 5200/7000->8000/10500 (user 2026-06-10 'entire terrain white': the rock-by-height fix unmasked snow gates tuned pre-4x; full snow from 5.2km whitened the 11.6km massifs; coldSnow onset = snowEdges.x*0.5 follows)   // bandEdgesHi 1600/3200->3500/6500 (user 2026-06-10 'rockface everywhere'): tuned on the pre-4x terrain; with 11.6km peaks everything above 3200m read rock BY HEIGHT alone -- rescale the treeline to the new elevation range
    seaDepthM: 3000.0, slopeRock: [0.25,0.5],    // [0.25,0.5] USER-SET 2026-06-11 (explicit: 'set __gen.state.biome.slopeRock = [0.25,0.5]'); supersedes the [0.22,0.6] anti-rock-patch revert and the earlier [-0.6,1] calibration
  },
  // REAL-WORLD LOOK overhaul (terraformable lighting/shading levers; applyShaderGlobals sets window
  // globals; gl-render reads them via _g()). Beer-Lambert ocean, biome sat, mottle, sky-fill relief,
  // terminator glow, night floor + earthshine, exposure + post-ACES Look.
  look: {
    exposure: 1.0, skyFill: 0.45, biomeSat: 0.72, variationAmt: 0.04, colorVar: 0.5, vertexAO: 1.0,   // variationAmt 0.08->0.04 (user 2026-06-10 'blotchy': the ~50km value mottle painted light/dark patches across the massifs)
    nightFloor: 0.16, termWidth: 0.25, terminatorGlow: 0.30, lookSat: 1.15, lookContrast: 1.08,   // nightFloor 0.05->0.16: no black night terrain (2026-06-09)
    detailOverlay: 6.0, hazeMul: 0.65,   // 2026-06-10 'pale hazy + featureless': perlin-everywhere albedo+elevation fbm (user-tuned 6) + aerial-perspective strength cut
    ocean: { deep: [0.008,0.025,0.06], shallow: [0.07,0.22,0.26], k: [0.030,0.012,0.0045] },
  },
  // HPF band scales (multipliers on the anchor-field band base values; anchor-field.setBandScales).
  hpf: {
    enabled: 1,
    band: [
      // [continental, regional, local] -- scales multiply the band's baked param magnitude.
      { seaBiasScale: 1.0, elevAmpScale: 1.0, roughnessScale: 1.0 },
      { seaBiasScale: 1.0, elevAmpScale: 1.0, roughnessScale: 1.0 },
      { seaBiasScale: 1.0, elevAmpScale: 1.0, roughnessScale: 1.0 },
    ],
  },
};

function deepClone(o){ return JSON.parse(JSON.stringify(o)); }

function applyShaderGlobals(state){
  window.__pvNormal      = state.normal.pvNormal;
  window.__fsNormal      = state.normal.fsNormal;
  window.__elevEdgeInset = state.geometry.elevEdgeInset;
  if(state.lod.splitFactor != null) window.__splitFactor = state.lod.splitFactor;
  const L = state.look; if(L){
    window.__exposure = L.exposure; window.__skyFill = L.skyFill; window.__biomeSat = L.biomeSat;
    window.__variationAmt = L.variationAmt; window.__nightFloor = L.nightFloor; window.__termWidth = L.termWidth;
    if(L.colorVar != null) window.__colorVar = L.colorVar;
    if(L.detailOverlay != null) window.__detailOverlay = L.detailOverlay;
    if(L.hazeMul != null) window.__hazeMul = L.hazeMul;
    if(L.vertexAO != null) window.__vertexAO = L.vertexAO;
    window.__terminatorGlow = L.terminatorGlow; window.__lookSat = L.lookSat; window.__lookContrast = L.lookContrast;
    // gl-render reads window['__'+uniformName] (gl-render.js o3 helper) -> the globals must carry
    // the 'u' prefix (__uOceanDeep, not __oceanDeep); the unprefixed names were dead levers.
    if(L.ocean){ window.__uOceanDeep = L.ocean.deep; window.__uOceanShallow = L.ocean.shallow; window.__uOceanK = L.ocean.k; }
  }
}
function applyHpf(state){
  const f = HPF(); if(!f) return false;
  if(f.setBandScales){ state.hpf.band.forEach((b,i)=> f.setBandScales(i, b)); }
  if(window.__hpfRebake) window.__hpfRebake();
  return true;
}

const __gen = {
  defaults: DEFAULTS,
  state: deepClone(DEFAULTS),

  apply(){
    applyShaderGlobals(this.state);
    applyHpf(this.state);
    const orch = ORCH(); if(orch && orch.clearCache) orch.clearCache();   // re-sync biome/HPF into the next frame
    return { ok: true };
  },

  // set one field by dot-path (e.g. 'biome.seaDepthM', 'hpf.band.1.elevAmpScale') and re-apply.
  set(path, v){
    const parts = path.split('.'); let o = this.state;
    for(let i=0;i<parts.length-1;i++){ o = o[parts[i]]; if(o==null) return {err:'bad-path:'+path}; }
    o[parts[parts.length-1]] = v;
    return this.apply();
  },

  // read the LIVE shader globals back (truth, not just this.state).
  get(){
    return { state: deepClone(this.state),
      liveShader: { pvNormal:window.__pvNormal, fsNormal:window.__fsNormal,
                    elevEdgeInset:window.__elevEdgeInset, splitFactor:window.__splitFactor } };
  },

  reset(){ this.state = deepClone(this.defaults); return this.apply(); },
  serialize(){ return JSON.stringify(this.state); },
  load(json){ try{ this.state = (typeof json==='string')?JSON.parse(json):json; return this.apply(); }catch(e){ return {err:String(e)}; } },
};

if (typeof window !== 'undefined') window.__gen = __gen;
export default __gen;
