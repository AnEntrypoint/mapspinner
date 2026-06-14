// mapspinner WebGL2 terrain render shader.
// VS: spherical deformation + direct per-vertex sphere projection.
// FS: sample normal + albedo, sun-lit Lambert with ambient floor.
// The JS prepends #version + precision and compiles _VERTEX_ / _FRAGMENT_ separately.

// --- DIRECT per-vertex sphere-projection uniforms ------
// SINGLE INSTANCED DRAW: defOffset (ox,oy,l,level) + the face frame are PER-INSTANCE now (one
// gl.drawElementsInstanced over the whole visible leaf set, no per-quad uniform churn). In the VS
// they come from instance attributes (iOffset + iFace); defRadius/defViewProjRel stay uniforms
// (same for every instance). The _PROBE_ collision program uses its own probeDir, not these.
// W7 highp ISLANDS: every world-scale quantity here is ~6.4e6 m and the projection cancels at that
// magnitude -- fp16 (mediump, range +-65504, ~3 sig digits) would shatter it. Declared highp under the
// mediump global default so the planet-scale fp32 cancellation fix (camera-relative vRel) stays intact.
uniform highp float defRadius;         // R, sphere radius (~6.4e6 m)
uniform highp mat4 defViewProjNoEye;   // proj*viewRel WITHOUT folded translate(-eye) -- for camera-relative VS pos
uniform highp vec3 defCamDir;          // unit camera direction (eye/|eye|) -- vertex-jitter precision fix
uniform highp float defCamAlt;         // camera altitude above the sphere (|eye|-R)

// HIERARCHICAL PARAMETER FIELD (HPF) continental texture (shared VS+FS): per-face 2D-arrays
// baked from anchor-field.js. W12 PACK (mob-w12): the old single RGBA32F (16B/texel) is split
// into TWO smaller textures -- hpfPool RG16F (R=seaBias[m], G=elevAmp; the two PRECISION-sensitive
// floats the geometry rides on) + hpfPool2 RG8 (R=temp, G=humid; both already [0,1] climate, 8-bit
// is ample). 4B+2B = 6B/texel vs 16B = 62% less HPF VRAM/bandwidth, with NO silhouette change
// (seaBias/elevAmp keep float precision; temp/humid quantize to 1/255, far below the biome
// soft-threshold widths). hpfSample DECODES both back into the same vec4 callers expect
// (seaBias, elevAmp, temp, humid) so every consumer (VS bias, vClimate, probe, heightbake) is
// unchanged. UNCONDITIONAL format (single version; no tier branch).
uniform sampler2DArray hpfPool;    // RG16F: r=seaBias[m], g=elevAmp
uniform sampler2DArray hpfPool2;   // RG8:   r=temp[0,1], g=humid[0,1]
uniform int hasHpf;
uniform float uHpfInset;          // HPF seam-inset sampler (window.__hpfInset; 0=centred bake default, 1=edge-inset seam->0). MUST match planet-orchestrator bakeFace fu mapping. Declared here (before hpfSample) so all stages see it.
// W11 FORMAT PROBE (NOT a quality tier): OES_texture_float_linear probed ONCE at init (gl-render.js
// sets this). 1 = the float/half-float atlas pools filter LINEAR in hardware -> the manual 4-tap
// bilinear in hpfSample collapses to a single hardware texture() call.
// 0 = the driver would silently fall back to NEAREST (the 'square steps / banding' regression) so the
// manual 4-tap is kept for correctness. Numerically equivalent either way -- a FORMAT-correctness
// branch, the one admissible runtime branch.
uniform int uFloatLinearOK;
// world unit dir -> cube face + uv[0,1] (matches anchor-field.js dirToFaceUV / FACE_FRAME).
void hpfFaceUV(vec3 d, out int face, out vec2 uv) {
    vec3 a = abs(d);
    float u, v, sc;
    if (a.x >= a.y && a.x >= a.z) { sc = 1.0/a.x;
        if (d.x > 0.0) { face = 0; u = -d.z*sc; } else { face = 1; u = d.z*sc; } v = d.y*sc; }
    else if (a.y >= a.z) { sc = 1.0/a.y;
        if (d.y > 0.0) { face = 2; v = -d.z*sc; } else { face = 3; v = d.z*sc; } u = d.x*sc; }
    else { sc = 1.0/a.z;
        if (d.z > 0.0) { face = 4; u = d.x*sc; } else { face = 5; u = -d.x*sc; } v = d.y*sc; }
    uv = vec2(u*0.5 + 0.5, v*0.5 + 0.5);
}
highp vec4 hpfSample(vec3 dir) {   // W7: R=seaBias is metres (~1600) -> highp so cbias keeps full precision
    if (hasHpf == 0) return vec4(0.0, 1.0, 0.5, 0.5);   // seaBias 0, elevAmp 1, temp/humid .5
    int face; vec2 uv; hpfFaceUV(normalize(dir), face, uv);
    // W11: when float-linear filters in hardware, one texture() per pool == the 4-tap result. The
    // bake samples at texel centres so the centred (non-inset) hardware tap matches the manual one
    // exactly; the inset bake keeps the manual taps (its tap centres are k/(sz-1), not the hardware
    // (k+0.5)/sz grid) so it stays correct regardless.
    // HARDWARE-TAP FAST PATH DISABLED for the crease fix: a single hardware texture() is HARDWARE
    // BILINEAR = the SAME C0 texel-edge slope kink as raw manual weights, so on a float-linear GPU it
    // bypassed the quintic-weight fix below and the seaBias crease persisted (user 2026-06-09). Always
    // take the manual path so the quintic C2 weights apply to seaBias/elevAmp (geometry-critical). Cost:
    // 4 taps vs 1, but only per-vertex in VS/PROBE (negligible; FS reads the interpolated varying).
    // (uHpfInset>0.5 inset bake already used the manual path; this just unifies onto it always.)
    // MANUAL BILINEAR (2026-06-06, user: 'elevation divided into squares -> square STEPS / STAIRS,
    // should be smooth'). hpfSample drives vH = cbias + bShape (terrain.glsl ~743): cbias=seaBias and
    // reliefMul/ridgeMul derive from this sample, so any quantization here steps the GEOMETRY in
    // ~50km (128-texel/face) squares. The hpfPool texture is FLAGGED LINEAR (planet-orchestrator.js)
    // but RGBA32F LINEAR filtering REQUIRES OES_texture_float_linear; if that extension is absent the
    // driver SILENTLY falls back to NEAREST -> per-texel-cell constants -> the visible stairs (the
    // orchestrator's own getExtension comment names this exact 'elevations look square from NEAREST'
    // failure). Fix: bilinear in-shader so the field is smooth REGARDLESS of hardware float-linear --
    // textureSize for the texel grid, 4 taps at the surrounding texel centres, 2D lerp. Tap coords are
    // clamped to texel centres in [0.5/sz, 1-0.5/sz] = CLAMP_TO_EDGE equivalent, so this reproduces the
    // intended hardware LINEAR exactly (no new cube-face seam) when the ext IS present, and fixes the
    // steps when it is not.
    vec2 sz = vec2(textureSize(hpfPool, 0).xy);
    // SEAM-INSET MATCHED SAMPLER (hpf-seam-inset-bake, gated by uHpfInset to match the bake fu=x/(RES-1)).
    // Inset: texel k sits at uv=k/(sz-1) (edge texels at 0 and 1), so t=uv*(sz-1) and the tap centres are
    // k/(sz-1). Centred (default): texel k at uv=(k+0.5)/sz, t=uv*sz-0.5. Both must agree with the bake.
    bool inset = uHpfInset > 0.5;
    vec2 denom = inset ? (sz - 1.0) : sz;
    vec2 t  = inset ? (uv * denom) : (uv * sz - 0.5);
    vec2 f  = fract(t);
    vec2 t0 = floor(t);
    vec2 hb = 0.5 / sz;                           // half-texel in uv (clamp band, centred case)
    vec2 c0 = inset ? (t0)              / denom : clamp((t0 + 0.5)        / sz, hb, 1.0 - hb);
    vec2 c1 = inset ? (t0 + vec2(1.0, 0.0)) / denom : clamp((t0 + vec2(1.5, 0.5)) / sz, hb, 1.0 - hb);
    vec2 c2 = inset ? (t0 + vec2(0.0, 1.0)) / denom : clamp((t0 + vec2(0.5, 1.5)) / sz, hb, 1.0 - hb);
    vec2 c3 = inset ? (t0 + vec2(1.0, 1.0)) / denom : clamp((t0 + 1.5)        / sz, hb, 1.0 - hb);
    vec2 uv00 = clamp(c0, vec2(0.0), vec2(1.0));
    vec2 uv10 = clamp(c1, vec2(0.0), vec2(1.0));
    vec2 uv01 = clamp(c2, vec2(0.0), vec2(1.0));
    vec2 uv11 = clamp(c3, vec2(0.0), vec2(1.0));
    float ff = float(face);
    // W12 DECODE: 4 bilinear taps from EACH packed texture, reassembled into the legacy
    // vec4 (seaBias, elevAmp, temp, humid). Same uv/weights for both -> one bilinear field.
    vec2 s00 = texture(hpfPool,  vec3(uv00, ff)).rg;   // (seaBias, elevAmp)
    vec2 s10 = texture(hpfPool,  vec3(uv10, ff)).rg;
    vec2 s01 = texture(hpfPool,  vec3(uv01, ff)).rg;
    vec2 s11 = texture(hpfPool,  vec3(uv11, ff)).rg;
    // QUINTIC C2 bilinear weights (NOT raw f=fract(t)). Linear weights are C0 -> the interpolated
    // seaBias/elevAmp SLOPE kinks at every texel-cell edge; seaBias is added RAW to h (cbias+bShape)
    // so on a slope crossing a texel boundary that kink is a STRAIGHT ELEVATION CREASE (user 5 grids
    // 2026-06-09: a vertical cliff at one grid column, 0.4-38m, all rows). Quintic w is value-identical
    // at texel centres (w(0)=0,w(1)=1) so amplitude/field UNCHANGED -- only the inter-texel slope is C2.
    // Same fix class as the vnoise2 quintic. SEAM-SAFE: the texel-centre clamp (above) is untouched.
    vec2 w = f*f*f*(f*(f*6.0-15.0)+10.0);
    vec2 se  = mix(mix(s00, s10, w.x), mix(s01, s11, w.x), w.y);
    vec2 t00 = texture(hpfPool2, vec3(uv00, ff)).rg;   // (temp, humid)
    vec2 t10 = texture(hpfPool2, vec3(uv10, ff)).rg;
    vec2 t01 = texture(hpfPool2, vec3(uv01, ff)).rg;
    vec2 t11 = texture(hpfPool2, vec3(uv11, ff)).rg;
    vec2 th  = mix(mix(t00, t10, w.x), mix(t01, t11, w.x), w.y);   // same quintic C2 weights (temp/humid)
    return vec4(se.x, se.y, th.x, th.y);
}

// 3D value noise of a world-space point (continuous everywhere on the sphere). SHARED by the VS
// (broadShapeM continuous-field shape) AND the FS (riverMask drainage network), so it must live
// in the common preamble -- NOT inside #ifdef _VERTEX_, else the FS sees no snoise3 and the
// fragment program fails to link ('snoise3: no matching overloaded function'), aborting renderer init.
// W7 highp ISLAND: snoise3/shash3 lattice inputs reach freq*dir ~1.4e4 (fine octaves) and rely on
// integer floor()/fract() cell precision -- fp16 (~3 digits) would collapse every cell to mush. ALL
// noise primitives + their P/p args are highp; the [-1,1] RESULT narrows back to the mediump default.
highp float shash3(highp vec3 p){ p=fract(p*0.3183099+vec3(0.1,0.2,0.3)); p+=dot(p,p.yzx+19.19); return fract((p.x+p.y)*p.z + (p.y+p.z)*p.x); }
float snoise3(highp vec3 P){ highp vec3 i=floor(P),f=fract(P); highp vec3 u=f*f*(3.0-2.0*f);
  float n000=shash3(i),n100=shash3(i+vec3(1,0,0)),n010=shash3(i+vec3(0,1,0)),n110=shash3(i+vec3(1,1,0));
  float n001=shash3(i+vec3(0,0,1)),n101=shash3(i+vec3(1,0,1)),n011=shash3(i+vec3(0,1,1)),n111=shash3(i+vec3(1,1,1));
  float x00=mix(n000,n100,u.x),x10=mix(n010,n110,u.x),x01=mix(n001,n101,u.x),x11=mix(n011,n111,u.x);
  return mix(mix(x00,x10,u.y),mix(x01,x11,u.y),u.z)*2.0-1.0; }   // [-1,1]

// ANALYTIC-DERIVATIVE value noise (IQ): returns vec4(value[-1,1], d/dPx, d/dPy, d/dPz). The exact
// gradient of the SAME trilinear+smoothstep field as snoise3 above, so the geometry it shapes and the
// normal it lights are the one field with NO finite-difference step to alias the fine octaves -- this
// is the core of the one-comprehensive-system design (mem tv8-one-system-design-analytic-deriv-fbm):
// the lit normal becomes exact at every scale, killing the deck flat-clay (the 200m/2000m FD steps
// averaged the fine octaves out). du = derivative of the smoothstep weight 6f(1-f).
// (snoise3D analytic-derivative noise DELETED 2026-06-11 dead-code sweep: its sole consumer broadShapeMD was removed earlier.)

// ---- CONTENT CARVE FIELDS (SHARED VS+FS preamble). These cut lakes/rivers/canyons into the
// elevation (VS uses the depth) AND gate the water/rock colour (FS uses the wet/depth mask), so the
// colour sits EXACTLY in the carved depression at every LOD -- they MUST be visible to BOTH stages
// (defining them inside #ifdef _VERTEX_ made the FS fail to link: 'no matching overloaded function',
// the same class as the historical snoise3-in-VS-only bug). Pure fn of world dir -> seam-safe +
// LOD-invariant. Each has an EROSION profile: a wide graded shoulder/valley/bench that blends into
// terrain plus a deeper core, and returns a wet/depth mask the FS colours.
const float LAKE_CARVE_DEPTH = 90.0;   // metres of bowl depth at basin centre
float lakeBasinField(vec3 dir){ return 0.5 + 0.5 * snoise3(dir * 55.0 + vec3(4.0, 9.0, 1.0)); }
float lakeCarveM(vec3 dir, out float wet){
    float basin = lakeBasinField(dir);
    float shoulder = smoothstep(0.30, 0.74, basin);     // gentle outwash apron (blends far into terrain)
    float bowl     = smoothstep(0.50, 0.80, basin);     // deeper basin starts earlier
    wet = smoothstep(0.50, 0.70, basin);                // softer shoreline transition
    return -LAKE_CARVE_DEPTH * (0.65 * shoulder + 0.35 * bowl);   // more apron, gentler banks
}
float lakeCarveM(vec3 dir){ float w; return lakeCarveM(dir, w); }
// River and canyon are the SAME incision algorithm (a 4-octave ridged 1-abs(snoise3) network),
// differing only in base frequency, phase, depth, and threshold band. One shared field helper.
// `d` is already the sampling coordinate (normalized dir, optionally phase-shifted by the caller).
//
// AXIS-BIAS + ORGANIC FIX (user: 'canyons draw H/V lines, nothing in between; less predictable').
// snoise3 is VALUE noise on an integer lattice, so ridged 1-abs(snoise3) crests align to the x/y/z
// axes -> horizontal/vertical channel lines. Two fixes, both keeping it a pure world-dir field:
//   ROTATE the sample domain per octave by a fixed CONSTANT 3D rotation (precomputed mat3, no
//   per-call trig/normalize, no extra noise taps -- cheap) so each octave's lattice axes point
//   differently -> the summed ridged crests run in EVERY direction, not just the x/y/z lattice H/V.
//   The matrix rows are a fixed tilted basis (~no axis alignment); det ~= 1 so it doesn't drift scale.
const mat3 OCT_ROT = mat3( 0.80, 0.36, -0.48,   -0.48, 0.86, -0.18,    0.36, 0.36, 0.86);
// FXC UNROLL-DEFEAT runtime-bounded loops (2026-06-12): uniform int guards prevent FXC from
// fully unrolling these loops (which triggers mis-translation on AMD D3D11). Each guard falls
// back to the original octave count when the uniform is 0 or unset (e.g. probe program).
uniform int uOctMax;            // broadShapeM octave count (12); runtime-bound to defeat FXC unrolling
uniform int uInciseRidgeOcts;   // inciseRidgeField octave count (4); runtime-bound to defeat FXC unrolling
uniform int uBroadLowOcts;      // broadShapeLowM octave count (8); runtime-bound to defeat FXC unrolling
uniform int uPeakOcts;          // broadShapeM peak crest octave count (3); runtime-bound to defeat FXC unrolling
uniform int uVtxBaseOcts;       // vtxDisplace base fBm octave count (6); runtime-bound to defeat FXC unrolling
uniform int uVtxErodeOcts;      // vtxDisplace mountain erosion octave count (4); runtime-bound to defeat FXC unrolling
uniform int uDetailFbmOcts;     // detailFbm octave count (3); runtime-bound to defeat FXC unrolling
uniform int uFSDetailOcts;      // FS detail overlay octave count (3); runtime-bound to defeat FXC unrolling
float inciseRidgeField(vec3 d, float baseFreq, float freqMul){
    vec3 p = d;
    float freq = baseFreq, amp = 1.0, sum = 0.0, norm = 0.0;
    int irOcts = (uInciseRidgeOcts > 0) ? uInciseRidgeOcts : 4;
    for (int o = 0; o < irOcts; o++){
        sum += amp * (1.0 - abs(snoise3(p * freq)));
        norm += amp; freq *= freqMul; amp *= 0.5; p = OCT_ROT * p;   // rotate domain each octave
    }
    return sum / norm;                                         // ->1 on the channel network
}
const float RIVER_INCISE_DEPTH = 120.0;   // metres at the channel thalweg
float riverRidgeField(vec3 dir){ return inciseRidgeField(normalize(dir), 40.0, 2.03); }
float riverCarveM(vec3 dir, out float wet){
    float ridge = riverRidgeField(dir);
    float valley  = smoothstep(0.30, 0.94, ridge);             // gentle eroded valley sides, start wider
    float thalweg = smoothstep(0.75, 0.96, ridge);             // deep channel core
    wet = smoothstep(0.78, 0.94, ridge);                       // flowing-water line
    return -RIVER_INCISE_DEPTH * (0.7 * valley + 0.3 * thalweg);   // more valley, gentler banking
}
float riverCarveM(vec3 dir){ float w; return riverCarveM(dir, w); }
const float CANYON_INCISE_DEPTH = 1400.0;  // metres at the gorge floor (user 2026-06-02: deepen+widen so canyons visibly sculpt the elevation; was 480m = invisible vs multi-km relief)
uniform float canyonDepthMul;              // LIVE canyon-depth lever (window.__canyonDepth; 1.0 default)
uniform float uVsCheap;                     // VS profiling: >0.5 skips all carves in composeHeight (window.__vsCheap; gpuTimer carve-cost A/B)
// W5: uNrmGain deleted (only fed the removed VS slopeGain).
uniform float uVertexAO;                   // per-vertex shading/AO strength lever (window.__vertexAO; 1.0 default)
// SEPARATE WATER SURFACE (user 2026-06-11 'the ocean should be a separate surface'): the same
// program draws TWO instanced passes -- terrain (uIsWater=0, true seabed geometry, land shading)
// then water (uIsWater=1, the same leaves pinned to sea level, animated ocean shading, alpha-
// blended over the seabed). One program + a uniform flag = no second cold compile, no duplicated
// per-frame uniform churn, and the branch is uniform-coherent (free on the GPU).
uniform float uIsWater;                    // 0 = terrain pass, 1 = water-surface pass
uniform float uUnderwater;                 // 0 = camera above water, 1 = camera below sea level
uniform float uBeachTopM;                  // beach ceiling (m): below this, grass/snow yield to sand (window.__beachTop; 30 default)
uniform float uBeachShelfM;                // land coastal-shelf top (m): h<this is eased (h*h/S) so the coast rises gently from the waterline = wide beach (window.__beachShelf; 300 default). GEOMETRY (composeHeight+vH).
uniform float uHiFreqCut;                  // hi-freq elevation-noise attenuation, applied to ALL hi-freq sources:
                                           // broadShapeM/MD fine octaves (o>=6) + vtxDisplace micro-relief
                                           // (window.__hiFreqCut; default 0.25 = the user's 4x reduction, 2026-06-06)
// ANCHOR-STEP A/B TOGGLES (per-area stairstep root-hunt, workflow wrxo0rr7a). Each defaults to 0.0 =
// CURRENT behaviour; set window.__<name>=1 to WIDEN that anchor-keyed smoothstep band so the 0->1 swing
// spreads over more ground (the narrow-band-on-slow-anchor mechanism that snaps relief+normal "here and
// there" along anchor contours). The user binary-searches these live at the affected spot; the one that
// dissolves the step = the root. Once found, the winning widen is baked in unconditionally + toggle removed.
uniform float uMtnBandWide;                // widen mtn=smoothstep(16.8,18.6,elevAmp) -> (14.5,19.5) [TOP SUSPECT: unlocks 2600m belt massif across a thin contour]
uniform float uClimateRelief;              // widen wetLowFlat(0.66,0.9,humid) + coldFlat(0.18,0.34,temp) reliefMul gates
uniform float uIsleWide;                    // widen isleZone seaBias gates (50,350)+(900,1600) -> (30,600)+(600,2200)
uniform float uCarveWide;                   // widen the river/canyon/lake/dune CLIMATE gates so carve depth fades in over a wide span
float canyonRidgeField(vec3 dir){ return inciseRidgeField(normalize(dir) + vec3(13.7, -4.2, 8.9), 380.0, 2.07); }  // phase offset matches FS canyonMask. baseFreq 380 = ~100km gorge network (user 2026-06-14: doubled 380->760 to restore deck-visible canyons after a 380->96 regression that put gorges ~400km apart = 'missing', then HALVED 760->380 back to the ~100km baseline). Shared VS carve + FS mask -> congruent by construction.
// CANYON cross-section now reads as a CANYON, not a V-notch: STEEP WALLS + a FLAT FLOOR.
//   wall = a sharp smoothstep band -> the carve drops fast over a narrow ridge interval (the cliff
//          walls), instead of the old gentle bench+gorge blend that made shallow V-troughs.
//   floor = clamped: once past the wall the depth saturates to the flat gorge bottom (a river runs
//          there, not an ever-deepening point). bench keeps a soft eroded rim above the wall lip.
// Pure world-dir (LOD-invariant, seam-safe). depth out = 0 rim -> 1 floor (drives FS strata + AO).
float canyonCarveM(vec3 dir, out float depth){
    float ridge = canyonRidgeField(dir);
    // WIDENED bands (user 2026-06-02 deepen+widen): the old .78/.86/.905/.93 intervals were a very
    // narrow ridge slice -> thin hairline gorges. Widen so the canyon reads as a BROAD gorge: a wide
    // eroded rim shoulder, a steep but visible wall, and a wide flat floor that the gorge bottoms out on.
    float bench = smoothstep(0.62, 0.78, ridge);               // wide eroded rim shoulder
    float wall  = smoothstep(0.70, 0.86, ridge);               // STEEP cliff wall (wider band)
    float floorF= smoothstep(0.80, 0.93, ridge);               // reach the flat gorge floor, then clamp
    depth = max(wall, floorF);
    // 0.18 rim shoulder + 0.82 wall-to-floor; floorF clamps so the bottom is flat (no infinite V).
    float profile = 0.18 * bench + 0.82 * max(wall, floorF);
    float dmul = canyonDepthMul > 0.0 ? canyonDepthMul : 1.0;   // 0 = uniform unset (e.g. probe prog)
    float carve = -CANYON_INCISE_DEPTH * dmul * profile;
    // FINE TRIBUTARY GULLIES (user 2026-06-02: 'at 2m our canyons are <2m'). The main canyon network
    // (freq 380 ~100km) is sparse + broad, so close up the ground has no canyon-scale erosion. Add a
    // FRACTAL CONTINUATION: 2 finer ridged incision octaves (~10km + ~2km wavelength) carving shallower
    // gullies/ravines (~55m + ~16m deep) so arid terrain shows branching erosion channels at the 10-
    // 100m scale the low-altitude camera sees. Pure world-dir (LOD-invariant); the mesh resolves them
    // at maxLevel 16 (~7m cells). Each gully octave = a thinned ridged field, depth scaled to its wl.
    vec3 dn = normalize(dir);
    float g1 = inciseRidgeField(dn + vec3(5.1, 8.3, -2.7), 219.0, 2.11);   // ~40km tributaries (was 875->438)
    float g2 = inciseRidgeField(dn + vec3(-7.4, 1.9, 6.2), 875.0, 2.05);  // ~10km gullies (was 3500->1750)
    float g3 = inciseRidgeField(dn + vec3(2.2, -4.1, 8.8), 1750.0, 2.02); // ~1.2km branching ravines (deck scale)
    // HIGH-FREQ fractal depth (user 2026-06-11 'leave the low frequency canyons deep, only reduce the
    // high frequency fractals'). DEEPENED + sharpened 2026-06-14 (user: mountains need visible canyons):
    // the main 100km gorge network keeps full CANYON_INCISE_DEPTH; the tributary octaves incise deeper
    // with narrower walls so ravines read at the deck. g3 subdivides the bigger gullies at maxLevel.
    carve += -450.0 * dmul * smoothstep(0.66, 0.90, g1);   // ~10km tributaries (deeper, narrower)
    carve += -200.0 * dmul * smoothstep(0.70, 0.92, g2);   // ~2.5km gullies
    carve +=  -90.0 * dmul * smoothstep(0.72, 0.93, g3);   // ~1.2km branching ravines
    return carve;
}
float canyonCarveM(vec3 dir){ float dd; return canyonCarveM(dir, dd); }

// CLIFFS AS REAL MESA LANDFORMS (user 2026-06-02: 'cliffs still patches of dots', 'we dont have
// realistic cliffs yet'). The height-QUANTIZATION terrace produced thin scattered contour risers =
// dots, NOT escarpments. REDESIGN: a cliff is a coherent raised PLATEAU (mesa) with a sharp RIM.
//   mesaField(d) -> a smooth low-frequency field; where it exceeds a threshold the land is a mesa
//   TOP, raised by a fixed height; the THRESHOLD CROSSING is a narrow band = the near-vertical cliff
//   RIM, a CONNECTED line around the whole mesa (not speckle). Stacking two octaves gives mesas +
//   buttes (smaller mesas on top). Pure world-dir, LOD-invariant, seam-safe. Returns metres to ADD;
//   cliffOut ->1 on the rim band (drives FS strata + steep material) so cliffs are lit + textured.
const float MESA_HEIGHT = 1000.0; // metres a mesa top stands above the floor (user 2026-06-02 deepen+widen; was 380m = invisible vs multi-km relief)
uniform float cliffAmt;           // LIVE cliff-strength lever (window.__cliffAmt; 1.0 default)
highp float broadShapeLowM(vec3 dir);   // W7: forward decl (metres -> highp); atlas-backed slope helper for the region slope gate
// COHERENT BADLANDS REGION MASK: a LOW-FREQUENCY world-dir noise (freq 11 ~ continental patches)
// thresholded to a minority fraction -> WHERE mesa country occurs. Mesas exist only inside a region
// so cliffs cluster into badlands, never speckle the whole planet.
float badlandsRegion(vec3 d){
    float r = snoise3(d * 11.0 + vec3(31.7, -12.3, 5.1));   // [-1,1], coherent ~continental patches
    return smoothstep(0.30, 0.55, r);                       // top fraction -> 1 inside badlands
}
// Smooth mesa potential in [0,1]: a couple of mid-frequency octaves (freq ~55/130 -> mesas ~80km /
// ~35km across). Domain-warped so mesa outlines are organic (not blobby circles).
float mesaField(vec3 d){
    vec3 w = d; w.xy += 0.10 * vec2(snoise3(d*40.0+5.0), snoise3(d*40.0+11.0));   // 3->2 warp taps (max-speed sweep): z-warp dropped, outlines stay organic
    float a = snoise3(w * 55.0 + vec3(3.1, 7.7, 1.3));     // big mesas
    float b = snoise3(w * 130.0 + vec3(9.4, 2.2, 6.6));    // buttes on top
    return 0.5 + 0.5 * (0.7 * a + 0.3 * b);                // [0,1]
}
highp float cliffTerraceM(vec3 dir, highp float h, out float cliffOut){   // W7: h is metres (highp); cliffOut mask stays mediump
    cliffOut = 0.0;
    if (h <= 0.0) return 0.0;
    vec3 d = normalize(dir);
    // REGION GATE: mesas only inside a coherent badlands region (no planet-wide speckle).
    float region = badlandsRegion(d);
    if (region <= 0.0) return 0.0;
    // EXCLUDE steep mountains (a mesa is a flat-topped raised block, not a peak). Macro slope from
    // broadShapeLowM over a fixed 1.2km step -> fade mesas out where the broad land is already steep.
    const float e = 1200.0;
    vec3 ux = normalize(cross(abs(d.y) < 0.99 ? vec3(0.0,1.0,0.0) : vec3(1.0,0.0,0.0), d));
    highp float h0 = broadShapeLowM(d);                          // W7: metres, highp
    highp float gx = (broadShapeLowM(normalize(d + ux * (e/defRadius))) - h0) / e;
    float flatness = 1.0 - smoothstep(0.05, 0.16, abs(gx));
    float gate = region * flatness;
    if (gate <= 0.0) return 0.0;
    // MESA STEP: raise the land where mesaField crosses THR over a NARROW band TH -> the band is the
    // cliff RIM (a connected contour around the mesa), the inside is the raised flat-ish top. Two
    // levels (top + a higher butte tier) so cliffs stack like real mesa country.
    float m = mesaField(d);
    const float THR = 0.56, TH = 0.07;             // rim centre + half-width (widened for the taller 1000m escarpment -> readable slope, still a sharp rim)
    float top  = smoothstep(THR - TH, THR + TH, m);              // 0 floor -> 1 mesa top
    float butte= smoothstep(0.74 - TH, 0.74 + TH, m);            // higher butte tier
    float rise = (0.7 * top + 0.3 * butte) * MESA_HEIGHT;        // metres raised
    // rim mask: ->1 INSIDE either transition band (where the step is climbing = the steep face).
    float rim = max(top * (1.0 - top), butte * (1.0 - butte)) * 4.0;   // bell, peak 1 mid-rim
    cliffOut = clamp(rim, 0.0, 1.0) * gate;
    float camt = cliffAmt > 0.0 ? cliffAmt : 1.0;  // 0 = uniform unset (e.g. probe prog)
    return rise * camt * gate;                     // metres added (mesa top raised; rim = the cliff)
}

// DUNE FIELD (arid/desert anchors, ref: keaukraine/webgl-dunes). A rolling sand surface = large
// smooth low-frequency dunes with ASYMMETRIC crests (windward shallow, lee steep) + superimposed
// fine wind ripples. Pure world-dir field (LOD-invariant). Returns metres of dune relief (>=0,
// added on top of land) and `crest` in [0,1] (1 at a dune ridge -> the FS lightens sand there).
const float DUNE_AMP = 120.0;     // metres of big-dune relief
float duneFieldM(vec3 dir, out float crest){
    vec3 d = normalize(dir);   // UNIT dir keeps freq*d bounded -> no planet-scale fp32 lattice break.
    // BIG DUNES ~3km: low-freq smooth noise, asymmetric crest (pow sharpens the lee face).
    float big = snoise3(d * 2000.0 + vec3(2.0, 7.0, 5.0)) * 0.5 + 0.5;  // [0,1]
    float dune = pow(big, 1.6);                                          // windward-shallow/lee-steep
    // MEDIUM dunes ~1km riding on the big ones. (The old ~150-220m wind RIPPLES were REMOVED: at
    // that wavelength the mesh vertices undersample them -> shimmer/moire 'UV issue' + a wild FD
    // normal, with no resolvable benefit. The rolling big+medium dunes are the dune read.)
    float med = snoise3(d * 6000.0) * 0.5 + 0.5;
    crest = smoothstep(0.55, 0.92, big);
    return DUNE_AMP * dune + 40.0 * (med * dune);
}

// TERRAIN HEIGHT FIELD: one continuous world-dir fBm -- a finer LOD is a denser SAMPLE of the same
// field (LOD-invariant, seam-safe by construction).
// CHEAP 8-octave variant for the VS lit-normal finite difference (silhouette+macro slope only).
highp float broadShapeLowM(vec3 dir){   // W7: metres (~13000) + freq (~49152) accumulators are highp islands
  if (hasHpf == 0) return 0.0;
  vec3 d = normalize(dir);
  highp float amp = 6500.0, freq = 0.75, sum = 0.0;
  int blOcts = (uBroadLowOcts > 0) ? uBroadLowOcts : 8;
  for (int o=0; o<blOcts; o++){ sum += amp * snoise3(d*freq); amp *= (o < 6 ? 0.66 : 0.82); freq *= 2.0; }
  return sum - 500.0;
}
// ---- SHARED micro-relief helpers (MOVED to the common preamble, THC-Normal W1, so composeHeight()
// in the VS/PROBE region can call them). vtxDetail = the micro-relief A/B global; vnoise2
// is the 2D value-noise vtxDisplace octaves sample; ruggedFromElevAmp + faceWarp map anchor->amplitude
// and face-local metres->warped metres. All pure, dependency-light (defRadius + smoothstep only).
uniform float vtxDetail;   // micro-relief strength (1=on) -- kept as a global so it can be A/B'd
// W7 highp ISLAND: vnoise2 args are face-local metres / wavelength = fp(~6.4e6)/wl(~1m) -> ~6e6, far
// past fp16. The vtxDisplace micro-relief lattice needs full precision or the ground bump quantizes.
// W10 PRECISION: the old fract(p*443.897) hash overflowed fp32 integer-exactness at fine octaves
// (wl~10m -> p~6.4e5, *443.897 -> 2.84e8 >> 2^24=1.67e7), so fract() lost low bits and the corner
// hashes QUANTIZED to a few discrete values -> per-cell height plateaus = faint ~0.48m terracing on
// flat ground. Fix: hash the INTEGER lattice cell with a uint bit-mix (no large float, no fract).
// Pure function of the world-consistent integer cell index -> adjacent LODs and cube faces agree
// EXACTLY (no new seam). Amplitude/range UNCHANGED: still a uniform [0,1] per-cell value, same field
// statistics; this removes the quantization, it does not smooth. (p is the exact integer lattice
// index <2^24, so ivec2(p) is exact.)
highp float vhash(highp vec2 p){
  uvec2 q = uvec2(ivec2(p));                 // two's-complement bits of the exact integer cell index
  uint h = q.x * 1597334677u + q.y * 3812015801u;   // decorrelate the two axes
  h ^= h >> 16; h *= 2654435769u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;   // bit avalanche
  return float(h) * (1.0 / 4294967296.0);    // [0,1)
}
// QUINTIC C2 interp (Perlin u=6t^5-15t^4+10t^3), NOT the C1 cubic smoothstep (3t^2-2t^3). The cubic's
// SECOND derivative is discontinuous at every lattice cell edge -> the noise SLOPE kinks at each ~wl cell
// boundary; on a gentle mountain flank (where the kink is not swamped by relief) that ~2.7m/10m slope spike
// every ~320m (vtxDisplace wl0) reads as a LOCAL stairstep in both elevation and the central-diff normal
// (user 2026-06-09 'only a few very local spots'). Quintic is C2 -> slope continuous across cells, NO kink,
// and amplitude UNCHANGED (this does not smooth the terrain, it removes the discontinuity in its derivative).
float vnoise2(highp vec2 p){ highp vec2 i=floor(p),f=fract(p); vec2 u=f*f*f*(f*(f*6.0-15.0)+10.0);
  float a=vhash(i),b=vhash(i+vec2(1,0)),c=vhash(i+vec2(0,1)),d=vhash(i+vec2(1,1));
  return mix(mix(a,b,u.x),mix(c,d,u.x),u.y)*2.0-1.0; }   // [-1,1]
// (vnoise2D DELETED 2026-06-11 dead-code sweep: no callers since the VS gradient block was removed.)
// anchor elevAmp -> rugged multiplier (BAKED elevAmp range ~14.5-18.9).
float ruggedFromElevAmp(float elevAmp){ return mix(0.5, 1.8, smoothstep(15.0, 18.6, elevAmp)); }
// TANGENT-ADJUSTED cube->sphere warp: face-local metres -> warped metres (near-uniform cell area).
// Edge s=+-1 -> tan(+-pi/4)=+-1 (identity) => cross-face shared edges meet exactly (seam-safe).
highp vec2 faceWarp(highp vec2 p){ return defRadius * tan((p / defRadius) * 0.7853981634); }   // W7: ~6.4e6 m result, highp
// PERLIN-EVERYWHERE lever -- shared by the composeHeight elevation term (VS/PROBE) and the FS albedo
// overlay, so it must be declared in ALL stages (outside the VS/PROBE guard below).
uniform float uDetailOverlay;   // amplitude lever (user-tuned 6; 0 = off; __detailOverlay)
uniform float uGrid;             // interior cells per tile edge (GRID=16); for mesh-based vertex normals
uniform float uNrmStepM;        // lit-normal FD step in metres (150); uniform-fed to defeat FXC constant folding
#if defined(_VERTEX_) || defined(_PROBE_) || defined(_HEIGHTBAKE_)
highp float broadShapeM(vec3 dir, float reliefMul, float ridgeMul){   // W7: returns metres (~13000) -> highp
  if (hasHpf == 0) return 0.0;
  float mtnAmp = 1.0;   // mountain-amplitude (was a window.__mtnAmp uniform; inlined at neutral 1.0)
  vec3 d = normalize(dir);
  highp float amp = 6500.0, freq = 0.75, sum = 0.0;   // W7 highp ISLAND: amp/freq(~49152)/sum(~13000) overflow fp16
  int octMax = (uOctMax > 0) ? uOctMax : 12;   // runtime bound (FXC unroll-defeat); 12 when unset
  for (int o=0; o<octMax; o++){   // W9: 14->12 octaves, drop the finest 2 (wavelengths <2km) UNCONDITIONALLY
    // FINE-OCTAVE 5x FREQ (user 2026-06-06: the high-freq normals-affecting GROUND bump should be 5x
    // SMALLER). The o>=6 fine octaves ARE that bump (they drive the VS lit normal via the broadShapeMD
    // gradient bEx/bEy; vtxDisplace + rock dN are separate, confirmed by live __vtxDetail toggle). Sample
    // the fine band at 5x frequency so the bump grain is 5x finer; the o<6 base octaves (silhouette +
    // hypsometry, CLI-validated) keep freq untouched. MUST match broadShapeMD exactly (geometry+normal).
    highp float sf = (o >= 6) ? freq * 0.667 : freq;   // W7: freq island. fine/ridge band 3x WIDER (user 2026-06-10 'mountains noise still narrow, 3x wider'): *2.0 -> *0.667 = the visible o>=6 ridge texture (13-104km) widens 3x
    float nn = snoise3(d*sf);
    if (o >= 6) {
      float r = (1.0 - abs(nn)) * 2.0 - 1.0;     // rounded ridged crests for mountain belts
      nn = mix(nn, r, ridgeMul * 0.8);
      nn *= reliefMul;                            // per-biome amplitude on regional+fine octaves
      nn *= 1.36;                                 // FINE-OCTAVE INTENSITY -- 4x (user 2026-06-09 '4x our high
                                                  // frequency perlin noise intensity'): 0.34 -> 1.36, net
                                                  // 1.36 * uHiFreqCut 0.5 = 0.68 (was 0.17). The o>=6 fine
                                                  // octaves are the high-freq normals-affecting ground bump
                                                  // (user-confirmed 2026-06-06). uHiFreqCut(0.5) stays the
                                                  // live trim dial. Silhouette octaves o<6 untouched.
      nn *= uHiFreqCut;                           // hi-freq cut, default WIDENED 0.25->0.5 (see gl-render):
      // apply the SAME fine-octave attenuation that broadShapeMD uses, so the COLLISION/probe height
      // (sampleGroundM runs this scalar broadShapeM) matches the RENDERED geometry (broadShapeMD). Before
      // this, only broadShapeMD cut the fine band -> the collision height diverged from the surface by the
      // fine-octave amount. uHiFreqCut default 0.25 = the user's 4x reduction of all hi-freq elevation noise.
    }
    sum += amp * nn;
    amp *= (o < 6 ? 0.66 : 0.80); freq *= 2.0;   // macro decay 0.66; fine-octave decay 0.74->0.80 (widen, 2026-06-08)
  }
  // PEAK-LIFT, MOUNTAIN-BELT GATED: lift the high end so ridges become tall peaks; plains flat.
  highp float hi = max(sum - 500.0, 0.0);            // W7: metres
  float peak = smoothstep(1200.0, 5000.0, hi);
  float liftK = 0.06 + 0.16 * clamp(reliefMul, 0.0, 1.7);   // 2x LESS elevated (user 2026-06-10: halved 0.12+0.32 -> 0.06+0.16)
  sum += hi * peak * liftK;
  // MOUNTAIN-BELT MASSIF: a low-freq range-scale envelope lifts the belt interior as a coherent
  // landmass; the ridged mid octaves above supply the rugged peaks ON TOP of the raised base.
  float belt = clamp((reliefMul - 0.45) / 1.25, 0.0, 1.0);
  if (belt > 0.0 && hi > 0.0) {
    // (massif envelope freqs restored to 4.0/7.0 -- they set continental belt PLACEMENT, not the visible
    // ridge spacing; the 3x-wider knob is the o>=6 fine band above, not this envelope.)
    float e0 = snoise3(d * 2.0 + vec3(11.0, 3.0, 7.0)) * 0.5 + 0.5;
    float e1 = snoise3(d * 3.5 + vec3(2.0, 9.0, 4.0)) * 0.5 + 0.5;
    float modu = 0.7 + 0.3 * clamp(e0 * 0.7 + e1 * 0.3, 0.0, 1.0);
    float landGate = smoothstep(0.0, 800.0, hi);
    float base = belt * landGate * modu;
    sum += base * 5200.0 * mtnAmp;               // 2x LESS elevated (user 2026-06-10: halved 10400 -> 5200)
    // STEEP RUGGED PEAKS: a high-base-freq (~30km) ridged stack concentrated into steep crests.
    if (base > 0.01) {
      // pf 100 REVERTED to 200 (user 2026-06-11 'most of the terrain is flat now'): the halved peak
      // frequency flattened the 4x mountains planet-wide. The 'slopes everywhere' diagnosis it served
      // conflated the braided TEXTURE patches (rockSlope breakup noise, since deleted) with real slope;
      // with the braids, the dark raw-photo far field, and the UDN normal-frame bug all fixed, the
      // original steep terrain stands.
      highp float pf = 50.0; float pa = 1.0, ps = 0.0, pn = 0.0;   // W7: pf feeds the noise lattice -> highp. pf 100->50 (user 2026-06-14: halve the mountain frequency = 2x wider peak massifs)
      int pkOcts = (uPeakOcts > 0) ? uPeakOcts : 3;
      for (int o = 0; o < pkOcts; o++) {
        ps += pa * (1.0 - abs(snoise3(d * pf + vec3(3.3, 7.7, 1.1))));
        pn += pa; pa *= 0.65; pf *= 2.13;
      }
      float crest = ps / pn;
      sum += base * pow(crest, 4.5) * 8400.0 * mtnAmp;   // pow 4.5: sharp steep peaks that engage the rock-face slope gate (3.5 rounded them flat -- user 2026-06-14 regression revert)
    }
  }
  // SOFT ELEVATION CEILING (mob-w8-ceiling 2026-06-08: 'peaks come to points, no flat plateau'). Even at
  // CK=6500/CMAX=13000 the tanh(x/(CMAX-CK)) knee was tight enough that the high band still rolled over onto
  // a soft plateau and needed a *1.15 post-scale crutch. WIDEN THE KNEE: divide the excess by a fixed 8000m
  // so the tanh stays in its near-linear region far longer -- peaks keep climbing to points instead of
  // asymptoting to a ceiling -- and lift CMAX to 15000 so the asymptote sits well above any real peak. With
  // the wider knee the high band no longer collapses, so the *1.15 scale-up crutch is DROPPED.
  {
    highp float e = sum - 500.0;                     // W7: metres
    // GLOBAL ELEVATION SCALE 0.85 (user 2026-06-13).
    // 723m, relief only ~4km -> mountains invisible/clamped-looking, NOT a top clip). 0.65 expands the
    // range so mountains stand out: p50 ~1370m, p99 ~8920m, MAX ~11600m, relief ~7.5km. Scales the WHOLE
    // relief uniformly (base octaves + belt massif + peaks) on the excess-over-900. Max 11.6km still well
    // under the ceiling (CK 6500 / CMAX 42000, knee /26000) so no clip.
    e *= 0.85;
    // CEILING RAISED + KNEE WIDENED for the 4x mountaintops (user 2026-06-09: 'must not hit the ceiling
    // anywhere, run under the normal max + elevate the maximum'). MEASURED 4x-hf+4x-mtn pre-ceiling max
    // = 28550m (p99.9 21457m); CMAX 42000 sits 1.47x above it = no peak ever approaches the asymptote, and
    // the knee /26000 keeps tanh near-LINEAR through the whole real range (at the 22050m excess top,
    // tanh(0.85)=0.69 -> peaks come to POINTS, never the flat plateau the tight knee caused).
    float CK = 6500.0, CMAX = 42000.0;
    if (e > CK) { highp float x = e - CK; e = CK + (CMAX - CK) * tanh(x / 26000.0); }
    return e;
  }
}
// FD-GRADIENT VARIANT (FPS lever, measured: browser-9 low-alt VS=53ms/95% of frame). The lit-normal
// finite-difference in the VS samples broadShapeM at +/- a 2km world step; octaves whose wavelength is
// BELOW the FD step (~the finest 4 of the 14, wavelength < ~2km) cannot be resolved by a 2km FD -- they
// only add sub-step noise that smooths/aliases out. So this variant runs 10 (not 14) macro+mid octaves
// and a 3-octave (not 5) steep-peak stack, keeping the FULL massif/peak-lift/ceiling AMPLITUDE so the FD
// slope still tracks the real mountain geometry (the broadShapeLowM regression was dropping that
// amplitude, not the sub-step octaves). It is a GLOBALLY-FIXED world-dir function => LOD-invariant, no
// tile-edge seam (the refuted per-tile/LOD octave fade diverged at shared edges; this does not). Used
// ONLY for the two FD taps; the displaced HEIGHT still uses the full broadShapeM (full detail on the
// geometry, slightly-coarsened slope on the lit normal -- the FS dFdx normal carries the fine relief).
// ANALYTIC-DERIVATIVE broadShapeM: returns vec4(height, dHeight/dDir in WORLD-DIR space). Same field
// as broadShapeM, but every octave also accumulates its exact gradient via snoise3D, and the smooth
// post-terms (gamma compression, peak-lift, tanh ceiling) scale the gradient by their scalar
// derivative (chain rule). The FS builds the lit normal from this gradient instead of a finite
// difference -> exact relief shading at EVERY scale, no fine-octave aliasing (the deck flat-clay fix).
// The massif/steep-peak ridged terms contribute their dominant gradient; sub-tap detail rides the
// octave gradient. Returns the SAME height as broadShapeM (verified by construction).
// broadShapeFD (the reduced-octave FD-tap variant) REMOVED 2026-06-05: the analytic-derivative
// broadShapeMD replaced the 3-tap finite-difference lit-normal entirely, so the reduced-octave FD
// helper is dead. One comprehensive gradient source, net-smaller shader (one-system cleanup).
highp float broadShape(vec3 dir){ return broadShapeM(dir, 1.0, 0.0); }   // W7: metres -> highp
highp float broadShape(vec3 dir, float tileM){ return broadShape(dir); }

// PER-VERTEX micro-relief (rolling-ground bump). MOVED into the shared VS/PROBE region
// (THC-Normal W1) so geometry (VS), collision (PROBE) and the height-bake all call the SAME field --
// no parallel re-derivation. fp = face-local warped metres (continuous across tiles), tileM = quad
// size for the Nyquist octave fade, rugged = anchor-driven amplitude. Pure value (no gradient).
float vtxDisplace(highp vec2 fp, float tileM, float rugged){   // W7: fp = face-local metres ~6.4e6 -> highp; fp/wl feeds the noise lattice
  if (vtxDetail <= 0.0 || rugged <= 0.0) return 0.0;
  float wl0 = 960.0;         // coarsest fine octave wavelength (m) -- 3x WIDER / less frequent (user 2026-06-09:
                             // 'make this high frequency elevation noise 3x wider, its very nice but too small').
                             // Was 320m; 320*3=960. The noise the user sees on ALL land widens 3x (finest octave
                             // 30m vs 10m). The mountain-gated erosive copy below rides the same widened basis.
  float amp0 = 11.0;         // 8->11m: wider features carry a touch more amplitude to stay as visible as the old bump
  float ridge = smoothstep(1.3, 1.75, rugged);   // only the highest mountain belts fold to ridges
  float sumF = 0.0, sumR = 0.0, a = amp0, wl = wl0;
  // PER-PATCH STAIR-STEP FIX (user 'stair steps were per patch, not per vert'): the old Nyquist fade keyed
  // cellM=tileM/16 / nyqWl on tileM = the LEAF SIZE (a per-patch constant). Two ADJACENT leaves at different
  // LOD (tileM differs 2x) then weighted the octaves DIFFERENTLY, so their micro-relief disagreed at the
  // shared edge by a constant -> a per-patch HEIGHT STEP at every LOD boundary. Make vtxDisplace a FIXED
  // octave set, a PURE function of the world-pos fp with NO tileM dependence -> every leaf computes the
  // IDENTICAL height at a given point = C0 across all LOD boundaries by construction (no per-patch step).
  // A coarse leaf's 16-cell mesh simply under-samples the finest octaves (smooth interpolation, not a step);
  // the amplitude is small (amp0 8m * uHiFreqCut 0.25 * ruggedAmp) so any residual aliasing is minor and
  // far less visible than the per-patch steps it removes. (tileM kept in the signature for callers.)
  int vbOcts = (uVtxBaseOcts > 0) ? uVtxBaseOcts : 6;
  for (int o=0; o<vbOcts; o++){
    float n = vnoise2(fp / wl);             // [-1,1]
    sumF += a * n;                           // smooth fBm
    float r = 1.0 - abs(n); r *= r;          // ridged: sharp crest at n=0
    sumR += a * (r * 2.0 - 1.0);
    a *= 0.55; wl *= 0.5;
  }
  float sum = mix(sumF, sumR * 0.9, ridge);
  // MOUNTAIN EROSIVE DETAIL (user 2026-06-09: 'take the high-freq elevation noise, make it 3x WIDER (less
  // frequent) -- nice but too small -- and add it to all mountainous zones at a little less intensity as
  // part of the mountain pattern, to simulate erosive detail'). SAME vnoise2(fp/wl) basis as the bump above
  // (so it reads as the same noise the user likes), but wl0 = 3*320 = 960m (3x wider / lower frequency).
  // Gated by `mtnGate` (the rugged/mountain signal) so it ONLY lands in mountainous zones, and at LOWER
  // amplitude than the base bump (ea0 5m vs 8m) per 'a little less intensity'. 4-octave ridged-leaning fBm
  // for erosive gully/ridgeline shape. Added INSIDE vtxDisplace so it shares the LOD-invariant, seam-safe,
  // face-local path (no per-patch step) and is picked up by the composeHeight central-diff lit normal.
  float mtnGate = smoothstep(0.30, 0.70, rugged);   // ramps in as soon as the massif lifts so ALL mountains get erosive perlin (user 2026-06-14); was (0.40,0.85)
  float eros = 0.0;   // mountain erosive relief, kept SEPARATE from the uHiFreqCut trim (see return)
  if (mtnGate > 0.0) {
    float ea = 8.0, ewl = 1440.0, esumF = 0.0, esumR = 0.0;   // ea 4->8 + ewl 1920->1440: stronger, sharper rugged ravine relief so mountains read ROCKY (engage the rockSlope gate), not round/smooth (user 2026-06-14). Decoupled from uHiFreqCut so it lands full-strength.
    int veOcts = (uVtxErodeOcts > 0) ? uVtxErodeOcts : 4;
    for (int o = 0; o < veOcts; o++) {
      float en = vnoise2(fp / ewl);
      esumF += ea * en;
      float er = 1.0 - abs(en); er *= er;
      esumR += ea * (er * 2.0 - 1.0);
      ea *= 0.55; ewl *= 0.5;
    }
    eros = mix(esumF, esumR * 0.9, 0.6) * mtnGate;   // ridged-leaning erosive relief, mountain-gated
  }
  float ruggedAmp = clamp(rugged, 0.4, 1.15);
  // DECOUPLE (user 2026-06-14, workflow-found root cause): the everywhere-bump `sum` still rides the
  // uHiFreqCut altitude-blotch trim (0.25), but the mountain `eros` does NOT -- it was being silently
  // quartered (6.0->1.5m), which is why mountains read as smooth grass. eros now lands full-strength.
  return (sum * uHiFreqCut + eros) * vtxDetail * ruggedAmp;
}

// PERLIN-EVERYWHERE detail fbm (user 2026-06-10): ONE 3-octave value fbm shared by the FS albedo
// overlay and the composeHeight elevation term below (same field -> the brightness variation and the
// relief variation correlate, reading as one landform). World-dir keyed, seam-safe.
highp float detailFbm(vec3 dir) {
    float ov = 0.0, oa = 0.0;
    float fq = 75.0, am = 1.0;
    int dfOcts = (uDetailFbmOcts > 0) ? uDetailFbmOcts : 3;
    for (int o = 0; o < dfOcts; o++) {
        ov += am * snoise3(dir * fq + vec3(float(o) * 7.3));
        oa += am;
        fq *= 5.0; am *= 0.75;
    }
    return ov / oa;
}

// THC-Normal W1: the SINGLE composite height field. Lifts the inline vH accumulation out of the VS
// main() (was terrain.glsl ~801-902) into ONE value-only scalar so geometry (VS) and collision (PROBE)
// are derived from the EXACT same composition -- aligned by
// construction, no parallel mirror to drift. Returns the signed elevation h (metres) for dir0; the
// caller adds nothing (cbias is folded in here). dir0 = world dir of the sample (faceWarp'd),
// faceLocal = face-local warped metres (for vtxDisplace), tileM = quad size (Nyquist fade).
highp float composeHeight(vec3 dir0, highp vec2 faceLocal, float tileM){   // W7: faceLocal metres + returned h -> highp islands
  vec4 hpf0 = hpfSample(dir0);          // (seaBias=r, elevAmp=g, temp=b, humid=a)
  highp float cbias = hpf0.r;           // W7: seaBias metres
  float rugged = ruggedFromElevAmp(hpf0.g);
  float vDisp = vtxDisplace(faceLocal, tileM, rugged);
  // per-biome relief from the anchor sample (mirror of the old VS main morphology)
  float bTemp = hpf0.b, bHum = hpf0.a, bAmp = hpf0.g;
  // mtn band: uMtnBandWide=1 spreads the 16.8->18.6 contour to 14.5->19.5 (covers the live elevAmp
  // range ~14.5-18.9) so the belt-massif/reliefMul transition is a planet-wide gradient, not a thin
  // iso-contour snapping a 2600m bulk lift on across gentle land (workflow wrxo0rr7a topSuspect, 2600m).
  float mtn = smoothstep(mix(16.8, 14.5, uMtnBandWide), mix(18.6, 19.5, uMtnBandWide), bAmp);
  // climate reliefMul gates: uClimateRelief=1 widens both bands so the cold/wet flattening fades over
  // latitude/moisture gradients instead of a per-cell contour (wrxo0rr7a coldFlat/wetLowFlat 80m+).
  float wetLowFlat = smoothstep(mix(0.66, 0.50, uClimateRelief), mix(0.9, 1.0, uClimateRelief), bHum) * (1.0 - mtn);
  float coldFlat = (1.0 - smoothstep(mix(0.18, 0.05, uClimateRelief), mix(0.34, 0.45, uClimateRelief), bTemp));
  float reliefMul = clamp(0.45 + 1.25 * mtn - 0.30 * wetLowFlat - 0.25 * coldFlat, 0.40, 1.7);
  float ridgeMul  = clamp(mtn * 1.1, 0.0, 1.0);
  // ISLAND-TYPE VARIETY (pure fn of world dir) -- mirror of the VS main isle block.
  // uIsleWide=1 spreads the seaBias double-gate so the volcanic/atoll remix ramps in (wrxo0rr7a 2000m).
  float isleZone = smoothstep(mix(50.0, 30.0, uIsleWide), mix(350.0, 600.0, uIsleWide), cbias)
                 * (1.0 - smoothstep(mix(900.0, 600.0, uIsleWide), mix(1600.0, 2200.0, uIsleWide), cbias));
  if (isleZone > 0.0) {
    float isleType = snoise3(dir0 * 9.0);
    float volcanic = smoothstep(0.25, 0.7, isleType);
    float atoll    = smoothstep(0.25, 0.7, -isleType);
    reliefMul = mix(reliefMul, mix(reliefMul, 1.6, volcanic) * (1.0 - 0.7 * atoll), isleZone);
    ridgeMul  = mix(ridgeMul, max(ridgeMul, 0.9 * volcanic), isleZone);
  }
  highp float bShape = broadShapeM(dir0, reliefMul, ridgeMul);  // W7: metres
  highp float h = cbias + bShape;                                // W7: composite elevation metres (~13000)
  // REALISTIC BATHYMETRY (user 2026-06-11 'the depth under the water doesnt seem right -- the
  // landscape should continue underwater realistically'): raw cbias+bShape plunges at land-relief
  // gradients, so the seabed hit kilometre depths within sight of the beach. Real margins have a
  // wide gently-sloped CONTINENTAL SHELF (0..~120m over tens of km), then a steeper continental
  // SLOPE down to the abyssal plain (which keeps its raw depth). Monotone remap of h, pure fn of
  // the field -> seam-safe, LOD-invariant, and the collision probe shares it by construction.
  if (h < 0.0) {
      highp float d = -h;
      h = -(min(d, 500.0) * 0.24 + max(d - 500.0, 0.0) * 1.19);
      h = max(h, -11000.0);   // cap depth at Mariana Trench (~11km)
  } else {
      // LAND COASTAL SHELF (user 2026-06-14: 'beaches not wide enough'): the underwater shelf above
      // gives a gentle seabed; mirror it on the LAND side so the coast rises GENTLY from the waterline
      // = a wide beach. h*h/S is flat at the waterline (derivative 0) and identity at h=S, so high land
      // is unchanged (no drowning) and only the low coastal band is stretched horizontally. Pure fn of
      // the field -> seam-safe + the collision probe shares it. GUARD: a stale/unset uniform (0) would
      // disable the shelf -> default 600 so it always applies. window.__beachShelf dials S live.
      highp float bShelf = uBeachShelfM > 1.0 ? uBeachShelfM : 600.0;
      // C1-CONTINUOUS shelf (user 2026-06-14 'hard shading line where landscape meets beach'): the old
      // h*h/S met the identity at h=S with SLOPE 2 (vs 1) -> a derivative kink -> the slope-keyed shading
      // (rock/AO/material) SNAPPED into a hard line at the shelf top. f = (h*h/S)*(2 - h/S) keeps f(0)=0,
      // f'(0)=0 (flat at the waterline = wide beach) AND f(S)=S, f'(S)=1 (smooth join to natural land).
      if (h < bShelf) h = (h * h / bShelf) * (2.0 - h / bShelf);
  }
  // displacement now continues UNDERWATER (the old land-only gate served the flat-clamped ocean,
  // gone since 026d530): the seabed carries the same micro-relief as land = realistic continuation.
  h += vDisp;
  // PERLIN-EVERYWHERE ELEVATION (user 2026-06-10 'it must also affect elevation'): the same detailFbm
  // the FS albedo overlay shows, as real relief (~30m per lever unit -> ~180m at the user-tuned 6).
  // Shore-gated (fades in over the first 250m of land) so the coastline and the flat water planes are
  // untouched and no noise islets pop offshore. The VS FD lit-normal picks it up automatically.
  h += detailFbm(dir0) * uDetailOverlay * 30.0 * step(0.0, h);
  // VS-PROFILING GATE (2026-06-14): uVsCheap>0.5 returns BEFORE all the carves (valley/lake/river/
  // canyon/cliff/dune) so gpuTimer can A/B the per-vertex CARVE cost (the doctrine's suspected
  // dominant term). Transient profiling toggle (window.__vsCheap); 0 = full path (default).
  if (uVsCheap > 0.5) return h;
  // FLAT-AREA VALLEY NETWORKS + LAKES (user 2026-06-13): incised valley systems in low-relief
  // plains. Replaces the old noise bumps with a ridge-field valley network for connected
  // linear depressions and lakes that fill the valley bottoms. Fades to zero by reliefMul ~0.5.
  float flatGate = max(0.0, 1.0 - reliefMul * 2.0);
  float valleyV = 1.0 - inciseRidgeField(dir0, 31.0, 2.0);
  float valleyVal = smoothstep(0.25, 0.80, valleyV) * flatGate * uDetailOverlay * 24.0;
  h -= valleyVal;
  // LAKE CARVE + flat-water plane
  float lakeWetV; float lakeCarveRaw = lakeCarveM(dir0, lakeWetV);
  // Flat-area lakes: lower humidity gate so plains valleys fill with lakes
  float lakeGateLo = mix(0.60, 0.50, uCarveWide) - 0.12 * flatGate;
  float lakeGate = smoothstep(lakeGateLo, mix(0.85, 0.95, uCarveWide), hpf0.a) * step(0.0, h);
  float lakeCarveV = lakeCarveRaw * lakeGate;
  h += lakeCarveV;
  float lakeWet = lakeWetV * lakeGate;
  if (lakeWet > 0.0) { float waterLevel = max(h, 0.0) - 25.0; h = mix(h, waterLevel, lakeWet); vDisp *= (1.0 - lakeWet); }
  // RIVER + CANYON incision (clamped so coastal gorges never punch fake inland seas)
  float riverWet   = smoothstep(mix(0.30, 0.20, uCarveWide), mix(0.55, 0.65, uCarveWide), hpf0.a) * smoothstep(mix(0.20, 0.12, uCarveWide), mix(0.34, 0.46, uCarveWide), hpf0.b);
  float riverWetMask; float riverCarveV = riverCarveM(dir0, riverWetMask) * riverWet * step(0.0, h);
  float canyonDepMask; float canyonCarveV = canyonCarveM(dir0, canyonDepMask) * step(0.0, h);
  float inciseTot = riverCarveV + canyonCarveV;
  // min(...,0): the floor term goes POSITIVE for any h below -60 and was LIFTING the entire ocean
  // floor to exactly -60m -- the whole seabed was a uniform pan (root of 'depth under the water
  // doesnt seem right'; probe-witnessed minSeen -60 planet-wide). Carves still cannot punch land
  // below -60; the ocean keeps its real bathymetry.
  inciseTot = max(inciseTot, min(-60.0 - h, 0.0));           // land: h + inciseTot >= -60m; ocean: untouched
  h += inciseTot;
  // CLIFF TERRACING (mesa/butte benches) -- after carves so canyon walls + risers compose
  float cliffFaceMask; float cliffCarveV = cliffTerraceM(dir0, h, cliffFaceMask) * step(0.0, h);
  h += cliffCarveV;
  // FLAT RIVER WATER
  float riverWetLine = riverWetMask * riverWet * step(0.0, h);
  if (riverWetLine > 0.0) { float rWaterLevel = h - 20.0; h = mix(h, rWaterLevel, riverWetLine); vDisp *= (1.0 - riverWetLine); }
  // DUNES on the low sand desert
  float duneSand = smoothstep(mix(0.62, 0.50, uCarveWide), mix(0.85, 0.95, uCarveWide), 1.0 - hpf0.a) * smoothstep(mix(0.40, 0.30, uCarveWide), mix(0.58, 0.68, uCarveWide), hpf0.b) * (1.0 - smoothstep(40.0, 160.0, h));
  float duneCrest; float duneV = duneFieldM(dir0, duneCrest) * duneSand * step(0.0, h);
  h += duneV;
  return h;
}
#endif   // broadShapeM/broadShape/vtxDisplace/composeHeight: VS/PROBE (excluded from render FS, FS-1)

#ifdef _VERTEX_
layout(location=0) in vec3 vertex;   // vertex.xy in [0,1] parametric quad coord
layout(location=1) in highp vec4 iOffset;  // W7: PER-INSTANCE (ox, oy, l, level) face-local metres ~6.4e6 -> highp
layout(location=2) in float iFace;   // PER-INSTANCE cube face index 0..5
layout(location=3) in float iLayer;  // THC: PER-INSTANCE height-pool array layer for this tile (uThc on)
// THC (Tile Heightmap Cache): when uThc>0.5 the VS reads the baked composeHeight from uHeightPool
// (a 2D-array, one layer per visible tile, parametric uv = vertex.xy) instead of re-evaluating
// composeHeight 5x/vertex. uPoolLinear=0 -> manual bilinear (R32F not linear-filterable).
uniform float uThc;
uniform sampler2DArray uHeightPool;
uniform float uPoolRes;
uniform float uPoolLinear;
highp float thcSample(highp vec2 uv, float layer){
    highp vec2 t = clamp(uv, 0.0, 1.0) * (uPoolRes - 1.0);
    if (uPoolLinear > 0.5) return texture(uHeightPool, vec3((t + 0.5) / uPoolRes, layer)).r;
    highp vec2 f = floor(t); highp vec2 fr = t - f;
    highp vec2 b0 = (f + 0.5) / uPoolRes, b1 = (f + 1.5) / uPoolRes;
    highp float h00 = texture(uHeightPool, vec3(b0.x, b0.y, layer)).r;
    highp float h10 = texture(uHeightPool, vec3(b1.x, b0.y, layer)).r;
    highp float h01 = texture(uHeightPool, vec3(b0.x, b1.y, layer)).r;
    highp float h11 = texture(uHeightPool, vec3(b1.x, b1.y, layer)).r;
    return mix(mix(h00, h10, fr.x), mix(h01, h11, fr.x), fr.y);
}
out highp vec3 vWorld;   // W7 highp ISLAND: absolute world pos ~6.4e6 m (FS lighting/atmosphere) -- fp16 would jitter it
out highp float vH;      // W7: signed elevation (metres, ~13000) -- highp for the material ramp / strata / ocean depth
out highp vec3 vNrm;     // W8: world-space per-vertex analytic normal (fixed-step central diff of the FULL composeHeight). The SOLE FS lit normal -- replaces the jittery cross(dFdx,dFdy). highp to match vWorld on the ~6.4e6 m planet.
out vec3 vTexWarp;       // texture domain warp, computed ONCE in the VS (2026-06-12 'make warp as performant
                         // as possible': was 9 snoise3 PER PIXEL in the FS splat; the warp waves are >=1.8km
                         // so per-vertex evaluation + linear interpolation is visually identical at any GRID).
                         // The FS applies it EXACTLY ONCE (wt += vTexWarp * uTexWarp) -- single-application
                         // by construction, every texture layer inherits it through the shared wt.
// W5: vNrmPV/vMacroSlope/vShadeAO out-varyings deleted -- the lit normal, rock-gate slope and per-vertex
// AO are all derived in the FS from the Sobel normal now (THC sole path).
// UNIFIED-FIELD water/incision masks: computed ONCE per vertex from the same carve fields that cut
// the geometry, then INTERPOLATED to the FS. The FS no longer re-evaluates the sharp ridged carve
// fields per-pixel (that was a separate high-freq evaluation that aliased = the biome-localized
// moire the user reported). One field, sampled at the vertices, smooth in between.
out float vLakeWet;     // lake open-water mask (carve basin)
out float vRiverWet;    // river thalweg wet line
out float vWaterDepth;  // metres the flat inland-water plane sits ABOVE the local terrain floor (>0 = submerged)
out float vCanyonDep;   // canyon gorge depth [0,1]
out float vCliffFace;   // cliff/escarpment riser face [0,1] (1 = steep terrace face)
out float vDuneCrest;   // dune crest [0,1]
out float vLevel;       // quad LOD level (per-instance iOffset.w) for the patches debug view
out vec2  vGrid;        // per-quad parametric mesh coord [0,1] (for the wireframe overlay)
out vec4 vClimate;      // (seaBias, elevAmp, temp, humid) sampled ONCE per vertex from the HPF
                        // texture + INTERPOLATED, so the FS never reads the HPF texture per-pixel
                        // (that per-pixel texture read showed the HPF texel grid as UV lines/moire
                        // up close, biome colours following the blocky cells -- user: 'definitely UV').

// cube face index -> face-local->world rotation (col0=U, col1=V, col2=center). MUST match the JS
// FACE_FRAME in planet-orchestrator.js + render localToWorld3(). Built per-instance from iFace so
// the whole visible leaf set draws in ONE instanced call.
mat3 faceFrame(float f){
    int i = int(f + 0.5);
    if (i==0) return mat3( 0.0,0.0,-1.0,  0.0,1.0,0.0,   1.0,0.0,0.0);   // +X
    if (i==1) return mat3( 0.0,0.0, 1.0,  0.0,1.0,0.0,  -1.0,0.0,0.0);   // -X
    if (i==2) return mat3( 1.0,0.0,0.0,   0.0,0.0,-1.0,  0.0,1.0,0.0);   // +Y
    if (i==3) return mat3( 1.0,0.0,0.0,   0.0,0.0, 1.0,  0.0,-1.0,0.0);  // -Y
    if (i==4) return mat3( 1.0,0.0,0.0,   0.0,1.0,0.0,   0.0,0.0,1.0);   // +Z
    return            mat3(-1.0,0.0,0.0,  0.0,1.0,0.0,   0.0,0.0,-1.0);  // -Z
}

// continental elevation bias (meters) from the HPF (shared hpfSample), as a continuous
// function of world dir -> replaces the old hardcoded sin/cos lobe with the editable field.
highp float continentalBias(vec3 dir) { return hpfSample(dir).r; }   // W7: R = seaBias (meters) -> highp

// ---- CONTINUOUS BROAD+MID SHAPE (LOD-uniformity fix). The
// per-level g_noiseAmp cascade adds a DIFFERENT noise draw at each LOD, so consecutive levels
// look different (1500km vs 1200km "way different"). The fix: source the broad+regional SHAPE
// from ONE continuous fBm of WORLD DIRECTION, sampled the same at every LOD -> a finer level is
// a denser SAMPLE of the SAME field, divergence ~0 BY CONSTRUCTION (CLI continuous-field-proof
// meanDiv 0.02-0.08 vs cascade 0.52). Because it is a pure function of world dir it is:
//   - LOD-invariant  -> identical in zf and zc, so CLOD blend is automatically smooth
//   - C0 across tile AND cube-face boundaries -> seamless by construction (no seam probe needed)
// The wasm cascade is demoted (g_noiseAmp L0-6 ~0) so it only adds fine sub-silhouette detail.
// (shash3/snoise3 are defined in the SHARED preamble above so the FS riverMask can use them too.)
// broad+mid fBm of unit world dir -> metres. ONE field sampled IDENTICALLY at every LOD: a finer
// LOD is a denser SAMPLE of the SAME function, so consecutive LODs are maximally similar BY
// CONSTRUCTION (CLI continuous-field-proof meanDivergence 0.02 vs cascade 0.52). Crucially there
// is NO tile-size/LOD term here -- a per-tile octave fade was tried and REFUTED live (it made the
// field itself LOD-DEPENDENT: 1500km faded out octaves a 1200km tile kept, so they diverged).
// 8 fixed octaves span continent->regional->local; baseFreq sets the coarsest continent
// wavelength; seaBias offset sets land fraction (~0.3). Validated regime A0 5000, gain 0.6.
// PER-BIOME RELIEF (reliefMul, ridgeMul): the continent BASE octaves (o0-5) are ALWAYS full so
// the land/sea silhouette + hypsometry + LOD-uniformity are untouched (they were CLI/browser
// validated). The REGIONAL+FINE octaves (o6-13) are scaled by reliefMul and folded toward RIDGES
// by ridgeMul, BOTH supplied per-point from the anchor biome (mountains: high relief + ridged;
// plains/meadow: low relief; desert: medium smooth; swamp/tundra: very low). This makes the
// SHAPE distinguish biomes (not just color) WITHOUT breaking the broad silhouette or LOD-safety
// (still a pure fn of world dir -> denser-sample-of-same-field, no per-LOD term).
// ---- PER-VERTEX micro-relief: the FINE CONTINUATION of the single terrain field, below the
// scale broadShapeM resolves at vertex rate. Keyed on ABSOLUTE world wavelengths (fixed metres),
// NOT tile size, so a finer LOD is a DENSER SAMPLE of the SAME fine field -> detail grows
// monotonically on approach and never "pops" between LODs (the old tileM-relative wavelength was
// a different draw per level, which popped, so it was disabled). Pure fn of face-local world
// position (continuous across tiles on a face -> seamless). Default ON: this is what gives a
// close-up patch real relief instead of sitting flat inside one broad feature.
// vtxDetail/vhash/vnoise2/vnoise2D MOVED to the common preamble (THC-Normal W1) so composeHeight() can
// call them; vtxDisplace MOVED to the shared VS/PROBE region. (were here.)
// anchor elevAmp -> rugged multiplier. MEASURED LIVE (browser-1887): rendered elevAmp clusters
// ~8.0-8.5 (NOT the full 7.5-10.5 band range), so the old smoothstep(7.8,9.6) damped EVERYWHERE
// to ~0.2 -> the whole planet read flat. Recalibrate to the actual distribution: map [7.9,8.7]
// so typical terrain gets meaningful relief (~0.5-1.8) and only genuinely-low belts soften,
// high belts amplify into mountains. Floor 0.5 keeps plains gently rolling, not dead-flat.
// elevAmp range is REALLY ~29-40 (browser-4626), not ~8 -- the old smoothstep(7.9,8.7) saturated to
// 1.8 everywhere (uniform rugged -> uniform bumpiness, no plains/mountain differentiation). Map the
// real range so plains (low elevAmp) are gently rolling and only the high tail is rugged.
// ruggedFromElevAmp + faceWarp MOVED to the common preamble (THC-Normal W1, shared by composeHeight).

void main() {
    // PER-INSTANCE deform params (one instanced draw over the whole leaf set): the quad origin+size
    // and its face frame come from the instance attributes, replacing the old per-quad uniforms.
    highp vec4 defOffset = iOffset;            // W7: face-local metres ~6.4e6 -> highp
    mat3 defLocalToWorld = faceFrame(iFace);   // face-local -> world rotation
    // (elevation atlas removed -- terrain height is the GPU fractal broadShapeM, no tile sample.)
    // world direction of this vertex (pre-deform) for the continental mask.
    vec3 dir0 = normalize(defLocalToWorld * vec3(faceWarp(vertex.xy * defOffset.z + defOffset.xy), defRadius));  // W7: faceWarp/defRadius highp -> pre-normalize ~6.4e6 stays highp
    highp vec4 hpf0 = hpfSample(dir0);    // (seaBias=r, elevAmp=g, temp=b, humid=a) -- W7: R seaBias metres
    highp float cbias = hpf0.r;           // W7: seaBias metres
    // ANCHOR-DRIVEN morphology: the anchor elevAmp belt decides whether this region is flat
    // lowland (floodplain/meadow) or rugged mountain -- the anchor map is the primary spreader.
    float rugged = ruggedFromElevAmp(hpf0.g);
    // per-vertex micro-relief: unique height per vertex from world-continuous face-local pos
    // (fixes the flat 2x2 atlas-texel blocks; gives sub-mesh-cell relief). Only on land.
    highp vec2 faceLocal = faceWarp(vertex.xy * defOffset.z + defOffset.xy);   // W7: warped metres ~6.4e6 -> highp
    float vDisp = vtxDisplace(faceLocal, defOffset.z, rugged);
    // CONTINUOUS broad+mid SHAPE (LOD-uniformity fix): one world-dir fBm sampled the SAME at
    // every LOD -> no per-level shape divergence. PER-BIOME RELIEF: scale the regional+fine octaves
    // (continent base untouched -> silhouette/hypso/LOD-uniformity preserved) by an anchor-derived
    // reliefMul + ridgeMul so the SHAPE distinguishes biomes: mountains (high elevAmp) = strong +
    // ridged; meadow/plains = gentle. All pure fn of world dir -> still seam-safe + LOD-invariant.
    float bTemp = hpf0.b, bHum = hpf0.a, bAmp = hpf0.g;
    // MOUNTAIN-BELT GATE on the BAKED elevAmp range (~14.5-18.9): the high tail (~17%) is the
    // mountain belt; the rest is genuine plains so the mountain biome is a differentiated minority.
    float mtn = smoothstep(16.8, 18.6, bAmp);                     // mountain-belt weight 0..1
    float wetLowFlat = smoothstep(0.66, 0.9, bHum) * (1.0 - mtn); // swamp/wetland -> flat
    float coldFlat = (1.0 - smoothstep(0.18, 0.34, bTemp));       // tundra/ice -> flat
    float reliefMul = clamp(0.45 + 1.25 * mtn - 0.30 * wetLowFlat - 0.25 * coldFlat, 0.40, 1.7);
    float ridgeMul  = clamp(mtn * 1.1, 0.0, 1.0);                 // ridged crests only in mountain belts
    // ISLAND-TYPE VARIETY: small offshore swells get a per-region type (volcanic cone / low atoll)
    // from a world-dir noise so islands are not all scaled continents. Pure fn of world dir.
    float isleZone = smoothstep(50.0, 350.0, cbias) * (1.0 - smoothstep(900.0, 1600.0, cbias));
    if (isleZone > 0.0) {
      float isleType = snoise3(dir0 * 9.0);
      float volcanic = smoothstep(0.25, 0.7, isleType);
      float atoll    = smoothstep(0.25, 0.7, -isleType);
      reliefMul = mix(reliefMul, mix(reliefMul, 1.6, volcanic) * (1.0 - 0.7 * atoll), isleZone);
      ridgeMul  = mix(ridgeMul, max(ridgeMul, 0.9 * volcanic), isleZone);
    }
    highp float bShape = broadShapeM(dir0, reliefMul, ridgeMul);  // W7: metres
    // ONE FRACTAL: the continuous world-dir field (bShape) + the continental swell (cbias) carry the
    // entire silhouette+hypsometry. The old wasm upsample-and-add cascade (zfc.x) was REMOVED -- it
    // was a SECOND shape source that diverged per-LOD (the detail-inversion root). bShape amplitude
    // was retuned (A0 6500, off -900) so this single field hits Earth hypso without the cascade.
    vH = cbias + bShape;
    // shelf/slope bathymetry remap + underwater displacement -- MUST mirror composeHeight exactly
    // (this is the FS-material running value; composeHeight is the geometry/probe height).
    if (vH < 0.0) {
        highp float dSea = -vH;
        vH = -(min(dSea, 500.0) * 0.24 + max(dSea - 500.0, 0.0) * 1.19);
        vH = max(vH, -11000.0);   // cap depth at Mariana Trench (~11km)
    } else {
        highp float bShelf = uBeachShelfM > 1.0 ? uBeachShelfM : 600.0;   // LAND COASTAL SHELF -- mirror composeHeight exactly (wide beach, user 2026-06-14); guard stale/unset uniform
        if (vH < bShelf) vH = (vH * vH / bShelf) * (2.0 - vH / bShelf);   // C1-continuous (mirror composeHeight) -- removes the beach-top slope kink / hard shading line

    }
    vH += vDisp;
    vH += detailFbm(dir0) * uDetailOverlay * 30.0 * step(0.0, vH);
    // FLAT-AREA VALLEY NETWORKS + LAKES (user 2026-06-13): incised valley systems in low-relief
    // plains -- must mirror composeHeight for FS material consistency.
    float flatGate = max(0.0, 1.0 - reliefMul * 2.0);
    float valleyV = 1.0 - inciseRidgeField(dir0, 31.0, 2.0);
  float valleyVal = smoothstep(0.25, 0.80, valleyV) * flatGate * uDetailOverlay * 24.0;
    vH -= valleyVal;
    // LAKE CARVE: carve a real basin into the elevation on WET LAND so lakes are part of the
    // fractal (no fade-in). Flat-area gate widened so plains valleys fill with lakes.
    float lakeWetV; float lakeCarveRaw = lakeCarveM(dir0, lakeWetV);
    float lakeGateLo = 0.60 - 0.12 * flatGate;
    float lakeGate = smoothstep(lakeGateLo, 0.85, hpf0.a) * step(0.0, vH);
    float lakeCarveV = lakeCarveRaw * lakeGate;
    vH += lakeCarveV;
    // FLAT WATER: inside the wet core the surface must read as flat water (user: 'water should be
    // flat'), not a noisy bowl floor. Pin vH toward a constant water level (a few m below the local
    // rim) weighted by the wet mask, and suppress the micro-relief there. The lakeCarve shoulder
    // already erodes/grades the surrounding terrain into the basin (the 'eroded margin').
    float lakeWet = lakeWetV * lakeGate;
    highp float waterPlane = 0.0; highp float floorBefore = 0.0; float haveWater = 0.0;   // W7: waterPlane/floorBefore are metres (~13000); their small DIFFERENCE (vWaterDepth) needs highp to avoid cancellation
    if (lakeWet > 0.0) {
        highp float waterLevel = max(vH, 0.0) - 25.0;     // flat plane just below the rim
        floorBefore = vH;                            // carved bowl floor before flattening
        vH = mix(vH, waterLevel, lakeWet);
        waterPlane = waterLevel; haveWater = lakeWet;
        vDisp *= (1.0 - lakeWet);                    // no micro-bumps on the water surface
    }
    // RIVER + CANYON INCISION into the elevation (user: content makes up part of the elevation as
    // erosion works). Both are pure world-dir carves (LOD-invariant, no fade). River: wet/temperate
    // land. Canyon: arid + elevated (h>~60m) land, opposite climate to rivers. Gated to land
    // (step(0,vH)) so they never punch the ocean or create fake seas. The incision is bounded so it
    // does not drive coastal land below sea level (clamp the post-carve vH floor handled implicitly
    // by the small depths + land gate; coastline hypso re-witnessed in incision-hypso-landfrac-gate).
    float riverWet   = smoothstep(0.30, 0.55, hpf0.a) * smoothstep(0.20, 0.34, hpf0.b);   // moist, not frozen
    float riverWetMask; float riverCarveV = riverCarveM(dir0, riverWetMask) * riverWet * step(0.0, vH);
    float canyonDepMask; float canyonCarveV = canyonCarveM(dir0, canyonDepMask) * step(0.0, vH);
    // NO-SUB-SEA-COAST GUARD (user 2026-06-02 deepen+widen): with the deepened canyon (-1400m + gullies)
    // a gorge on 200m coastal land would punch vH to ~-1200m = fake inland seas. Clamp the TOTAL incision
    // so post-carve land bottoms out at a small floor (-60m: a gorge may reach near sea level but never
    // carves a deep basin below it). Bounded against the PRE-carve vH so deep inland canyons keep full
    // depth while coastal ones are limited by their own available headroom.
    float inciseTot = riverCarveV + canyonCarveV;               // both negative (downcut)
    inciseTot = max(inciseTot, min(-60.0 - vH, 0.0));           // land floor only -- ocean keeps real depth (see composeHeight note)
    vH += inciseTot;
    // CLIFF TERRACING: snap the arid+elevated land into flat benches with steep risers (mesa/butte
    // cliff country). Gated by the SAME canyonArid mask so cliffs share canyon regions (a coherent
    // arid badlands look). The snap delta is added to vH; cliffFaceMask (->1 on a riser face) goes to
    // the FS for strata banding + steep-rock material. Applied AFTER carves so canyon walls + cliff
    // risers compose. step(0,vH) keeps it on land.
    float cliffFaceMask; float cliffCarveV = cliffTerraceM(dir0, vH, cliffFaceMask) * step(0.0, vH);
    vH += cliffCarveV;
    // FLAT RIVER WATER (user: 'rivers/lakes should NOT be bumpy'): like lakes, the river thalweg
    // surface must read as flat water, not the bumpy micro-relief floor. Pin vH down to the channel
    // water level + suppress the micro-displacement on the wet line. riverWetMask is the thalweg mask.
    float riverWetLine = riverWetMask * riverWet * step(0.0, vH);
    if (riverWetLine > 0.0) {
        highp float rWaterLevel = vH - 20.0;         // W7: metres
        highp float rFloorBefore = vH;
        vH = mix(vH, rWaterLevel, riverWetLine);
        vDisp *= (1.0 - riverWetLine);               // no micro-bumps on the river surface
        // record the deeper/stronger of lake|river as the water surface for vWaterDepth
        if (riverWetLine > haveWater) { waterPlane = rWaterLevel; floorBefore = rFloorBefore; haveWater = riverWetLine; }
    }
    // DUNES on the SAND DESERT (very dry + warm + LOW land): rolling dune relief replaces harsh rock
    // here (ref: webgl-dunes). Gated opposite to canyons (which want elevated arid plateaus) -- dunes
    // ride the low desert floor. World-dir field -> LOD-invariant.
    float duneSand = smoothstep(0.62, 0.85, 1.0 - hpf0.a) * smoothstep(0.40, 0.58, hpf0.b) * (1.0 - smoothstep(40.0, 160.0, vH));
    float duneCrest; float duneV = duneFieldM(dir0, duneCrest) * duneSand * step(0.0, vH);
    vH += duneV;

    // PER-VERTEX SEAMLESS NORMAL = (-dz/dx, -dz/dy, 1) in the tile tangent frame. After the
    // one-fractal collapse the old cascade-atlas finite-difference term is gone (it was identically
    // zero), so the lit normal is built purely from the micro-relief (dEx), broad-shape (bEx) and
    // carve (rcEx) gradients below -- those ARE the one field.
    // Include the per-vertex micro-displacement gradient so the new relief is LIT (not just
    // geometrically displaced). MOIRE-ON-DESCENT FIX (user 2026-06-01h: fine green speckle/moire at
    // closeup, GONE with vtxDetail off): the FD step was ONE MESH CELL (defOffset.z/24), which
    // SHRINKS as tiles get fine on descent -> the micro-relief lit normal captured ever-higher-freq
    // slope cell-to-cell = shimmer/moire (the same shrinking-step trap the broadShape FD warns of).
    // Use a FIXED ~600m world step so the micro-relief SHADING reflects a stable slope at every
    // altitude; the geometry still displaces per-vertex (vH), the normal just stops chasing the cell.
    // W5: the per-vertex lit-normal gradient block (dEx/dEy from vtxDisplaceD; bEx/bEy/rcEx/rcEy/fEx/fEy
    // from broadShapeMD + the cbias/carve central-differences) is DELETED -- it only fed the now-deleted
    // vNrmPV/vMacroSlope assembly. THC's per-pixel Sobel is the sole lit normal. The carve VALUES (h) are
    // computed above (~906-) and untouched; only their gradient finite-differences are gone.
    // W5: the entire per-vertex normal/AO assembly (vMacroSlope, vShadeAO crease+micro AO, slopeGain, the
    // vNrmPV true-gradient sum) is DELETED -- THC is the sole path, so the lit normal is the FS Sobel of
    // heightPool and the rock-gate slope + per-vertex AO are recomputed in the FS from that Sobel normal.
    // The gradient taps (dEx/dEy from vtxDisplaceD, bEx/bEy/rcEx/rcEy/fEx/fEy from broadShapeMD) that fed
    // this block are removed upstream; only the SCALAR height fns (broadShapeM/vtxDisplace) remain, for the
    // procedural h fallback when a leaf has no cached tile.

    highp float R = defRadius;   // W7: ~6.4e6 m -> highp
    // ONE HEIGHT FUNCTION: every vertex gets the procedural composeHeight() (broadShapeM + cbias + carves
    // + vtxDisplace, with lake/river water-plane flattening). vH (FS material/strata, computed above as the
    // water-flattened running value) stays for the FS; the GEOMETRY height h is the single composeHeight.
    // WATER GATE (perf sweep 2026-06-11): on the water pass the geometry height is pinned to sea level
    // (hR=0 below) and the normal is radial (:914), so composeHeight's result is never consumed --
    // skip the 12-oct eval for ~540 water instances/frame. Uniform-coherent branch, zero fidelity change.
    // SINGLE-INSTANCE FD TAPS (2026-06-12, THE AMD ROOT -- witnessed by __flatNormal A/B: forcing the
    // radial normal turned the whole 'rock patch' region back into smooth grass, so vNrm was the broken
    // carrier). FXC inlines composeHeight per CALL SITE and optimizes each copy differently; the center
    // and offset taps then disagree by tens of metres on FLAT ground -> fake slope -> rock material +
    // slope-AO darkness + dead normals (d3d11-only; vulkan compiles one consistent version). Evaluating
    // ALL THREE taps through ONE runtime-bounded loop forces FXC to emit a SINGLE composeHeight instance,
    // so whatever approximations it picks cancel exactly in the differences. uNrmStepM>0.0 keeps the
    // bound non-constant (same defeat class as uOctMax).
    // VERTEX NORMALS: interior vertices use the mesh-based edge cross product (ordinary
    // face normals). Tile-edge vertices use the tangent-frame finite difference so both
    // sides of a seam share the same dir0/tangent frame -> consistent normals, no row artifact.
    highp float hN0 = 0.0;
    highp vec3 vN = dir0;
    if (uIsWater < 0.5 && uThc > 0.5) {
        // THC FAST PATH: height + symmetric central-diff normal from the baked pool. 5 bilinear texture
        // taps replace 5 full composeHeight evals (the VS-bound cost). The 4 normal taps use the SAME
        // parametric step as the composeHeight path; the dir at each tap is a cheap faceWarp+normalize
        // (no field eval). uNrmStepM-scaled step => the same smoothed (low-pass) normal as the slow path.
        hN0 = thcSample(vertex.xy, iLayer);
        highp float nStepM = (uNrmStepM > 0.0) ? uNrmStepM : 300.0;
        highp float duP = clamp(nStepM / max(defOffset.z, 1.0), 1.0 / ((uGrid > 0.0) ? uGrid : 16.0), 0.34);
        highp float hPU = thcSample(vertex.xy + vec2(duP, 0.0), iLayer);
        highp float hMU = thcSample(vertex.xy + vec2(-duP, 0.0), iLayer);
        highp float hPV = thcSample(vertex.xy + vec2(0.0, duP), iLayer);
        highp float hMV = thcSample(vertex.xy + vec2(0.0, -duP), iLayer);
        highp vec3 dPU = normalize(defLocalToWorld * vec3(faceWarp((vertex.xy + vec2(duP,0.0)) * defOffset.z + defOffset.xy), defRadius));
        highp vec3 dMU = normalize(defLocalToWorld * vec3(faceWarp((vertex.xy + vec2(-duP,0.0)) * defOffset.z + defOffset.xy), defRadius));
        highp vec3 dPV = normalize(defLocalToWorld * vec3(faceWarp((vertex.xy + vec2(0.0,duP)) * defOffset.z + defOffset.xy), defRadius));
        highp vec3 dMV = normalize(defLocalToWorld * vec3(faceWarp((vertex.xy + vec2(0.0,-duP)) * defOffset.z + defOffset.xy), defRadius));
        vN = normalize(cross(dPU * (defRadius + hPU) - dMU * (defRadius + hMU),
                             dPV * (defRadius + hPV) - dMV * (defRadius + hMV)));
        if (dot(vN, dir0) < 0.0) vN = -vN;
    } else if (uIsWater < 0.5) {
        hN0 = composeHeight(dir0, faceLocal, defOffset.z);   // center = the geometry height h
        // VERTEX NORMAL = CENTRAL DIFFERENCE in PARAMETRIC MESH SPACE over the FULL composeHeight (2026-06-14
        // jagged-normal fix). Two earlier methods both jagged: (a) interior FORWARD mesh-cell cross product
        // = each vertex got its forward triangle's FACE normal (faceted) at a vertex-spacing step (noisy);
        // (b) the tangent-frame FD passed the SAME faceLocal to every tap, so vtxDisplace CANCELLED in the
        // differences -> the normal ignored the bumps the geometry has -> smooth normal on a bumpy mesh =
        // facets show through. THIS evaluates the full field (incl vtxDisplace) at +/-du/+/-dv in parametric
        // space (faceWarp gives BOTH the dir and faceLocal at each offset, so vtxDisplace varies correctly)
        // and takes a CENTRAL (symmetric) world-space cross product = smooth AND matches the displaced
        // surface. ONE formula for every vert (no interior/edge split = no discontinuity); seam-safe because
        // adjacent same-LOD tiles sample the identical world offsets (the field is a pure fn of world pos).
        // NORMAL SMOOTHING (user 2026-06-14: 'angular normals = no vertex smoothing'): low-pass the
        // per-vertex normal by taking the central difference over a fixed ~METRIC step (uNrmStepM, ~300m)
        // instead of ~1 mesh cell. A cell-sized step samples the high-freq erosion/canyon relief so the
        // normal swings sharply vertex-to-vertex = angular; a fixed larger step averages it = adjacent
        // verts vary smoothly = smooth shading, at any GRID. duP = stepM / tile-span (defOffset.z), clamped
        // so it never drops below the mesh cell (would re-alias) nor exceeds ~1/3 the tile.
        highp float nStepM = (uNrmStepM > 0.0) ? uNrmStepM : 300.0;
        highp float duP = clamp(nStepM / max(defOffset.z, 1.0), 1.0 / ((uGrid > 0.0) ? uGrid : 16.0), 0.34);
        highp float hPU = 0.0, hMU = 0.0, hPV = 0.0, hMV = 0.0;
        highp vec3 dPU = dir0, dMU = dir0, dPV = dir0, dMV = dir0;
        int fdIters = (uGrid >= 0.0) ? 4 : 1;   // 4 offset taps; runtime-bounded (FXC unroll-defeat, see uOctMax)
        for (int i = 0; i < fdIters; i++) {
            highp vec2 off = (i == 0) ? vec2(duP, 0.0) : (i == 1) ? vec2(-duP, 0.0) : (i == 2) ? vec2(0.0, duP) : vec2(0.0, -duP);
            highp vec2 fl = faceWarp((vertex.xy + off) * defOffset.z + defOffset.xy);
            highp vec3 dd = normalize(defLocalToWorld * vec3(fl, defRadius));
            highp float hh = composeHeight(dd, fl, defOffset.z);
            if (i == 0) { hPU = hh; dPU = dd; } else if (i == 1) { hMU = hh; dMU = dd; }
            else if (i == 2) { hPV = hh; dPV = dd; } else { hMV = hh; dMV = dd; }
        }
        highp vec3 wPU = dPU * (defRadius + hPU), wMU = dMU * (defRadius + hMU);
        highp vec3 wPV = dPV * (defRadius + hPV), wMV = dMV * (defRadius + hMV);
        vN = normalize(cross(wPU - wMU, wPV - wMV));
        if (dot(vN, dir0) < 0.0) vN = -vN;   // keep it outward (terrain has no overhangs)
    }
    highp float h = hN0;
    // OCEAN TOP = A SEPARATE, ELEVATION-BASED SURFACE (user 2026-06-10: 'terrain should extend into
    // the ocean, the ocean top separate and elevation based'). composeHeight keeps carrying the TRUE
    // signed bathymetry (vH -> the FS depth tint/Beer-Lambert keys on it), but the RENDERED surface
    // smooth-clamps to sea level: open water is a flat plane at R (proper waterline + horizon), the
    // seafloor field lives on UNDER it as the depth signal. The smoothstep ramp (not a hard max)
    // also fixes 'the terrain curve creates a hard line at the base': the beach profile eases
    // tangentially into the waterline over the last ~60m instead of meeting it in a crease.
    // SEPARATE WATER SURFACE (user 2026-06-11): the terrain pass renders the TRUE seabed (the old
    // smoothstep(-60,60) sea-level clamp is GONE -- sand/rock bathymetry is real geometry now); the
    // water pass (uIsWater=1, second instanced draw of the same leaves) pins this mesh to sea level
    // and shades it as the animated ocean, alpha-blended over the seabed (depth test keeps it
    // behind land).
    highp float hR = (uIsWater > 0.5) ? 0.0 : h;
    // VERTEX NORMAL: central-difference (computed above) for land; radial for water.
    vNrm = (uIsWater > 0.5) ? dir0 : vN;
    // DIRECT per-vertex sphere projection (replaces the old corner-blend deform, which
    // bilinearly interpolated 4 deformed corners -> FLAT quad interior -> faceted at high GRID).
    // dir0 is THIS vertex's world direction (faceWarp'd, defLocalToWorld-mapped); place it on the
    // sphere at radius R+h and project. Every vertex curves -> round at any tessellation.
    // SKIRT: outer-ring verts (vertex.z==1, xy clamped to the true edge) drop radially below the
    // surface to form a near-vertical curtain that hides LOD T-junction cracks without painting a
    // visible flat band (the old overlap ring's artifact). Depth scales with the tile size so the
    // skirt always reaches below a coarser neighbor's surface. Hidden behind the surface from above.
    // water pass: NO skirt (user 2026-06-11) -- the surface is an exact sphere at R, so adjacent
    // LODs agree exactly and there are no T-junction cracks to hide; a skirt would only drape a
    // visible curtain through the transparent shallows.
    highp float skirt = (vertex.z > 0.5 && uIsWater < 0.5) ? max(defOffset.z * 0.12, 60.0) : 0.0;   // W7: metres (tile-size scaled) -> highp
    vWorld = dir0 * (R + hR - skirt);   // ABSOLUTE world pos (RENDER height: ocean top flat) -> FS lighting/atmosphere
    // TEXTURE DOMAIN WARP -- VS-side, DOUBLED BACK (user 2026-06-12): 225/900/3500 -> 450/1800/7000.
    {
        highp vec3 w0 = dir0 * 450.0, w1 = dir0 * 1800.0, w2 = dir0 * 7000.0;
        vTexWarp = vec3(snoise3(w0), snoise3(w0 + vec3(7.3)), snoise3(w0 + vec3(23.9))) * 1.2
                 + vec3(snoise3(w1), snoise3(w1 + vec3(13.7)), snoise3(w1 + vec3(31.1))) * 0.6
                 + vec3(snoise3(w2), snoise3(w2 + vec3(5.1)), snoise3(w2 + vec3(17.9))) * 0.3;
    }
    // CAMERA-RELATIVE PROJECTION (vertex-jitter fix): forming dir0*(R+h) at ~6.4e6m rounds to ~0.5m in
    // fp32 -> vertices quantize and JITTER as the camera moves (the same fp32 cancellation gl-render.js
    // documents for the cull, which the per-vertex draw path did NOT avoid). Build the SMALL camera-
    // relative position directly -- (dir0-defCamDir)*R is the lateral offset computed from a unit-vector
    // difference (precise, no 6.4e6 intermediate), plus the small radial terms -- and project with
    // defViewProjNoEye (no folded translate(-eye)). Result magnitude ~horizon scale, fp32-precise.
    highp vec3 vRel = (dir0 - defCamDir) * R + dir0 * (hR - skirt) - defCamDir * defCamAlt;   // W7 highp ISLAND: camera-relative projection (render height; planet-scale fp32 cancellation fix kept intact)
    gl_Position = defViewProjNoEye * vec4(vRel, 1.0);
    // UNIFIED carve masks -> FS (interpolated; the FS no longer re-evaluates the sharp carve fields
    // per-pixel). riverWetMask/canyonDepMask/duneCrest are the out-params captured above; lakeWetV
    // from the lake-carve block. Gated by the SAME climate masks the geometry used.
    vLakeWet   = lakeWetV * lakeGate;
    vRiverWet  = riverWetMask * riverWet * step(0.0, vH);
    // SUBMERGED DEPTH (user 2026-06-01i: 'carves dont line up with the water'). The inland-water
    // COLOR must land ONLY where the surface is actually at/below the flat water plane, not over the
    // whole carve mask (which includes the graded erosion shoulder ABOVE the waterline). vWaterDepth
    // = metres the water plane sits above the pre-flatten carved floor, >0 ONLY inside the true open
    // water; the FS gates ALL inland-water shading on this so blue == the flat water, banks stay land.
    vWaterDepth = max(waterPlane - floorBefore, 0.0) * step(0.001, haveWater);
    vCanyonDep = canyonDepMask * step(0.0, vH);
    vCliffFace = cliffFaceMask * step(0.0, vH);
    vDuneCrest = duneCrest * duneSand;
    vGrid = vertex.xy;   // parametric mesh-cell coord for the wireframe overlay
    vLevel = defOffset.w;   // quad LOD level -> FS patches view
    vClimate   = hpf0;   // (seaBias, elevAmp, temp, humid) -> FS reads this, not the HPF texture
}
#endif

#ifdef _FRAGMENT_
in highp vec3 vWorld;   // W7: MUST match the VS highp vWorld (world pos ~6.4e6 m) -- precision-mismatched varyings fail to link
in vec3 vTexWarp;       // VS-computed texture domain warp (halved freqs); applied once in the splat block
in highp float vH;      // W7: match VS highp vH (signed metres)
in highp vec3 vNrm;     // W8: world-space analytic normal from the VS (matches VS highp out). Sole lit normal.
in float vLakeWet;    // carve masks computed in the VS, interpolated (no per-pixel re-eval -> no moire)
in float vRiverWet;
in float vWaterDepth; // metres of submerged water (>0 = real open inland water at the flat plane)
in float vCanyonDep;
in float vCliffFace;
in float vDuneCrest;
in float vLevel;          // quad LOD level (patches view)
in vec2  vGrid;            // per-quad parametric mesh coord (wireframe overlay)
uniform float uWireframe;  // 1 = overlay the mesh grid lines (window/cam wireframe toggle)
uniform float uFsCheap;    // GPU-TIMER VS/FS attribution: 1 = short-circuit the FS to a trivial
                           // constant color immediately after the per-vertex normal is read, so a
                           // timed cheap frame measures VS+raster cost only; (full - cheap) = FS cost.
                           // Set by window.__gpuTimer's measure frame (gl-render). 0 in normal render.
in vec4 vClimate;     // (seaBias, elevAmp, temp, humid) interpolated -- FS does NOT sample the HPF texture
layout(location=0) out vec4 fragColor;

uniform vec3 sunDir;       // world-space sun direction (normalized) -- unit, mediump-safe
uniform int displayMode;   // 0 = lit, 1 = raw normals, 2 = material albedo (unlit)
uniform highp vec3 camWorld;     // W7: world-space camera position ~6.4e6 m -> highp
uniform highp float terrainR;    // W7: sphere radius ~6.4e6 m -> highp
uniform highp float oceanTime;   // W7: animation time (seconds) grows unbounded -> highp (fp16 would freeze the wave phase)
uniform float oceanAmp;    // wave amplitude scale (0..1+), drives normal perturbation
uniform float oceanChoppy; // wave directional sharpness / count weighting
uniform float oceanFoam;   // foam amount (0..1): whitecaps on steep wave slopes
                           // CONTINUOUS sampled world position (seamless across tiles), 0 = atlas normal

// ---- Animated ocean: sum-of-Gerstner-wave NORMAL perturbation in the surface tangent
// frame. We don't displace geometry (FS-only v1) -- we synthesize an animated water
// normal from a few directional waves and shade it with fresnel + sun glint + a depth
// tint. wave dirs are 2D in the local (ux,uy) tangent plane; phase advances with time.
// Returns a tangent-space normal perturbation (dx,dy) to add to the flat (0,0,1) normal.
vec2 oceanWaveSlope(highp vec2 p, highp float t) {   // W7: p = camera-relative wave coord, t = unbounded oceanTime -> highp phase
    // a handful of directional waves with varied freq/dir/speed (kept cheap)
    vec2 slope = vec2(0.0);
    // (dir.xy, wavelength_m, speed, steepness)
    const int N = 5;
    vec2 dirs[5];   float wl[5];  float spd[5];  float amp[5];
    dirs[0]=vec2( 1.0, 0.0);  wl[0]=520.0; spd[0]=1.10; amp[0]=1.0;
    dirs[1]=vec2( 0.6, 0.8);  wl[1]=310.0; spd[1]=1.35; amp[1]=0.7;
    dirs[2]=vec2(-0.4, 0.9);  wl[2]=170.0; spd[2]=1.60; amp[2]=0.5;
    dirs[3]=vec2( 0.9,-0.3);  wl[3]= 95.0; spd[3]=2.10; amp[3]=0.35;
    dirs[4]=vec2(-0.7,-0.6);  wl[4]= 47.0; spd[4]=2.80; amp[4]=0.22;
    for (int i=0;i<N;i++){
        vec2 d = normalize(dirs[i]);
        highp float k = 6.2831853 / wl[i];                 // W7: highp wave number
        highp float phase = k*dot(d,p) + t*spd[i]*k*8.0;   // W7: highp accumulated phase (unbounded t)
        // slope contribution: derivative of a sine height field -> cosine, weighted by
        // amplitude and choppiness. amp[i] tapers the higher-freq waves.
        float a = amp[i] * oceanAmp * (1.0 + oceanChoppy);
        slope += d * (cos(phase) * a * 0.06);
    }
    return slope;
}

// Procedural height+slope material ramp (placeholder until the OrthoProducer lands):
// water / shore / lowland / grass / rock / snow by signed elevation, with rock on steep
// slopes. World-continuous (pure function of h + slope) so it has no per-tile seam.
// LIVE-ADJUSTABLE biome ramp (full-adjustability via window.__gen.biome). Each color + band edge
// is a uniform defaulting to its tuned literal when the JS global is unset (no behaviour change
// until edited). bcXxx = band colors; bandEdgesLo=(shore->lowland end, lowland->grass end),
// bandEdgesHi=(rock start, rock end), snowEdges=(snow start, snow end), seaDepthM, slopeRock=(lo,hi).
uniform vec3 bcDeepSea, bcSea, bcShore, bcLowland, bcGrass, bcRock, bcSnow;
uniform vec2 bandEdgesLo;   // (lowland-blend end, grass-blend end)
uniform vec2 bandEdgesHi;   // (rock start, rock end)
uniform vec2 snowEdges;     // (snow start, snow end)
uniform float seaDepthM;    // depth (m) over which sea->deepSea
uniform vec2 slopeRock;     // (lo, hi) slope range that forces rock
uniform float uAoAmt;       // canyon/cliff ambient-occlusion strength (window.__aoAmt; 1.0 default)
uniform float uBiomeBandBias;// elevation/latitude biome-band bias strength (window.__biomeBandBias; 1.0)
// REAL-WORLD LOOK overhaul uniforms (2026-06-05 workflow wxb9n2907) -- all window.__gen-overridable.
uniform vec3  uOceanDeep;    // deep open-ocean color (near-black navy)            [0.008,0.025,0.06]
uniform vec3  uOceanShallow; // shallow-water turquoise (first metres)             [0.07,0.22,0.26]
uniform vec3  uOceanK;       // per-channel Beer-Lambert extinction (kR>>kG>>kB)   [0.030,0.012,0.0045]
uniform float uBiomeSat;     // biome-palette saturation pull toward luminance (1=full, <1 desaturate) 0.72
uniform float uVariationAmt; // intra-biome value mottle amplitude (+/-)           0.08
uniform float uHazeMul;      // aerial-perspective strength multiplier (1 = full haze, 0 = none)
// LIVE A/B ISOLATION TOGGLES (window.__rockBump / __chroma / __strata, default 1). Multiply each material
// detail layer so the user can flip one to 0 and see which layer produces the close-up uv scramble.
uniform float uFlatNormal;   // 1=force the SMOOTH analytic normal (nBand), bypassing the raw cross(dFdx,dFdy) geometric normal that scrambles on steep deck faces (A/B isolation)
uniform float uSkyFill;      // sky-ambient fill weight (was implicit 1.0)         0.45
uniform float uTerminatorGlow;// sunset reddening strength at the grazing terminator 0.5
uniform float uNightLights;  // night/shadow fill intensity (lift dark areas, not black)  1.0
uniform float uNightFloor;   // unlit-hemisphere brightness floor (dim, not black) 0.05
uniform float uTermWidth;    // terminator half-width (wider=softer twilight)      0.25
uniform float uExposure;     // pre-tonemap exposure (was 1.25)                    1.0
uniform float uLookSat;      // post-ACES saturation Look (>1 = more saturated)    1.15
uniform float uLookContrast; // post-ACES contrast Look (>1 = more contrast)       1.08
// SURFACE PHOTO-TEXTURES (user 2026-06-10 'use the textures in textures/, calculate normals from
// displacement'). Two 1024x1024x4 sampler2DArrays loaded at runtime (gl-render loadSurfaceTextures):
// uSurfAlb = sRGB color (RGB) + displacement (A), uSurfNrm = tangent normal xy (RG, 0.5-biased) +
// displacement (B). Layers: 0=grass 1=rock 2=sand 3=snow. The normals are Sobel-derived from the
// displacement JPGs at load (wrap edges, JS). SUPERSEDES the 2026-06-01e 'no detail texturing'
// directive by explicit user request. Triplanar-sampled in the FS, blended by the existing material
// gates (slopeRock / snowEdges / climate), height(displacement)-sharpened, distance-faded by pxWorld.
uniform sampler2DArray uSurfAlb;
uniform sampler2DArray uSurfNrm;
uniform float uHasSurfTex;   // 1 once the arrays are uploaded (0 = procedural-only fallback)
uniform float uTexTileM;     // world metres per texture repeat (__texTile, default 2400 -- user: 24m read as noise/rock, '100x bigger')
uniform float uTexNrmK;      // texture detail-normal strength (__texNrmK, default 0.5; keep <=1: scramble lesson)
uniform float uTexMix;       // texture albedo blend amount (__texMix, default 0.85; 0 = off)
uniform float uTexWarp;      // anti-repetition domain-warp amplitude (__texWarp, default 1.0)
uniform float uReliefShade;  // gentle-slope relief-shading exaggeration (__reliefShade, default 1.7; 1 = off)
uniform float uTexPhoto;     // raw photo-color fraction (__texPhoto, default 0 = full macro tint; user: patch must match the shade it replaces)
uniform float uTexPhotoNear; // near-field MATERIAL-IDENTITY fraction (__texPhotoNear, default 0.45): up close the
                             // photo keeps its OWN hue at the macro's luminance, so ground reads clearly as
                             // grass/rock/sand/snow (user 2026-06-12 'light brown patches that need to be either
                             // grass or sand but is neither' -- the full macro tint painted biome browns/greys onto
                             // every layer). Far field returns to the shade-matched macro (no distance pop).
uniform vec4 uSurfMeanL;     // per-layer mean linear luminance of the photo color (loader-computed; shade-match divisor)
// MATERIAL-BOUNDARY DITHER REVERTED (2026-06-05): the threshold-perturbation approach (matEdgeNoise on
// the smoothstep input) produced HARD-EDGED PATCHES + a UV-like grid on uniform grass/snow (user live
// eye: 'hard uninteresting lines between rocky/grass', 'grass/snow UV problem') -- perturbing a near-
// binary boundary snaps material across wide areas instead of softly interfingering, and the high-freq
// octave aliased on bright materials. Reverted to the clean smoothstep boundaries; a non-aliasing
// 'interesting boundary' technique (e.g. a wide soft transition material band) is a separate future task.
vec3 terrainAlbedo(float h, float slope, float rockSlope, highp vec3 worldPos) {   // highp: worldPos feeds normalize(worldPos)*freq noise UVs -- mediump would scramble the lattice at close range
    vec3 c;
    if (h < 0.0) {
        // SEABED CONTINUES AS LAND MATERIAL (user 2026-06-11 'instead of turning terrain into water,
        // it should continue under water as sand and rock'): below sea level the macro albedo is a
        // sandy bed on gentle ground and rock on steep faces (same slope gate as land). The water
        // look (per-channel absorption, fresnel, waves) is composited OVER this in the ocean branch,
        // so the terrain level information stays readable through shallow water.
        c = mix(bcShore, bcRock, smoothstep(slopeRock.x, slopeRock.y, rockSlope));
    } else {
        c = mix(bcShore, bcLowland, smoothstep(0.0, bandEdgesLo.x, h));
        c = mix(c, bcGrass, smoothstep(bandEdgesLo.x, bandEdgesLo.y, h));
        float bandWarp = snoise3(normalize(worldPos) * 400.0) * 500.0;
        c = mix(c, bcRock, smoothstep(bandEdgesHi.x + bandWarp, bandEdgesHi.y + bandWarp, h));
        c = mix(c, bcSnow, smoothstep(snowEdges.x + bandWarp, snowEdges.y + bandWarp, h));
        c = mix(c, bcRock, smoothstep(slopeRock.x, slopeRock.y, rockSlope) * step(0.0, h));
        // OLD PROCEDURAL GREY ROCKFACE DELETED (max-speed sweep 2026-06-10, user 'replace the original
        // rock completely'): the photo-rock splat owns steep faces; the 3-tap grey fBm fallback is gone.
    }
    return c;
}
// CLIMATE-BIASED albedo: the anchor field carries latitude-driven temp + humidity (now
// SHIPPED). Drive Earth-like biome COLOR on land from climate so warm+wet reads lush green,
// warm+dry reads arid tan, and cold reads pale/desaturated with an earlier snow line --
// instead of a pure height ramp. Sea unchanged. temp,humid in [0,1].
// BIOME PALETTE: each biome a DISTINCT recognizable color, blended by soft temp/humidity
// thresholds (mirrors wasm/terrain-cli/biome-climate-tune.mjs which CLI-validated 7 distinct
// classes, entropy 2.59). World-continuous (pure fn of climate -> seam-safe).
vec3 biomeColor(float temp, float humid) {
    // PHYSICALLY-ANCHORED ALBEDOS (Real-World Look overhaul): real surface albedos are LOW and fairly
    // DESATURATED (conifer ~0.08, broadleaf ~0.15, grass ~0.22, dry sand ~0.35, snow ~0.85 the ONLY
    // bright class). The old palette was too bright + too saturated -> the sickly-yellow pastel
    // watercolour land (defect #1). Re-anchored to muted linear albedos; ICE stays bright.
    vec3 ICE     = vec3(0.86, 0.90, 0.96);   // snow/ice -- the only high-albedo class
    // TUNDRA WAS THE 'FLAT ROCK' (user 2026-06-11, witnessed by A/B: rock gate + texture OFF and the
    // grey patches remained = MACRO biome color): 0.34/0.36/0.31 is neutral grey, and the grass photo
    // tinted to it reads exactly as flat rock basins. Recolor to a muted olive-brown that is
    // unmistakably vegetation -- still duller/colder than MEADOW, no longer rock-grey.
    vec3 TUNDRA  = vec3(0.27, 0.33, 0.18);
    // FOREST CLASSES LIFTED (user 2026-06-11 'large rocky patches... flat areas should be snow sand
    // grass' -- same disease as the TUNDRA grey: 0.03-0.13 luminance is DARKER than the rock photo,
    // and desaturation + grass-photo tinting turns the near-black greens into flat dark patches that
    // read as rock). Lifted ~2x with the green channel dominant so they are unmistakably vegetation;
    // still the darkest land classes, canopy still deepest.
    vec3 TAIGA   = vec3(0.11, 0.19, 0.10);   // conifer green
    vec3 FOREST  = vec3(0.10, 0.22, 0.08);   // broadleaf green
    vec3 DEEPFOR = vec3(0.07, 0.16, 0.07);   // dense canopy (deepest green, no longer near-black)
    vec3 MEADOW  = vec3(0.28, 0.34, 0.15);   // muted olive grassland
    vec3 SAVANNA = vec3(0.46, 0.40, 0.22);   // dry gold (desaturated)
    vec3 STEPPE  = vec3(0.42, 0.39, 0.25);   // pale dry grass (desaturated)
    vec3 DESERT  = vec3(0.55, 0.45, 0.30);   // sand ochre (desaturated, darker)
    float cold = 1.0 - smoothstep(0.16, 0.34, temp);
    float warm = smoothstep(0.42, 0.62, temp);
    float dry  = 1.0 - smoothstep(0.34, 0.50, humid);
    // FOREST vs MEADOW split (user: they must look distinct, not one green). A CRISPER humidity
    // boundary (0.48->0.56, was 0.46->0.66) so meadow (drier-temperate) and forest (wetter) read
    // as DISTINCT adjacent regions instead of a long ambiguous blend; very-wet deepens to canopy.
    float wet  = smoothstep(0.48, 0.56, humid);
    float veryWet = smoothstep(0.62, 0.80, humid);
    vec3 c = MEADOW;                                          // temperate mid-humidity default
    c = mix(c, FOREST,  wet);                                 // wet -> forest (crisp boundary)
    c = mix(c, DEEPFOR, veryWet);                             // very wet -> dense dark canopy
    c = mix(c, SAVANNA, dry * warm);                          // dry + warm -> savanna
    c = mix(c, DESERT,  smoothstep(0.60,0.85,1.0-humid) * warm); // very dry + warm -> desert
    c = mix(c, STEPPE,  dry * (1.0-warm) * (1.0-cold));       // dry temperate -> steppe
    c = mix(c, TAIGA,   wet * (1.0 - smoothstep(0.34,0.50,temp))); // cool + wet -> taiga
    c = mix(c, TUNDRA,  cold);                                // cold -> tundra
    c = mix(c, ICE,     1.0 - smoothstep(0.10, 0.18, temp)); // very cold -> ice
    // GLOBAL SATURATION PULL toward luminance (Real-World Look): real terrain is less saturated than a
    // naive palette; pull ~28% toward grey so biomes read natural, not poster-paint. uBiomeSat<1.
    float bl = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(bl), c, uBiomeSat);
    return c;
}
// DISCRETE biome class -> a flat distinct color (for the biome-MAP diagnostic displayMode 9 +
// __biomeAt). Hard argmax of the same climate axes biomeColor blends, so the witness can COUNT
// contiguous biome regions instead of reading a continuous gradient. Water/ice keyed on h/temp.
// Returns a flat saturated key color per class (NOT the naturalistic biomeColor palette).
#ifdef _DEBUGVIEW_
vec3 biomeClassColor(float temp, float humid, float h) {
    if (h < 0.0) return vec3(0.10, 0.30, 0.75);                 // OCEAN (blue)
    if (temp < 0.14) return vec3(0.95, 0.97, 1.00);             // ICE (white)
    if (temp < 0.30) return vec3(0.55, 0.60, 0.55);            // TUNDRA (grey-green)
    // SWAMP: warm + very-wet + low ground (h<120m proxy) -> murky teal key color (distinct class).
    if (temp > 0.50 && humid > 0.66 && h >= 0.0 && h < 120.0) return vec3(0.20, 0.40, 0.30);
    bool warm = temp > 0.52;
    if (warm && humid < 0.22) return vec3(0.95, 0.80, 0.30);    // DESERT (yellow)
    if (warm && humid < 0.42) return vec3(0.80, 0.70, 0.20);    // SAVANNA (gold)
    if (humid > 0.66) return warm ? vec3(0.00,0.55,0.10)        // RAINFOREST (bright green)
                                  : vec3(0.05,0.25,0.12);       // TAIGA (dark green)
    if (humid > 0.45) return vec3(0.20, 0.65, 0.25);            // FOREST (green)
    return vec3(0.55, 0.70, 0.30);                              // MEADOW/STEPPE (yellow-green)
}
// RIVER NETWORK: thin continuous water lines on land. A world-dir-continuous ridged-noise
// network: riverField = 1 - |fBm(worldDir)| peaks (->1) along the zero-crossing ridge lines of
// the fBm, giving a connected branching channel pattern. Thresholded thin near 1.0 -> the river
// line. Pure fn of worldPos (vWorld) -> C0 seam-safe across tiles AND faces by construction (no
// per-tile state). Gated to land below a snowline + present in wet/temperate regions, scarce in
// desert (dry). `px` is a sub-pixel fade so the thin line does not alias/shimmer at altitude.
// NOT an elevation incision (FS albedo only) -> zero CLOD/seam/LOD-uniformity risk.
float riverMask(vec3 worldPos, float h, float temp, float humid, float px) {
    if (h <= 0.0) return 0.0;
    // ONE FRACTAL (user 2026-06-02, same fix as canyon): the FS used its OWN inline loop over
    // worldPos/terrainR (= dir*(1+h/R), elevation-shifted) -> a different field/position/resolution
    // than the VS riverCarveM. Call the IDENTICAL riverRidgeField the VS carve uses, at
    // normalize(worldPos) (== radial dir), so the FS river network coincides with the carved geometry.
    float ridge = riverRidgeField(normalize(worldPos));   // 0..1, ->1 on the channel network
    // thin the network to a line. MEASURED ridge distribution (diag-river.mjs nodejs-2037):
    // p90=0.858 p97=0.903 max=0.985; areaFrac>0.90 = 3.4%, >0.88 = 6%. Center the line band at
    // ~0.90 (lo 0.88 -> hi 0.935) for ~4-6% believable drainage density -- the old 0.92->0.985
    // band selected <2% and was invisible. Width grows slightly with px so the line stays >=1px.
    float wid = clamp(px * 0.0008, 0.0, 0.04);
    float line = smoothstep(0.875 - wid, 0.935, ridge);
    // climate gate: rivers in wet/temperate land, sparse in desert (very dry+warm) and on ice.
    float wetGate = smoothstep(0.30, 0.55, humid);          // need some moisture
    float notFrozen = smoothstep(0.20, 0.34, temp);          // not polar ice
    // NO altFade (user 2026-06-02: 'river field is fading in instead of integrated'). The line WIDTH
    // already grows with px (wid above) so the channel stays >=1px and AA-safe at every distance
    // WITHOUT vanishing -> the river network is a permanent part of the field, not a distance overlay.
    return line * wetGate * notFrozen;
}

// CANYONS (user-named): deep narrow incised gorges cutting ELEVATED ARID plateaus. Distinct from
// rivers (canyons = dry+deep+arid biome; rivers = water+shallow+wet). World-dir-continuous ridged
// network like riverMask but NARROWER + a HIGHER threshold (sparser, sharper) and a DIFFERENT
// noise phase so canyons and rivers do not coincide. Gated to ELEVATED (h>250m) DRY+WARM regions.
// FS-albedo only (no elevation incision) -> zero CLOD/seam risk, AA via pxWorld. Returns line + a
// depth term for strata banding. out_depth = how deep into the gorge (0 rim -> 1 floor), for strata.
float canyonMask(vec3 worldPos, float h, float temp, float humid, float px, out float depth) {
    depth = 0.0;
    // gentle elevation gate: canyons want SOME relief above the lowland, not only rare high
    // plateaus (h>250 made them NEVER appear -- witnessed canyonFieldFrac 0). Fade in over 60..200m.
    // ONE FRACTAL (user 2026-06-02: 'canyon field shows two different resolutions, expected the same
    // fractal'). The FS used its OWN inline loop over worldPos/terrainR (= dir*(1+h/R), elevation-
    // SHIFTED off the VS sample) -> a different field/position/resolution than the VS carve. Now call
    // the IDENTICAL canyonRidgeField the VS canyonCarveM uses, at normalize(worldPos) (== the radial
    // dir), so the FS network coincides EXACTLY with the carved geometry. Same field, no elevation shift.
    float ridge = canyonRidgeField(normalize(worldPos));
    float wid = clamp(px * 0.0006, 0.0, 0.03);                 // narrower than rivers
    float line = smoothstep(0.875 - wid, 0.94, ridge);        // ~river-density threshold so they appear
    depth = smoothstep(0.875, 0.95, ridge);                   // deeper toward the channel centre
    // NO altFade (user: canyon field fading instead of integrated). wid grows with px -> >=1px AA at
    // every distance without vanishing; the canyon network is permanent, not a distance overlay.
    return line * step(0.0, h);
}
#endif   // biomeClassColor/riverMask/canyonMask: DEBUGVIEW-only (called solely from displayMode blocks) -- excluded from render FS cold-compile (FS-2, workflow w4y1bnrqc)

vec3 terrainAlbedoClimate(float h, float slope, float rockSlope, float temp, float humid, highp vec3 worldPos, float pxWorld) {   // highp: worldPos feeds normalize(worldPos)*freq noise UVs (mottle/river/canyon ridge) -- mediump scrambles the lattice up close
    vec3 c = terrainAlbedo(h, slope, rockSlope, worldPos);
    if (h < 0.0) {
        // SEA ICE: near-polar ocean (very cold) freezes to white-blue pack ice. Pure fn of the
        // anchor temp -> seam-safe; the soft threshold gives an irregular (not hard-zonal) margin.
        float seaIce = 1.0 - smoothstep(0.12, 0.22, temp);
        return mix(c, vec3(0.82, 0.88, 0.94), seaIce * 0.9);
    }
    // biome weight: full on gentle lowland, fading where rock/snow/steep takes over.
    float veg = (1.0 - smoothstep(bandEdgesHi.x, bandEdgesHi.y, h)) * (1.0 - smoothstep(slopeRock.x, slopeRock.y, slope));
    // ELEVATION + LATITUDE BIOME BIAS (user 2026-06-02: 'the hypsometric ramp should influence biome
    // distribution, pulling biomes out of their anchor areas a bit for better distribution'). On top
    // of the anchor climate temp/humid, bias the EFFECTIVE temperature DOWN with elevation (lapse rate
    // ~6.5C/km -> normalize ~ h/4000 over the [0,1] temp scale) and a touch with latitude, so biomes
    // BAND by height+latitude: high ground -> alpine/tundra/ice, lowland keeps its anchor biome. This
    // spreads biomes into their physically-expected bands so grass/forest isnt one anchor blob. humid
    // also rises slightly on windward high ground (orographic) for variety. uBiomeBandBias scales it.
    float lat = asin(clamp(worldPos.y / max(length(worldPos), 1.0), -1.0, 1.0));
    float latCool = 0.18 * (abs(lat) / 1.5708);                  // poles cooler
    float elevCool = clamp(h / 4500.0, 0.0, 0.55) * uBiomeBandBias;  // lapse rate -> alpine bands
    float tempEff  = clamp(temp - elevCool - latCool * uBiomeBandBias, 0.0, 1.0);
    float humidEff = clamp(humid + clamp(h / 9000.0, 0.0, 0.12) * uBiomeBandBias, 0.0, 1.0);
    vec3 biome = biomeColor(tempEff, humidEff);
    // STRONG blend (0.82) so the biome DOMINATES the lowland color -> regions read distinct,
    // not the faint 0.55 tint that made the whole continent look the same.
    c = mix(c, biome, veg * 0.82);
    // INTRA-BIOME VALUE MOTTLE (Real-World Look): real terrain is never one flat color per biome --
    // soil/moisture/vegetation patchiness mottles albedo. Reuse the snoise3 already evaluated for
    // cliffRock (same normalize(worldPos)*~1k pattern, no new octave) as a VALUE-only multiplier (NO
    // hue jitter, low 1200 freq -> does NOT reintroduce the per-pixel swamp moire the pipeline removed).
    float mot = snoise3(normalize(worldPos) * 120.0);   // detail-tex-rockface-canyon-10x: *1200->*120 (~10x larger mottle features, user 2026-06-06)
    c *= (1.0 + uVariationAmt * mot);
    // earlier snow/ice on high ground in cold regions.
    float coldSnow = (1.0 - smoothstep(0.30, 0.75, temp)) * smoothstep(snowEdges.x * 0.5, snowEdges.x, h);
    c = mix(c, bcSnow, coldSnow * 0.7);
    // BEACH BAND (user 2026-06-11 'all land at the level of the ocean should be beach -- the grass
    // must stop and become sand, which continues under the water'): below uBeachTopM the biome/
    // grass/snow color yields to shore sand (the same bcShore the underwater bed starts from, so the
    // material is continuous through the waterline). Steep coastal cliffs keep their rock.
    // WIDENED TRANSITION (2026-06-13): 0.6->0.3 softens the grass-sand boundary from 12m to 21m.
    float beachM = (1.0 - smoothstep(uBeachTopM * 0.3, uBeachTopM, h))
                 * (1.0 - smoothstep(slopeRock.x, slopeRock.y, rockSlope));
    c = mix(c, bcShore, beachM);
    // INLAND WATER -- COLOUR GATED BY THE CARVE FIELD (alignment + no-fade fix, browser-2705/2699).
    // The water colour now keys off the SAME world-dir carve fields that cut the geometry
    // (lakeCarveM/riverCarveM at this fragment's world dir), NOT an independent h-smoothstep. So the
    // blue sits EXACTLY in the carved bowl/channel at every LOD (color == depression by construction)
    // -- this kills both the 'lakes fade on approach' (was gated by h<70 which shifts with relief)
    // and the 'rivers/lakes out of alignment with their elevation' defects. Climate still gates WHICH
    // regions get water (wet for lakes/rivers) but no longer the SHAPE (the carve owns the shape).
    // INLAND WATER + INCISION COLOUR -- UNIFIED: keyed off the carve masks computed ONCE in the VS
    // and INTERPOLATED here (vLakeWet/vRiverWet/vCanyonDep), NOT re-evaluated per-pixel. This removes
    // the separate per-pixel ridged-field evaluation that aliased into biome-localized moire (user
    // 2026-06-01h: 'UV issue only in patches where the biome is like that'). The VS already folded
    // the climate gates into the masks, so the FS just applies colour.
    float warmth = smoothstep(0.48, 0.62, temp);
    // SWAMP: only the WARM + VERY-WET end of a lake basin reads murky olive-teal. The humid gate
    // (0.66-0.86) was lost in the unify rewrite -> the teal leaked into merely-warm wet land (user:
    // 'faint teal area, moire'); restored here. The per-pixel mottle snoise3 (freq 900) was the last
    // per-pixel noise field -> it gave the teal its grain; REMOVED (flat swamp colour, no aliasing).
    // INLAND WATER ALBEDO IS NOT SET HERE ANYMORE. Lakes and rivers are WATER (user 2026-06-01i:
    // 'use the same content as the sealevel to draw because its also water'), so they are shaded by
    // the SAME ocean water branch (fresnel + sun glint + depth tint) post-lighting, gated on the
    // submerged-depth varying vWaterDepth so the color lands EXACTLY on the flat water plane (lines
    // up with the carve) instead of being painted over the whole basin including the dry shoulder.
    // The terrain albedo (rock/biome) stays under the water and shows through where shallow.
    // CLIFF STRATA BLOCK DELETED (max-speed sweep 2026-06-10): the photo-rock splat owns cliff
    // appearance; the 2-tap bedding-warp strata fallback is gone (-2 snoise3, ~35 LOC).
    // DUNES: warm sand on the desert floor, keyed off the interpolated dune-crest mask (vDuneCrest =
    // crest * duneSand gate from the VS). Crest sun-bleached lighter, trough ochre.
    vec3 sandLo = vec3(0.78, 0.64, 0.40);   // trough ochre sand
    vec3 sandHi = vec3(0.90, 0.82, 0.62);   // sun-bleached crest sand
    vec3 sandCol = mix(sandLo, sandHi, smoothstep(0.0, 0.6, vDuneCrest));
    c = mix(c, sandCol, smoothstep(0.0, 0.25, vDuneCrest) * (1.0 - smoothstep(0.30, 0.50, slope)) * 0.9);
    // PERLIN-EVERYWHERE OVERLAY (user 2026-06-10 'pale + featureless in some areas -- add perlin
    // everywhere and overlay the other noises'): 3-octave value fbm over ALL materials (biome, rock,
    // snow, sand) so no area is ever a flat color. World-dir keyed (no camera scroll -- the lesson of
    // the deleted 2026-06-01 detail texturing), value-only (no hue shift), and each octave fades out
    // via a pxWorld Nyquist gate before its features go sub-pixel (no orbit speckle, no leopard band).
    {
        highp vec3 od = normalize(worldPos);
        float ov = 0.0, oa = 0.0;
        float fq = 75.0, am = 1.0;                 // octaves: ~84km / 17km / 3.4km features (halved to match VS detailFbm)
        int fdOcts = (uFSDetailOcts > 0) ? uFSDetailOcts : 3;
        for (int o = 0; o < fdOcts; o++) {
            float wl = 40000000.0 / fq;             // feature wavelength (m) ~ 2*pi*R / fq
            float nyq = 1.0 - smoothstep(wl * 0.03, wl * 0.12, pxWorld);   // fade before sub-pixel
            ov += am * nyq * snoise3(od * fq + vec3(float(o) * 7.3));
            oa += am;
            fq *= 5.0; am *= 0.6;
        }
        // ALBEDO amplitude decoupled from the elevation lever (user 2026-06-11 'normals seem wrongly
        // calculated, not on the sides of slopes' -- witnessed: at the user-tuned uDetailOverlay 6 the
        // albedo term was *= 1 +/- 0.54, painting slope-INDEPENDENT dark braids over every material
        // that read as misplaced shading / rock bands). 0.09 -> 0.02: lever 6 now gives +/-12% value
        // variation (anti-featureless, as asked) while the +/-180m ELEVATION term keeps the full lever.
        c *= 1.0 + uDetailOverlay * 0.02 * (ov / max(oa, 1e-3));
    }
    return c;
}

// ---- DETAIL MATERIAL (1-2 texel/pixel from anchor biome weights). The coarse biome+ortho
// PER-PIXEL DETAIL TEXTURING REMOVED (user 2026-06-01e: "get rid of any detail texturing, we only
// want one fractal"). The old FS grain (detailMaterial/detailNormal/grainH + the dnoise lattice)
// was a SECOND high-frequency source on top of the VS height fractal -- it caused the moire and the
// scroll/jump (its lattice was keyed on the moving camera). Deleted entirely. The single terrain
// fractal is the VS world-dir height field; closeup detail comes from the mesh subdividing into a
// denser sample of THAT one field, never a per-pixel fake-relief texture.

// SURFACE PHOTO-TEXTURE triplanar tap: 3 axis-projected samples of one array layer, blended by
// the pow-softened |n| weights (same continuous-projection rationale as the rock bump biplanar --
// no hard dominant-axis flip). wt = world pos in tile units (fragment-anchored, fp32-precise).
vec4 surfTriTap(sampler2DArray sm, highp vec3 wt, vec3 bw, float layer) {
    return texture(sm, vec3(wt.y, wt.z, layer)) * bw.x
         + texture(sm, vec3(wt.x, wt.z, layer)) * bw.y
         + texture(sm, vec3(wt.x, wt.y, layer)) * bw.z;
}
// WORLD-SPACE triplanar normal perturbation (UDN) -- THE 'math issue causing both' (user 2026-06-11):
// the old path triplanar-BLENDED the per-plane RG tangent normals, then applied the blend in the
// single radial (ux,uy) frame -- but each projection plane has its own tangent axes, so the blended
// RG was rotated arbitrarily vs the frame it was applied in (bumps lit from wrong directions =
// statistical darkening + shading off the slope sides), and the apply site SUBTRACTED an
// already-negated Sobel normal (double negation = lit from the wrong side outright). Here each
// plane's RG perturbs along that plane's own world axes, sign-flipped to push outward, blended by
// the same weights -- a world-space delta added to the lit normal. Same 3 taps.
vec3 surfTriNrm(sampler2DArray sm, highp vec3 wt, vec3 bw, float layer, vec3 sn) {
    vec2 px = texture(sm, vec3(wt.y, wt.z, layer)).rg * 2.0 - 1.0;   // X plane: in-plane axes (Y,Z)
    vec2 py = texture(sm, vec3(wt.x, wt.z, layer)).rg * 2.0 - 1.0;   // Y plane: in-plane axes (X,Z)
    vec2 pz = texture(sm, vec3(wt.x, wt.y, layer)).rg * 2.0 - 1.0;   // Z plane: in-plane axes (X,Y)
    return vec3(0.0, px.x, px.y) * (bw.x * sign(sn.x))
         + vec3(py.x, 0.0, py.y) * (bw.y * sign(sn.y))
         + vec3(pz.x, pz.y, 0.0) * (bw.z * sign(sn.z));
}

void main() {
    // PER-FRAGMENT tangent frame from vWorld (the sphere position, C0 across quad edges):
    // uz = up (radial), ux = normalize(Y x uz), uy = uz x ux. Continuous across adjacent quads
    // -> no per-quad shading-baseline seam.
    vec3 uz = normalize(vWorld);
    // AUDIT FIX 2026-06-06 (audit-tangent-frame-degeneracy): the reference axis was a FIXED vec3(0,1,0),
    // so cross(Y, uz) -> zero-length (NaN normal) wherever uz is parallel to Y -- i.e. the +Y/-Y cube
    // faces and the poles, exactly 'wrong normals in the wrong places'. Pick the world axis LEAST
    // parallel to uz so the cross product is always well-conditioned (|ux| ~ 1) everywhere on the sphere.
    vec3 refAxis = (abs(uz.y) < 0.99) ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 ux = normalize(cross(refAxis, uz));
    vec3 uy = cross(uz, ux);
    // ---- WATER-SURFACE PASS (uIsWater=1, the separate ocean surface at sea level). vH carries the
    // TRUE seabed elevation under this fragment -> depthM = the water column. Shade as animated
    // water (Gerstner normal + fresnel sky + sun glint + foam) and ALPHA-blend over the already-
    // rendered seabed: alpha rises with optical depth (Beer-Lambert) + fresnel/foam, so shallows
    // show the sand through real transparency and deep basins read opaque navy. Early return --
    // none of the terrain material/atmosphere work below runs for water fragments.
    if (uIsWater > 0.5) {
        if (vH > 1.0) discard;                       // surface under land: depth test culls it anyway; discard kills shoreline shimmer
        // UNDERWATER WATER SURFACE (camera below sea level): render the surface from below as
        // opaque deep blue with Gerstner wave animation. No sky reflection, no Fresnel (TIR from
        // below = mostly opaque deep water), no Beer-Lambert (there is no seabed above the camera).
        if (uUnderwater > 0.5) {
            highp vec3 wOriginW = floor(camWorld / 1024.0) * 1024.0;
            highp vec2 wpW = vec2(dot(vWorld - wOriginW, ux), dot(vWorld - wOriginW, uy));
            vec2 slopeW = oceanWaveSlope(wpW, oceanTime);
            highp float wDistW = length(camWorld - vWorld);
            slopeW *= clamp(1.0 - wDistW / 4000.0, 0.0, 1.0);
            vec3 wn = normalize(uz - ux * slopeW.x - uy * slopeW.y);
            vec3 viewW = normalize(camWorld - vWorld);
            float ndl = max(dot(wn, sunDir), 0.0);
            // Sunlight attenuated through the water column; deep blue ambient
            float depthAtten = exp(-max(0.0, terrainR - length(camWorld)) * 0.0005);
            vec3 sunUnder = vec3(1.0, 0.6, 0.3) * depthAtten * ndl;
            vec3 deepBlue = vec3(0.005, 0.06, 0.18);
            vec3 waveBright = vec3(0.0, 0.02, 0.06) * length(slopeW);
            vec3 wcol = deepBlue + sunUnder * 0.4 + waveBright;
            float macroMuW = dot(uz, sunDir);
            float dayShadeW = mix(uNightFloor, 1.0, smoothstep(-uTermWidth, uTermWidth, macroMuW));
            vec3 cW = wcol * dayShadeW * uExposure;
            vec3 mappedW = clamp((cW * (2.51 * cW + 0.03)) / (cW * (2.43 * cW + 0.59) + 0.14), 0.0, 1.0);
            float lumW = dot(mappedW, vec3(0.2126, 0.7152, 0.0722));
            mappedW = mix(vec3(lumW), mappedW, uLookSat);
            mappedW = clamp((mappedW - 0.5) * uLookContrast + 0.5, 0.0, 1.0);
            fragColor = vec4(pow(mappedW, vec3(1.0 / 2.2)), 1.0);
            return;
        }
        highp float depthM = max(-vH, 0.0);
        highp vec3 wOriginW = floor(camWorld / 1024.0) * 1024.0;   // snapped anchor (fp32 wave-phase fix, same as the old branch)
        highp vec2 wpW = vec2(dot(vWorld - wOriginW, ux), dot(vWorld - wOriginW, uy));
        vec2 slopeW = oceanWaveSlope(wpW, oceanTime);
        highp float wDistW = length(camWorld - vWorld);
        slopeW *= clamp(1.0 - wDistW / 4000.0, 0.0, 1.0);          // sub-pixel wave fade (anti-alias)
        vec3 wn = normalize(uz - ux * slopeW.x - uy * slopeW.y);
        vec3 viewW = normalize(camWorld - vWorld);
        float fres = 0.02 + 0.98 * pow(clamp(1.0 - max(dot(wn, viewW), 0.0), 0.0, 1.0), 5.0);
        highp float camAltW = length(camWorld) - terrainR;
        float specFade = clamp(1.0 - camAltW / 150000.0, 0.0, 1.0); // orbit glint fade (kept)
        vec3 hlW = normalize(sunDir + viewW);
        float spec = pow(max(dot(wn, hlW), 0.0), 220.0) * specFade;
        float ndl = max(dot(wn, sunDir), 0.0);
        vec3 T = exp(-uOceanK * depthM);                            // per-channel transmittance
        float Tavg = (T.r + T.g + T.b) * (1.0 / 3.0);
        vec3 waterBase = mix(uOceanDeep, uOceanShallow, Tavg);
        vec3 skyColW = vec3(0.30, 0.42, 0.55);
        vec3 wcol = mix(waterBase * (0.25 + 0.75 * ndl), skyColW, fres * 0.85)
                  + vec3(1.0, 0.95, 0.85) * spec * ndl;
        float foamAmt = clamp((length(slopeW) - 0.6) * 1.5, 0.0, 1.0) * oceanFoam;
        wcol = mix(wcol, vec3(0.9, 0.95, 1.0), foamAmt * (0.3 + 0.7 * ndl));
        wcol += waterBase * 0.05;                                   // ambient floor (never black)
        // distance haze: the terrain pass gets the full aerial-perspective march; give the water a
        // cheap matched fade toward the same sky-haze color so far ocean recedes like far land
        // (no 8-step march on the largest-area surface -- the perf theme of this pass).
        float apGW = smoothstep(3000.0, 120000.0, wDistW) * uHazeMul;
        wcol = mix(wcol, uSkyFill * vec3(0.40, 0.55, 0.78) * 1.6, apGW * 0.7);
        // day/night + tonemap chain kept consistent with the terrain pass so the two surfaces match.
        float macroMuW = dot(uz, sunDir);
        float dayShadeW = mix(uNightFloor, 1.0, smoothstep(-uTermWidth, uTermWidth, macroMuW));
        vec3 cW = wcol * dayShadeW * uExposure;
        vec3 mappedW = clamp((cW * (2.51 * cW + 0.03)) / (cW * (2.43 * cW + 0.59) + 0.14), 0.0, 1.0);
        float lumW = dot(mappedW, vec3(0.2126, 0.7152, 0.0722));
        mappedW = mix(vec3(lumW), mappedW, uLookSat);
        mappedW = clamp((mappedW - 0.5) * uLookContrast + 0.5, 0.0, 1.0);
        // coverage: optically thick water -> opaque; first metres see-through; grazing fresnel and
        // foam are opaque regardless of depth; feather the exact waterline contact.
        // THIN WATER STAYS CLEAR (user 2026-06-14 'water isnt transparent where its thin'): the grazing
        // fresnel forced shallow water OPAQUE even looking across a shoreline. Gate fresnel opacity by a
        // depth ramp so thin water shows the bed regardless of view angle; foam + Beer-Lambert depth
        // opacity stay so deep/foamy water still reads solid.
        float shallowClear = smoothstep(0.0, 6.0, depthM);   // 0 in thin water -> clear; 1 by ~6m -> full fresnel
        float alphaW = clamp(1.0 - Tavg, 0.0, 1.0);
        alphaW = max(alphaW, fres * 0.9 * shallowClear);
        alphaW = max(alphaW, foamAmt * 0.8);
        // SHALLOW-SURFACE FLOOR (2026-06-11, found by the coast witness after the shelf landed):
        // the continental shelf keeps water metres-deep for kilometres, where Beer-Lambert alpha
        // ~0 and nadir fresnel ~0 made the whole shoreline INVISIBLE (coverage 0 at a bisected
        // waterline pose). Real shallow water still shows its surface (sky reflection + ripple):
        // floor the opacity once genuinely submerged so shorelines read as water.
        // SHALLOW TRANSPARENCY (user 2026-06-14: 'water fully transparent where it meets land, see the
        // bed through it'): floor cut 0.30->0.12 and pushed deeper (start 1.5m) so the bed shows through
        // the shoreline + shallows; deeper water still goes opaque via Beer-Lambert (1-Tavg). The contact
        // fade is widened (0->3m) so the exact waterline is clear, not a hard opaque rim.
        alphaW = max(alphaW, 0.12 * smoothstep(1.5, 8.0, depthM));
        alphaW *= smoothstep(0.0, 3.0, depthM);
        fragColor = vec4(pow(mappedW, vec3(1.0 / 2.2)), alphaW);
        return;
    }
    // W8 SOLE FS LIT NORMAL (P1 data-first, P9 all-distances): vNrm is the per-vertex analytic central-
    // difference normal of the FULL composeHeight field, computed in the VS at a FIXED STABLE world step.
    // Full landform relief (broadShapeM fine octaves + carves) at ALL distances, NO per-pixel jitter, NO
    // fade, NO band-limit. Replaces the cross(dFdx,dFdy) geometric normal (it jumped per-pixel across the
    // bumpy displaced vWorld = the 'completely messed up' scramble) and the bandFadeN macro-only fade
    // (which made fine relief go missing). It is a continuous function of dir0 only -- adjacent fragments
    // get the linearly-interpolated VS value, so it is smooth by construction (no dFdx/dFdy, no fwidth).
    // uFlatNormal kept as a cheap escape hatch: pure sphere normal (uz) for A/B isolation.
    vec3 n = (uFlatNormal > 0.5) ? uz : normalize(mix(vNrm, uz, 0.05));

    // GPU-TIMER VS-ISOLATION: a cheap measurement frame short-circuits the whole per-pixel shade
    // here (uses vNrmPV + vWorld so the VS outputs are not dead-code-eliminated) -> the timed cheap
    // frame is VS+raster only; (full - cheap) attributes the per-pixel FS cost. No effect when 0.
    if (uFsCheap > 0.5) { fragColor = vec4(n * 0.5 + 0.5, 1.0); return; }


    // DIAGNOSTIC displayModes (1,5,6,7,8,9,10,11,12) are gated behind _DEBUGVIEW_ so they are
    // ONLY compiled into the lazily-built DEBUG program (gl-render.js), NOT the hot render program.
    // They added 7132 chars (25%) of translated HLSL to the render FS (browser-1590) -- pure dead
    // weight on the lit path, and the heavy ones pull in riverMask/canyonMask/biomeClassColor which
    // ANGLE then drops from the render program once unreferenced. Cheap albedo modes 2/4 stay live
    // below (one fragColor=albedo each). The render program forces displayMode 0 (lit) effectively.
#ifdef _DEBUGVIEW_
    // displayMode 1 (Normals) MOVED below the splat (user 2026-06-11 'in the normals view we dont
    // see the texture normals'): this early return showed the pre-splat geometric normal, so the
    // displacement-derived texture normals (texDn) were invisible in the debug view. The handler
    // now lives after nLit/texDn assembly and shows the FINAL lit normal.
    // DIAG displayMode 5: land/sea map -- RED where vH>0 (land), BLUE where vH<0 (ocean).
    // Unambiguous test of the elevation sign distribution the render actually samples.
    if (displayMode == 5) { fragColor = (vH < 0.0) ? vec4(0.1,0.2,0.9,1.0) : vec4(0.9,0.3,0.1,1.0); return; }
    // DIAG displayMode 6: encode signed vH into RGB. R = vH/8000 (positive), B =
    // -vH/8000 (negative), G = |vH|/8000. Lets a readback recover the actual elevation
    // magnitude the render samples (independent of the biome ramp / readTile artifact).
    // ELEVATION displayMode 6: HYPSOMETRIC ramp (no hard clip / washed-out tops, user 2026-06-02).
    // Sea = blue (deeper -> darker); land ramps green->tan->brown->grey->white by height over a 0..9km
    // scale, each band a smoothstep so peaks read as distinct grey/white rock, NOT a saturated white
    // blob. >9km (shouldn't occur after the massif tame) just stays at the snow-rock top, no blow-out.
    // ELEVATION displayMode 6: a SCIENTIFIC (non-albedo) data ramp so it reads as an elevation MAP,
    // not naturalistic terrain colour (user 2026-06-02: 'elevation view looks weird like albedo').
    // Sea = dark->mid blue by depth; land = turbo-style cyan->green->yellow->orange->red->white by
    // height (8.5km full scale). Clearly a heatmap, no hard clip.
    if (displayMode == 6) {
        if (vH < 0.0) {
            float dep = clamp(-vH / 6000.0, 0.0, 1.0);
            fragColor = vec4(mix(vec3(0.10,0.55,0.85), vec3(0.0,0.05,0.30), dep), 1.0); return;
        }
        // full scale 11km so even an 8km peak reads RED/MAGENTA, not saturated white; white reserved
        // for the very rare >10.5km tip -> NO broad white-clipped areas (user 2026-06-02: 'fully white').
        float t = clamp(vH / 11000.0, 0.0, 1.0);
        vec3 ec = mix(vec3(0.0,0.80,0.85), vec3(0.10,0.85,0.20), smoothstep(0.0, 0.18, t));   // cyan->green
        ec = mix(ec, vec3(0.95,0.95,0.10), smoothstep(0.18, 0.38, t));    // -> yellow
        ec = mix(ec, vec3(0.95,0.50,0.05), smoothstep(0.38, 0.58, t));    // -> orange
        ec = mix(ec, vec3(0.80,0.10,0.10), smoothstep(0.58, 0.78, t));    // -> red
        ec = mix(ec, vec3(0.55,0.10,0.40), smoothstep(0.78, 0.93, t));    // -> magenta (very high)
        ec = mix(ec, vec3(1.0,1.0,1.0),    smoothstep(0.95, 1.0,  t));    // -> white only at the tip
        fragColor = vec4(ec, 1.0); return;
    }
#endif // _DEBUGVIEW_ (modes 1,5,6)

    float slope = 1.0 - max(0.0, dot(n, uz));   // 0 = flat, ->1 = steep
    // Rock classification keys on TRUE slope = 1-dot(n,uz) from the sole band-limited geometric normal.
    // gslope is the same quantity (kept as a name for the AO/detail-normal terms below).
    float gslope    = slope;
    // ROCK-FACE BREAKUP NOISE DELETED (user 2026-06-11, 3-way isolation witness: the braided
    // km-scale rock/grass interfingering at this noise's exact 2.5km wavelength WAS the 'rocky
    // patches' -- fake texture ridges the real terrain (normals-view smooth) does not have, hence
    // 'normals dont match elevation'. The breakup compensated for the over-steep pre-14ad9b8 peak
    // stack; with sane slopes the plain gate is correct.)
    float rockSlope = clamp(slope, 0.0, 1.0);
    // world metres spanned by one screen pixel (MOVED up from ~line 1321 so the close-up signal fade is
    // available to the rock-detail block below). nearFade=1 at the deck (a crag is many px) -> 0 at
    // altitude (the crag goes sub-pixel) so the entire close-up rock signal fades to zero from orbit and
    // the macro path is unchanged there (contrast-on-approach-safe, no orbit shimmer).
    float pxWorld  = max(length(fwidth(vWorld)), 0.001);
    // FADE CEILING 40->180m (live-deck witness 2026-06-05, browser-31: pxWorld is ~5m DIRECTLY below the
    // eye but climbs PAST 40m across the rest of a wide-FOV near-ground frame, so at 40m the close-up rock
    // detail + microAO + chromatic colorVar switched OFF across most of the visible deck -- the realism
    // never reached the FPS range the user actually flies (tens-to-hundreds of m/px). 180m keeps the whole
    // near-ground swath engaged while still fully fading by orbit (8m floor unchanged -> no orbit shimmer).
    float nearFade = 1.0 - smoothstep(8.0, 180.0, pxWorld);   // macro crease-AO distance fade (vAO only)
    // PROCEDURAL ROCK DETAIL-NORMAL DELETED (max-speed sweep 2026-06-10): the photo-rock
    // displacement normal owns rock micro-relief; the 3-tap snoise3D biplanar bump is gone.
    // microSlope/microCurv keep their macro defaults for the material/AO consumers below.
    vec3 nLit = n;
    float microSlope = rockSlope;
    float microCurv  = 0.0;
    // base biome albedo from the coherent height/slope ramp, biased by anchor CLIMATE
    // (latitude-driven temp + humidity) so lowland color sorts by biome (lush/arid/cold).
    vec4 climate = vClimate;   // (seaBias, elevAmp, temp=.z, humid=.w) -- INTERPOLATED from the VS,
                               // NOT a per-pixel HPF texture read (that read showed the HPF texel
                               // grid as UV lines/moire up close). Smooth biome transitions now.
    // (pxWorld is computed up near the slope block now -- moved so nearFade can fade the rock detail.)
    vec3 albedo = terrainAlbedoClimate(vH, slope, microSlope, climate.z, climate.w, vWorld, pxWorld);
    // CURVATURE MICRO-AO (close-up realism, workflow w5gywvug1): the concave creases between rock crags
    // (microCurv < 0) darken -- the contrast that reads as 3D rugged relief instead of flat lit clay.
    // Gated by microSlope (flat ground untouched); band-limited by the crag bump it derives from; scaled by the
    // live uAoAmt lever so it stays terraform-tunable. negCurv = how concave the crease is (0 on convex).
    float negCurv = max(0.0, -microCurv);
    float aoLever = (uAoAmt > 0.0 ? uAoAmt : 1.0);
    // LOW-CONTRAST crease AO (user 2026-06-06 flecked-rock fix): the crease darkening was up to 60% (the
    // dark half of the fleck speckle). Soften to <=30% so creases read as gentle shading, not hard specks.
    // microCurv already comes from the band-limited (rockBumpGate) crag bump, so this is present at all
    // distances and fades out naturally as the bump does -- no separate nearFade gate.
    float microAO = 1.0 - clamp(negCurv * 3.0, 0.0, 0.30) * smoothstep(0.04, 0.2, microSlope) * aoLever;
    albedo *= microAO;
    // CLOSE-UP CHROMA VARIATION DELETED (max-speed sweep 2026-06-10): the photo textures carry
    // near-field color variation now (-2 snoise3/pixel).
    // ---- SURFACE PHOTO-TEXTURE SPLAT (user 2026-06-10): triplanar grass/rock/sand/snow color +
    // displacement-derived normal, blended by the SAME gates the procedural materials use (slope ->
    // rock, snowline+cold -> snow, hot-dry climate / dunes -> sand, else grass) so texture and macro
    // material always agree. Distance-faded by pxWorld; mips + macro biome color carry the far field.
    // MIP-OUT, not fade-out (user 2026-06-11 'instead of fading out the textures, mip them out'):
    // the 15-20km camera-distance fade is GONE -- the splat persists at every distance and the
    // hardware mip chain carries the far field, where the texture averages to its mean color
    // (= the macro shade, by the layer-mean shade-match; rock's raw-photo mean = __surfRockMean).
    vec3 texDn = vec3(0.0);   // photo-texture WORLD-SPACE normal perturbation, applied after uReliefShade
    // SPLAT RUNS UNDERWATER TOO (user 2026-06-11 'continue under water as sand and rock'): the old
    // vH > -2 gate cut the photo textures at the waterline, leaving the seabed flat-colored.
    float texFarFade = 1.0 - smoothstep(8000.0, 10000.0, pxWorld);
    if (uHasSurfTex > 0.5 && uTexMix > 0.001 && texFarFade > 0.001) {
        // material weights from the existing gates (climate = vClimate: z=temp, w=humid)
        // SAND GATE = THE MACRO DESERT GATE (user 2026-06-11 'all the grassy areas need to have the
        // grass texture'): the old humid<0.42 band splatted SAND across savanna/steppe/meadow-edge
        // regions whose MACRO color is green/gold vegetation (biomeColor only reaches DESERT at
        // smoothstep(0.60,0.85,1-humid)) -- green ground wore the sand photo. One source of truth:
        // reuse the exact macro desert ramp, so the sand texture appears precisely where the macro
        // paints desert; savanna/steppe splat GRASS tinted dry-gold by the layer shade-match.
        float dryHot = smoothstep(0.60, 0.85, 1.0 - climate.w) * smoothstep(0.42, 0.62, climate.z);
        // BEACH (user 2026-06-10 'the beaches should be sand'): the shore profile eases over the
        // first ~60m of elevation, so low gentle land near sea level splats sand regardless of climate.
        // NOISE-VARIED THRESHOLDS (2026-06-13): vTexWarp.x adds an irregular wavy line so the sand
        // doesn't follow a strict elevation contour — reads as a natural beach, not a cut.
        // SHARED biome-band warp pattern (user 2026-06-14: the SAME warping on snow/rock bands AND the
        // beach->land crossover). 2-oct world-dir noise, 1/4 freq (~13km + ~5km waves). Computed once
        // here; the snow/rock bandWarp below reuses warpN. highp dir for the lattice precision.
        highp vec3 bwDir = normalize(vWorld);
        float warpN = snoise3(bwDir * 3325.0) + 0.5 * snoise3(bwDir * 7750.0);   // ~ +/-1.5
        // BEACH sand gate tied to uBeachTopM (so the sand TEXTURE scales with the wide beach, not a
        // hardcoded 80m strip) + the shared warp on its LAND edge so the beach->grass line is irregular.
        float beachW = warpN * uBeachTopM * 0.30;   // warp amplitude scales with the beach band height
        float beach = (1.0 - smoothstep(uBeachTopM * 0.12 + beachW, uBeachTopM + beachW, vH))
                    * (1.0 - smoothstep(0.15, 0.42, slope));
        // SAND BLEED (2026-06-13): patchy sand spills above the main beach line, modulated by VS
        // warp noise so the edge reads as wind-blown pockets, not a strict elevation cut. At peak it
        // adds ~0.3 sand weight far above the beach, creating a natural dappled transition.
        float sandBleed = max(0.0, vTexWarp.y) * 0.35 * max(0.0, 1.0 - smoothstep(30.0, 200.0, vH));
        // SAND REGIONS SUPPRESS ROCK (user 2026-06-10 'rock being used instead of sand'): in
        // deserts/dunes/beaches sand drapes moderate slopes; rock only wins on genuinely steep faces
        // there (gate shifted toward 0.5-0.7 inside sand regions instead of slopeRock 0.28-0.55).
        float sandRegion = clamp(max(max(dryHot, max(beach, sandBleed)), smoothstep(0.0, 0.25, vDuneCrest)), 0.0, 1.0);
        // SPLAT ROCK GATE DECOUPLED from the macro slopeRock (user 2026-06-11 'a lot of grass turning
        // into rocky patches again'): the user-calibrated global soft blend slopeRock [-0.6,1] puts
        // ~37% ROCK LAYER weight on perfectly flat ground, and the displacement-sharpened top-2
        // crossfade then flips whole displacement blobs to the rock PHOTO on grassland. The macro
        // color blend keeps the calibrated tone; the rock TEXTURE layer needs real slope: floor the
        // splat gate at 0.18 so flat/gentle land splats grass, rock starts on genuine slopes.
        // WIDENED TRANSITION (2026-06-13): srLo+0.2 -> srLo+0.4 so rock/grass and rock/sand/snow
        // boundaries have a wider blend band — the slope gradient between materials is no longer
        // a ~0.2-unit hard step but a ~0.4-unit gradual fade.
        float srLo = max(slopeRock.x, 0.05), srHi = max(slopeRock.y, srLo + 0.25);   // 0.18->0.10->0.05 lo; band +0.35->+0.25 (user 2026-06-14 repeated 'rock face angle more sensitive'): rock fully engages by a GENTLER slope. Stays above truly-flat (rockSlope~0).
        float wRockSlope = smoothstep(mix(srLo, 0.50, sandRegion), mix(srHi, 0.70, sandRegion), rockSlope);
        // WARPED BIOME BAND EDGES (user 2026-06-14: the snow/rock lines were STRAIGHT horizontal contours
        // viewed side-on; the vTexWarp domain warp was too low-freq (>1.8km waves = ~constant over one
        // mountain) so it had no visible effect). Use a dedicated 2-octave WORLD-DIR noise at mountain
        // scale (~1.3-3km waves) so EVERY elevation-keyed band wobbles +/-~700m vertically over a few km
        // = irregular natural lines, not level contours. World-dir keyed (seam-safe). Applied to snow,
        // the rock band, snowCold, and the beach below. highp dir (high-freq lattice needs the precision).
        float bandWarp = warpN * 450.0;   // reuse the shared warpN (defined at the beach gate above); +/-~700m undulation
        float snowHi   = smoothstep(snowEdges.x + bandWarp, snowEdges.y + bandWarp, vH);
        // ROCK BAND leading up to the snow (user 2026-06-14): a rocky belt ~0.4-2.2km below the snow
        // line so high mountains show rock between alpine grass and snow, not grass straight to snow.
        float rockBand = smoothstep(snowEdges.x - 2200.0 + bandWarp, snowEdges.x - 400.0 + bandWarp, vH) * (1.0 - snowHi);
        float wRock = max(wRockSlope, rockBand);
        float snowCold = (1.0 - smoothstep(0.30, 0.75, climate.z)) * smoothstep(snowEdges.x * 0.5 + bandWarp, snowEdges.x + bandWarp, vH);
        // POLAR/ICE-BIOME SNOW (user 2026-06-10 'put the snow texture on the snow'): the ICE biome
        // whitens cold lowland (biomeColor gate 1-smoothstep(0.10,0.18,tempEff)) at ANY elevation, but
        // wSnow was elevation-gated only -- polar snowfields were splatting the GRASS layer. Match the
        // effective-temperature ice gate (raw temp minus the elevation/latitude lapse the biome uses).
        float lat2 = asin(clamp(vWorld.y / max(length(vWorld), 1.0), -1.0, 1.0));
        float tempEff2 = clamp(climate.z - clamp(vH / 4500.0, 0.0, 0.55) * uBiomeBandBias
                                        - 0.18 * (abs(lat2) / 1.5708) * uBiomeBandBias, 0.0, 1.0);
        float iceClimate = (1.0 - smoothstep(0.10, 0.20, tempEff2)) * step(0.0, vH);   // no snow layer on the seabed
        float wSnow = clamp(snowHi + 0.7 * snowCold + iceClimate, 0.0, 1.0) * (1.0 - 0.6 * wRock);
        float wSand = sandRegion * (1.0 - wRock) * (1.0 - wSnow) * (1.0 - smoothstep(0.30, 0.70, slope));
        float wGrass = max(1.0 - wRock - wSnow - wSand, 0.0);
        vec4 w4 = vec4(wGrass, wRock, wSand, wSnow);   // layers 0..3 = grass,rock,sand,snow
        // BEACH BAND + UNDERWATER in one mask (user 2026-06-11): sand owns everything below the
        // beach ceiling uBeachTopM -- grass/snow weight folds into sand continuously through h=0 to
        // the seabed, so no vegetation can ever splat at or below the waterline (structural).
        // WIDENED (2026-06-13): 0.6->0.3 to match the macro beach transition, softer fade-out.
        float uwM = 1.0 - smoothstep(uBeachTopM * 0.3, uBeachTopM, vH);
        w4.z += (w4.x + w4.w) * uwM; w4.x *= 1.0 - uwM; w4.w *= 1.0 - uwM;
        w4 /= (w4.x + w4.y + w4.z + w4.w + 1e-4);
        // top-2 layer pick: 4-way blending would cost 24 taps; the gates are spatially near-exclusive,
        // so the two heaviest layers + a displacement-sharpened crossfade carry every transition.
        float lA = 0.0, wA = w4.x, lB = 0.0, wB = -1.0;
        if (w4.y > wA) { lB = lA; wB = wA; lA = 1.0; wA = w4.y; } else if (w4.y > wB) { lB = 1.0; wB = w4.y; }
        if (w4.z > wA) { lB = lA; wB = wA; lA = 2.0; wA = w4.z; } else if (w4.z > wB) { lB = 2.0; wB = w4.z; }
        if (w4.w > wA) { lB = lA; wB = wA; lA = 3.0; wA = w4.w; } else if (w4.w > wB) { lB = 3.0; wB = w4.w; }
        // tile-unit coords, FRAGMENT-ANCHORED snap (the rockO fp32 lesson): raw vWorld/tile loses
        // fp32 precision = visible UV stairs. Snap step = 1024 tiles exactly, so wt jumps by an
        // integer tile count across a snap boundary = identical wrapped sample (REPEAT), no phase
        // reset, camera-independent.
        highp float snapM = uTexTileM * 1024.0;
        highp vec3 wt = (vWorld - floor(vWorld / snapM) * snapM) / uTexTileM;
        // DOMAIN WARP anti-repetition (user 2026-06-10 'distort the textures so they dont look
        // repeated' + 'distort more, waves a bit large'): 2-octave world-dir warp -- ~11km waves at
        // +/-0.6 tile plus ~2.8km waves at +/-0.3 tile -- so repeats both shift AND shear locally;
        // no straight tile lattice survives. World-dir keyed -> seam-safe, camera-independent.
        // uTexWarp (__texWarp) scales the whole warp live.
        // VS-COMPUTED WARP, APPLIED EXACTLY ONCE (2026-06-12): vTexWarp carries the 3-octave halved-
        // frequency warp from the vertex shader (-9 snoise3/pixel). This is the ONLY place the warp
        // touches texture coordinates; every layer tap (albedo A/B, normal A/B, all four materials)
        // samples through this shared wt, so the warp cannot layer or double-apply.
        wt += vTexWarp * uTexWarp;
        vec3 tw = abs(n); tw = tw * tw; tw /= (tw.x + tw.y + tw.z + 1e-4);
        vec4 albA = surfTriTap(uSurfAlb, wt, tw, lA);
        vec3 nrmA = surfTriNrm(uSurfNrm, wt, tw, lA, n);
        float bAB = clamp(wA / max(wA + wB, 1e-4), 0.0, 1.0);
        vec4 texAlb = albA; vec3 texNrm = nrmA;
        if (wB > 0.02) {   // second layer only where a real transition exists (saves 6 taps elsewhere)
            vec4 albB = surfTriTap(uSurfAlb, wt, tw, lB);
            vec3 nrmB = surfTriNrm(uSurfNrm, wt, tw, lB, n);
            // displacement-sharpened transition -- WEIGHT-DOMINANT (user 2026-06-10 'large patches
            // of rock texture in mountains, not slope-keyed'): the old (hA-hB)*4 let the texture's
            // 2.4km displacement features decide the material outright wherever the gate weights sat
            // mid-range, so whole displacement blobs flipped to rock. The slope/snow/climate weight
            // now dominates (x3) and displacement only crisps the edge (x0.8) within the true
            // transition band -- material placement is the gates', texture only shapes the seam.
            // displacement term 0.8 -> 0.3 (user 2026-06-11 'still see bowls of rock texture'):
            // 0.8 still let the displacement photo's bowl-shaped blobs flip whole patches to rock
            // inside the transition band; 0.3 only feathers the seam edge.
            // SOFTENED (2026-06-13): transition sharpening reduced 3->1 so grass/rock/sand/snow
            // boundaries hold a natural blend band instead of a hard die-cut line. The 1.0 slope
            // still prefers the dominant layer but leaves a visible transition zone.
            float bSharp = clamp((bAB * 2.0 - 1.0) * 1.0 + (albA.a - albB.a) * 0.3 + 0.5, 0.0, 1.0);
            texAlb = mix(albB, albA, bSharp);
            texNrm = mix(nrmB, nrmA, bSharp);
        }
        float k = uTexMix * texFarFade;
        // macro-tinted detail (user 2026-06-10 'the textured patch must be tinted to the same shade
        // as the spot its replacing'): the texture contributes STRUCTURE + relative chroma only,
        // luminance-normalized onto the macro biome/climate color, so the splat never shifts the
        // shade of the ground it covers. uTexPhoto (default 0) can blend raw photo color back in.
        vec3 texC = texAlb.rgb;
        // LAYER-MEAN shade-match (user 2026-06-11 'dont see grass/snow textures' + 'terrain gets
        // darker'): dividing by the PER-PIXEL luminance cancelled all texture structure, and the
        // raw-photo blend that replaced it shifted the shade (the photos are darker than the macro).
        // Dividing by the layer's MEAN luminance (loader-computed uSurfMeanL) keeps every per-pixel
        // deviation visible while the patch average lands exactly on the macro shade.
        float mA = uSurfMeanL[int(lA + 0.5)];
        float texL = wB > 0.02 ? mix(uSurfMeanL[int(lB + 0.5)], mA, bAB) : mA;
        vec3 detail = texC * (albedo / max(texL, 0.02));
        // ROCK SHOWS THE TRUE PHOTO (user 2026-06-10 'we still see the original rock texture --
        // replace completely'): tinting rock to the macro shade just reproduced the old grey/tan,
        // so the rock layer takes the raw photo color; grass/sand/snow stay shade-matched.
        // raw-photo rock is NEAR-FIELD only (user 2026-06-11 'elevation and color mismatching'):
        // with mip-out the raw photo (mean lum .18, dark grey) was carrying to every distance, so
        // far mountains went dark instead of the macro tan. Fade photoF out as the texture detail
        // goes sub-pixel (pxWorld 60->240m); the far field returns to the shade-matched mip average.
        float nearTex = 1.0 - smoothstep(60.0, 240.0, pxWorld);
        float photoF = max(uTexPhoto, max(smoothstep(0.25, 0.6, w4.y) * 0.5, uTexPhotoNear) * nearTex);
        // MATERIAL IDENTITY (user 2026-06-12): the photo's OWN hue at the MACRO's luminance -- the
        // ground reads unambiguously as its material (grass green / sand tan / rock grey) without the
        // raw photo's darkness ('terrain gets darker' lesson) and without the macro's biome-brown
        // repaint ('neither grass nor sand' defect). Layer-mean normalized like `detail`.
        vec3 texIdent = texC * (dot(albedo, vec3(0.2126, 0.7152, 0.0722)) / max(texL, 0.02));
        albedo = clamp(mix(albedo, mix(detail, texIdent, photoF), k), 0.0, 1.0);
        // displacement-normal relief: WORLD-SPACE UDN perturbation from surfTriNrm (each projection
        // plane's tangent axes, not the radial frame). Amplitude capped low (scramble lesson d262b5e);
        // applied AFTER the uReliefShade exaggeration below so the exaggeration never amplifies it.
        texDn = texNrm * (uTexNrmK * k);
    }
#ifdef _DEBUGVIEW_
    // DIAG displayMode 7: raw river field -> blue where the river line fires, grey ridge field
    // elsewhere. Lets a witness SEE the drainage network + camera together to tune frequency
    // (not guess from screenshots). Pure diagnostic; no effect on the lit path.
    if (displayMode == 7) {
        float rv = riverMask(vWorld, vH, climate.z, climate.w, pxWorld);
        fragColor = vec4(mix(vec3(0.15), vec3(0.1,0.4,0.9), rv), 1.0); return;
    }
    // DIAG displayMode 8: RIVER GATING -- the ACTUAL integrated river the geometry+lit path uses (user
    // 2026-06-02: 'river gating is empty'). The old version re-evaluated riverMask per-pixel AND a dead
    // lake-suppression gated on humid>0.62 (most land is ~0.39) so it read ~0 = empty. Now it shows the
    // VS-integrated varyings directly: BLUE = vRiverWet (the flowing-water line the geometry carved +
    // the lit path composites), GREEN = vLakeWet (lake water), so it matches what actually renders. The
    // raw per-pixel network is still in mode 7.
    if (displayMode == 8) {
        vec3 col = vec3(0.12);
        col = mix(col, vec3(0.10, 0.45, 0.95), clamp(vRiverWet, 0.0, 1.0));   // river line (integrated)
        col = mix(col, vec3(0.15, 0.85, 0.55), clamp(vLakeWet, 0.0, 1.0));    // lake water
        fragColor = vec4(col, 1.0); return;
    }
    // DIAG displayMode 9: DISCRETE BIOME MAP -> each fragment flat-colored by its classified
    // biome (ocean/ice/tundra/desert/savanna/rainforest/taiga/forest/meadow). Lets a witness
    // COUNT contiguous biome regions + confirm logical placement (deserts in subtropics, ice at
    // poles, rainforest at equator), instead of eyeballing the blended lit palette.
    if (displayMode == 9) {
        fragColor = vec4(biomeClassColor(climate.z, climate.w, vH), 1.0); return;
    }
    // DIAG displayMode 10: CANYON field -> red where the gorge network fires (arid elevated only),
    // grey ridge field elsewhere. Lets a witness SEE the canyon network density independent of
    // landing the nadir exactly on a thin gorge line.
    if (displayMode == 10) {
        // ONE FRACTAL: canyonMask now samples the SAME canyonRidgeField as the VS carve (unified), so
        // the per-pixel network coincides with the geometry -- show the single full-res field (no more
        // overlaying canyonMask + vCanyonDep at two sample rates, which read as 'two resolutions').
        float cd; float cv = canyonMask(vWorld, vH, climate.z, climate.w, pxWorld, cd);
        // grey background -> orange canyon line by mask, deepening toward the gorge floor (cd).
        vec3 col = mix(vec3(0.15), mix(vec3(0.75,0.40,0.15), vec3(0.9,0.15,0.05), cd), cv);
        fragColor = vec4(col, 1.0); return;
    }
    // DIAG displayMode 11: CLIFF validation view -> RED = cliff/escarpment riser faces (vCliffFace),
    // GREEN = canyon walls (vCanyonDep on steep slope), so the user can SEE where cliffs are placed
    // independent of lighting/strata. Grey elsewhere. The witness for the cliff terrace placement.
    if (displayMode == 11) {
        float cliffR = vCliffFace;
        float canyonG = vCanyonDep * smoothstep(0.30, 0.55, slope);
        fragColor = vec4(mix(vec3(0.12), vec3(1.0, 0.2, 0.1), cliffR)
                       + vec3(0.0, 0.8, 0.2) * canyonG, 1.0); return;
    }
    // DIAG displayMode 12: PATCHES -> each LOD LEVEL a DISTINCT colour (user: validate the dense LOD
    // is always centered under the camera). Hue cycles by level via a golden-ratio rotation so any two
    // adjacent levels are far apart in hue; the finest (highest-level) patches under the camera read as
    // the hottest band, and the rings must be CONCENTRIC + CENTERED on the camera nadir. A thin dark
    // cell-grid (vGrid) overlays so individual quads are visible. Pure per-instance vLevel -> no field.
    if (displayMode == 12) {
        float h = fract(vLevel * 0.61803398875);            // golden-ratio hue per level (well-spread)
        // HSV->RGB at full sat/val for h, mid-V so text/grid reads.
        vec3 rgb = clamp(abs(fract(h + vec3(0.0, 0.6666667, 0.3333333)) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
        vec3 col = rgb * 0.85 + 0.1;
        // quad-cell grid overlay (same fwidth-AA lines as wireframe) so patch boundaries are crisp.
        vec2 g = vGrid * 24.0; vec2 gf = abs(fract(g) - 0.5); vec2 gw = fwidth(g) * 1.2;
        float line = 1.0 - min(smoothstep(0.0, gw.x, gf.x), smoothstep(0.0, gw.y, gf.y));
        col = mix(col, vec3(0.0), clamp(line, 0.0, 1.0) * 0.55);
        fragColor = vec4(col, 1.0); return;
    }
#endif // _DEBUGVIEW_ (modes 7-12)
    // Material = the biome height/slope/climate ramp (provably 0 edges). The ortho atlas + per-pixel
    // detail texturing are both REMOVED (GPU one-fractal): closeup detail comes from the mesh
    // subdividing into a denser sample of the single VS height fractal, never a tile sample or grain.
    if (displayMode == 4) { fragColor = vec4(albedo, 1.0); return; } // biome ramp
    if (displayMode == 2) { fragColor = vec4(albedo, 1.0); return; }

    // ---- Bruneton-style atmospheric lighting (ported analytic single-scatter) ----
    // Map the terrain point + camera into atmosphere space (km, surface=ATM_BOTTOM).
    highp vec3 pAtm   = atmPos(vWorld, terrainR);    // W7: km-scale (~6360) atmosphere coords -> highp for the AP segment precision
    highp vec3 camAtm = atmPos(camWorld, terrainR);
    // GENTLE-SLOPE RELIEF SHADING (user 2026-06-10 'even gentle slopes shaded so terrain elevation
    // is obvious'): exaggerate the tangential tilt of the LIT normal only -- the material gates keep
    // the true geometric n, so placement is unchanged; lighting contrast on rolling ground increases.
    if (vH > -2.0 && uReliefShade > 1.0) {
      // Relief exaggeration of the LIT normal's tangential tilt. The old SCREEN-SPACE dFdx(vH) term was
      // REMOVED (user 2026-06-14 'jagged normals, still there on the normals view'): dFdx/dFdy of the
      // INTERPOLATED vH varying is CONSTANT PER TRIANGLE and discontinuous at every triangle edge =
      // hard facets baked straight into the lit normal -- the jaggedness root, not the vertex normal.
      // It was a workaround for when vNrm ~= uz on gentle ground; the central-difference vNrm now
      // captures gentle slopes (real gradient), so (nLit-uz) is non-zero on real slopes and the plain
      // exaggeration is legible without the faceted screen-space hack.
      nLit = normalize(uz + (nLit - uz) * uReliefShade);
    }
    // photo-texture detail normal at its calibrated amplitude, OUTSIDE the relief exaggeration.
    // ADDED in world space (the old `- ux*dn.x - uy*dn.y` subtracted an already-negated Sobel
    // normal in a frame the triplanar RG was never expressed in -- the fundamental both-at-once
    // math bug: bumps lit from wrong directions everywhere the splat is active).
    nLit = normalize(nLit + texDn);
    vec3 nAtm   = nLit;   // relief-exaggerated normal for lighting (macro n still drives material)
#ifdef _DEBUGVIEW_
    // Normals view shows the FINAL lit normal (geometry + relief exaggeration + texture detail
    // normal) -- the normal the lighting actually uses, texture normals included.
    if (displayMode == 1) { fragColor = vec4(nAtm * 0.5 + 0.5, 1.0); return; }
#endif

    // ANIMATED OCEAN BRANCH REMOVED (user 2026-06-11 'the ocean should be a separate surface'):
    // sub-sea fragments are the real SEABED now (sand/rock, lit as land below); the animated water
    // shading moved to the uIsWater=1 water-surface pass (early return at the top of main), alpha-
    // blended over this seabed. Sea ice still keys the seabed albedo near the poles.

    vec3 skyIrr;
    vec3 sunIrr;
    if (uUnderwater > 0.5) {
        // Underwater terrain (seabed): sun attenuated through water column, blue ambient, no
        // atmospheric Rayleigh/Mie scattering. Sun colour shifts toward green-blue with depth.
        float depth = max(0.0, terrainR - length(camWorld));
        float atten = exp(-depth * 0.0004);
        float ndl = max(dot(nAtm, sunDir), 0.0);
        sunIrr = vec3(1.0, 0.65, 0.35) * atten * ndl * 0.7;
        skyIrr = vec3(0.02, 0.07, 0.18);
    } else {
        sunIrr = atm_sunSkyIrradiance(pAtm, nAtm, sunDir, skyIrr);
    }
    // The Rayleigh sky irradiance is strongly blue; at full strength it tints the lit
    // CONTINENTS blue-grey from orbit (witnessed: lit mean [83,95,112] vs warm albedo
    // [90,83,74] at 2000km -- the blue is the sky-irradiance term, NOT aerial inscatter).
    // Partially DESATURATE skyIrr toward its luminance so it still brightens shadowed/
    // ambient regions without washing the land's true color blue. Land albedo dominates;
    // a modest blue ambient remains (physically the sky does add some blue fill).
    float skyL = dot(skyIrr, vec3(0.2126, 0.7152, 0.0722));
    vec3 skyIrrBalanced = mix(vec3(skyL), skyIrr, 0.35);   // 0 = grey, 1 = full blue sky
    // RELIEF RECOVERY (Real-World Look): the bright flat sky fill washed N.sun ridge/valley contrast
    // to ~flat. CUT the fill (uSkyFill, was implicit 1.0) + cool-tint it so shadows read sky-blue, so
    // the direct sun becomes the dominant key and relief self-shading actually shows.
    skyIrrBalanced *= uSkyFill * vec3(0.85, 0.92, 1.10);
    // Lambert BRDF (albedo/PI) under sun + (balanced) sky irradiance. Keep an ambient
    // FLOOR so night/shadow faces are never pure black (PRD lit-terrain-dark-black).
    // CURVATURE / SLOPE AMBIENT-OCCLUSION (canyon floors + cliff bases): incised/concave fragments
    // see less sky, so darken the AMBIENT (sky + floor) there for contact-shadow depth. Keyed on the
    // interpolated canyon gorge depth (vCanyonDep ->1 in the gorge core) so floors darken while rims
    // stay bright. Direct SUN is NOT occluded (avoids double-darkening already-shaded faces). uAoAmt
    // lever. Pure fn of the carve varying -> no extra sampling, seam-safe.
    // OBJECT-SPACE AO (2026-06-05): augment the gorge-depth term with a cheap analytic sky-occlusion
    // estimate so cliff BASES and steep concave faces -- not just canyon floors -- pick up contact
    // shadow. A true N-sweep horizon-angle integral over the fractal (karim.naaji.fr/lsao) is the
    // research-best but per-pixel fractal sweeps would re-add the close-up VS/FS cost the pipeline is
    // sensitive to; instead approximate the sky-view factor from data already in hand: (a) the gorge
    // depth varying (deep incision -> walls occlude sky), and (b) surface STEEPNESS (a near-vertical
    // face sees only a hemisphere's edge of sky -> ~half occlusion at slope->1). Both darken AMBIENT
    // only (never the direct sun, avoiding double-shade). uAoAmt lever. Pure fn of varyings+normal ->
    // no extra sampling, seam-safe, zero FPS cost vs the old single-term version.
    float aoAmt    = (uAoAmt > 0.0 ? uAoAmt : 1.0);
    float gorgeAO  = 0.0;   // canyon AO decoupled (user 2026-06-10: canyons impose ELEVATION only, no material/AO keying)
    float slopeAO  = smoothstep(0.05, 0.55, slope) * 0.35;              // gentle-slope AO reveals midday relief (0.20->0.90 from 0.45->0.95, peak 0.35)
    float cliffAO  = 1.0 - min(gorgeAO + slopeAO, 0.75) * aoAmt;        // cap so faces never go black
    // STRONGER NORMAL-DRIVEN SHADING (user 2026-06-03: 'normals arent affecting the lit view properly').
    // The lit relief was too subtle (witnessed litON SD 6.4 vs flat 5.9 = only ~8% contrast) because
    // sunIrr (the N.sun direct term) was diluted by the flat sky ambient + the 0.06 albedo floor. Cut
    // the ambient floor (0.06->0.03) so shaded slopes go genuinely darker, and BOOST the directional
    // sun term (x1.6) so the N.sun slope contrast reads clearly. The (1/PI) Lambert norm + ACES tonemap
    // keep it from blowing out; a small floor still prevents pure-black shadow faces.
    // sunIrr boost 1.6 -> 1.25 + ambient floor 0.03 -> 0.04 (user 2026-06-03 screenshot: the 1.6 boost
    // BLEW OUT high/bright rock to flat white, losing relief detail). 1.25 keeps the slope contrast
    // strong (N.sun still drives the shading) without clipping the highlights to white on lit rock.
    // AMBIENT FLOOR lifted 0.025->0.05 (live-deck witness mut-1780681995401: with the analytic+fs normal ON
    // HALF the deck frame crushed to NEAR-BLACK -- mean 151->91, n898k->434k -- so relief read as black
    // HOLES not legible rugged texture). A genuine shadowed rock face is dark grey-blue (skylight), never
    // pure black; the higher floor keeps micro-relief readable as texture while the direct sun still drives
    // the ridge/valley key contrast. Tied to cliffAO so true contact-shadow concavities still darken.
    // PER-VERTEX SHADING (DEFECT 2): fold the interpolated vShadeAO crease/micro-relief occlusion into
    // the ambient term so geometry reads relief even where the FS detail-normal is faint. nearFade-gated
    // (full at the deck, ->1 i.e. no effect at orbit) so the macro/orbit look is unchanged. Combines
    // multiplicatively with cliffAO -- both are sky-occlusion factors that should compose.
    // W5: vShadeAO (per-vertex, from the deleted VS gradient) RECOMPUTED here from the Sobel slope (user
    // 2026-06-07 'recompute AO from Sobel'). creaseAO = slope*0.45 (steep landform -> valley/crease
    // occlusion); microAO from the fine per-pixel gslope. uVertexAO lever + the 0.45 floor preserved.
    float fsShadeAO = clamp(1.0 - (slope * 0.55 + gslope * 0.40) * uVertexAO, 0.50, 1.0);
    // FADE-IN FIX (user 'a layer fades in at close distance'): vAO was mix(1.0, fsShadeAO, nearFade) where
    // nearFade rises 0->1 as pxWorld shrinks 180->8m on approach -> the Sobel crease-AO darkening animated
    // IN as the camera neared = the visible 'layer fading in'. fsShadeAO is a per-pixel SOBEL-slope quantity
    // (not sub-pixel noise) so it does NOT alias and needs no distance gate -- present at all distances, no
    // pop. This is the nearFade gate the P5 unification missed on the shading-AO path. nearFade now unused.
    float vAO = fsShadeAO;   // crease-AO present at all distances (P9, no fade-in pop)
    float skyAO = cliffAO * vAO;
    // SHADOW-FACE FILL (user 2026-06-09: 'shadows are still full black'). On a day-side face pointing AWAY
    // from the sun, sunIrr->0 so ONLY this floor lights it. The old floor (albedo*0.05*cliffAO*vAO) was
    // crushed to ~0 on dark-albedo rock AND on steep/concave faces (cliffAO*vAO -> 0 exactly where shadows
    // are) -> black shadows. FIX: (a) an ALBEDO-INDEPENDENT additive sky term so even dark rock never goes
    // black, (b) DROP the cliffAO/vAO multipliers from the floor (they belong on the sky KEY, not the
    // guaranteed floor), (c) lift the albedo-tinted part to 0.22. Result: shadowed faces settle to a dim
    // cool-grey, never black; the direct sun still drives the lit/shadow contrast above this floor.
    // The 0.22 flat lift (no AO) flooded shadowed ANGLED faces to a uniform cool-grey, washing out the
    // relief (user 2026-06-10: 'angled faces turned to rock/grey'). Restore directional shading: scale the
    // albedo-tinted part by a SOFTENED sky exposure (mix(0.45,1.0,skyAO) -- never crushes to 0 like the old
    // cliffAO*vAO did, but lets faces that see less sky read darker = the relief returns), and lower the lift
    // 0.22->0.14. The albedo-INDEPENDENT flat sky floor stays as the never-black guarantee (the 9cfa643 win).
    vec3 ambientFloor = albedo * (0.14 * mix(0.45, 1.0, skyAO)) + vec3(0.020, 0.026, 0.038);
    // sky AO no longer SQUARED (was cliffAO*cliffAO): the squared sky term over-crushed shadowed micro-
    // relief to black at the deck. Single cliffAO still darkens concavities for depth, but shadowed
    // faces retain sky fill so the analytic normal reads as rugged texture, not black blotches.
    // (the old vH<-2 ocean bypass is gone -- the seabed is land now, lit like land; the water
    // surface is its own pass.)
    vec3 lit = albedo * (sunIrr * 1.25 + skyIrrBalanced * skyAO) * (1.0/ATM_PI) + ambientFloor;

    // AERIAL PERSPECTIVE (analytic single-scatter, re-added 2026-06-05 for the space->ground depth
    // cue -- the single biggest missing realism term for the seamless descent; ref Hillaire-2020,
    // LUT-free inline form). The OLD removed version was a flat 15km horizon haze band the user
    // disliked; this is physically-grounded instead: integrate Rayleigh+Mie in-scatter along the
    // EXACT camera->fragment segment and attenuate the lit color by the segment transmittance, so
    // the haze is PROPORTIONAL to how much atmosphere the view ray actually crossed. DISTANCE-GATED
    // hard at both ends: ~zero within a few km (FPS-ground stays crisp, no over-haze the user hated)
    // and ramping in only across tens-of-km+ (the depth cue that makes distant ridges recede and the
    // space->ground transition read as real). N=8 march (cheaper than the sky pass's 16; the surface
    // term tolerates it), single draw, reuses atm_* -> zero assets.
    vec3 color = lit;
    {
        highp vec3 camA  = atmPos(camWorld, terrainR);     // W7: km-scale -> highp
        highp vec3 segKm = pAtm - camA;                    // W7: camera->fragment in km (cancellation of two km-scale points)
        highp float dKm  = length(segKm);
        // gate: 0 below ~3km path, ramping to full by ~120km. keeps ground crisp, distance hazed.
        float apGate = smoothstep(3.0, 120.0, dKm);
        if (apGate > 0.002) {
            vec3 vRay = segKm / max(dKm, 1e-4);
            const int APN = 8;
            float apdt = dKm / float(APN);
            vec3 inscatR = vec3(0.0), inscatM = vec3(0.0);
            float odR = 0.0, odM = 0.0;                    // optical depth camera->sample
            for (int i = 0; i < APN; i++) {
                highp vec3 p = camA + vRay * (apdt * (float(i) + 0.5));   // W7: km-scale march point -> highp
                highp float pr = length(p);
                float dR = atm_rayleighDensity(pr) * apdt;
                float dM = atm_mieDensity(pr) * apdt;
                odR += dR; odM += dM;
                vec3 tView = exp(-(ATM_RAYLEIGH * odR + ATM_MIE_EXT * odM));
                vec3 tSun  = atm_transmittanceToSun(p, sunDir);
                vec3 t = tView * tSun;
                inscatR += t * dR;
                inscatM += t * dM;
            }
            float nu = dot(vRay, sunDir);
            vec3 apTrans = exp(-(ATM_RAYLEIGH * odR + ATM_MIE_EXT * odM));
            vec3 apInscat = ATM_SOLAR_IRRADIANCE * (
                inscatR * ATM_RAYLEIGH * atm_rayleighPhase(nu) +
                inscatM * ATM_MIE      * atm_miePhase(ATM_MIE_G, nu));
            // HORIZON HAZE FLOOR (Real-World Look): real aerial perspective fades distant terrain toward
            // a pale blue-grey sky, NEVER to black. When apTrans collapses (long limb path) the physical
            // single-scatter inscatter can dip in low-phase directions, leaving a black silhouette band.
            // Floor the fill with a multiple-scatter-style skylight ambient (1-apTrans) so the far ridge
            // dissolves into haze instead of going dark. Tied to uSkyFill so it's terraformable.
            vec3 skyHaze = uSkyFill * vec3(0.40, 0.55, 0.78) * (1.0 - apTrans);
            apInscat = max(apInscat, skyHaze);
            // attenuate the surface by transmittance and add the in-scatter, both faded by the gate
            // so the near field is untouched (apGate~0 -> mix back to the un-hazed lit color).
            vec3 hazed = lit * apTrans + apInscat;
            // TERMINATOR SUNSET REDDENING (Real-World Look): add a warm Rayleigh-out rim where the sun
            // GRAZES the limb (strictly gated on 1-|N.sun| near 0 so it never bleeds onto the full day
            // side). Rides the existing AP inscatter -- zero extra march. uTerminatorGlow lever.
            // peak at the terminator (N.sun~0), fall off toward both day-center and night-center, and
            // SQUARE it so the warm band is tight at the day/night line instead of a fat ring round the
            // whole limb. Warm amber (not pure red) so it reads as sunset haze, not a bruise.
            float gz = 1.0 - abs(dot(normalize(vWorld), sunDir));
            float graze = smoothstep(0.55, 1.0, gz); graze *= graze;
            // only on the LIT side of the terminator (mu>0) -- past the line the surface is night and
            // additive amber on near-black extinct land reads as a scorched maroon rim. Day-gate kills it.
            float termDay = smoothstep(-0.02, 0.18, dot(normalize(vWorld), sunDir));
            hazed += uTerminatorGlow * graze * termDay * vec3(1.0, 0.55, 0.34) * apGate;
            color = mix(lit, hazed, apGate * uHazeMul);   // uHazeMul lever (2026-06-10 'pale hazy': full-strength haze milked the midground)
        }
    }
    // UNDERWATER FOG (camera below sea level): replace the air-based Aerial Perspective with
    // absorption/scattering through water. Fog colour deepens with camera depth so the seabed
    // fades to blue-green with distance, red absorbed first. Applied only to terrain (uIsWater<0.5).
    if (uUnderwater > 0.5 && uIsWater < 0.5) {
        highp vec3 camA  = atmPos(camWorld, terrainR);
        highp vec3 segKm = pAtm - camA;
        highp float dKm  = length(segKm);
        float depth = max(0.0, terrainR - length(camWorld));
        // Water absorption coefficients (per km): red attenuates fastest, blue slowest.
        vec3 absorb = vec3(4.0, 0.8, 0.3) * (1.0 + depth * 0.0003);
        vec3 uwTrans = exp(-absorb * dKm);
        vec3 uwFog = vec3(0.002, 0.06, 0.16) + vec3(0.0, 0.02, 0.04) * depth / 1000.0;
        color = mix(color * uwTrans + uwFog * (1.0 - uwTrans), uwFog, smoothstep(50.0, 500.0, dKm * 1000.0));
    }
    // RIVERS post-lighting (witnessed browser-2115/2118: the river-blue in ALBEDO is multiplied
    // by the warm sun irradiance (sunIrr.b is low) so land rivers lose their blue in the lit
    // result -- 15k blue albedo px -> 0 lit px, even with the sun HIGH (sunDotUp 0.85). Ocean
    // sidesteps this by bypassing the irradiance multiply (vH<0 branch). So rivers, like open
    // water, must be composited AFTER lighting: blend the post-lit color toward a sun-scaled
    // water-blue on the river line. ndlSun keeps the river shaded by the sun so it is not flat.
    // INLAND WATER = REAL WATER, SHADED LIKE THE OCEAN (user 2026-06-01i: 'it should use the same
    // content as the sealevel to draw because its also water'). Where vWaterDepth>0 (the flat water
    // plane sits above the carved floor -> genuinely submerged) we run the SAME water model as the
    // ocean branch: fresnel sky reflection + sun glint + shallow->deep depth tint, composited AFTER
    // lighting (the warm sun-irradiance multiply would otherwise kill the blue, the reason the ocean
    // bypasses it via vH<0). Gating on vWaterDepth (NOT the whole carve mask) makes the water LINE UP
    // with the flat carved surface -- the graded erosion banks above the waterline stay dry land.
    if (vH >= 0.0 && vWaterDepth > 0.0) {
        // submergence 0..1 over ~0..40m -> shoreline (thin water, terrain shows through) to deep.
        float sub = clamp(vWaterDepth / 40.0, 0.0, 1.0);
        // wave-perturbed water normal (small inland ripple), same tangent frame as the ground.
        highp vec3 wOriginL = floor(camWorld / 1024.0) * 1024.0;   // W7: ~6.4e6 m snapped anchor -> highp
        highp vec2 wpL = vec2(dot(vWorld - wOriginL, ux), dot(vWorld - wOriginL, uy));   // W7: highp camera-relative wave coord
        vec2 slopeL = oceanWaveSlope(wpL, oceanTime) * 0.5;            // calmer than open ocean
        highp float wDistL = length(camWorld - vWorld);              // W7: camera->fragment cancellation -> highp
        slopeL *= clamp(1.0 - wDistL / 4000.0, 0.0, 1.0);             // fade ripple at distance (anti-alias)
        vec3 wnL = normalize(uz - ux * slopeL.x - uy * slopeL.y);
        vec3 viewL = normalize(camWorld - vWorld);
        float f0L = 0.02;
        float fresL = f0L + (1.0 - f0L) * pow(clamp(1.0 - max(dot(wnL, viewL), 0.0), 0.0, 1.0), 5.0);
        vec3 hlL = normalize(sunDir + viewL);
        float specL = pow(max(dot(wnL, hlL), 0.0), 220.0);
        float ndlL = max(dot(wnL, sunDir), 0.0);
        // depth tint: a touch greener than the ocean (freshwater/sediment), shallow -> deep.
        vec3 shallowL = vec3(0.10, 0.30, 0.36);
        vec3 deepL    = vec3(0.02, 0.10, 0.20);
        // congruent with the ocean: per-channel Beer-Lambert by true submerged depth (freshwater
        // slightly murkier), so lakes/rivers and the ocean read as one water system.
        vec3 TL = exp(-uOceanK * vWaterDepth * 1.6);
        vec3 waterBaseL = mix(deepL, shallowL, TL);
        vec3 skyColL = vec3(0.30, 0.42, 0.55);
        vec3 waterLitL = waterBaseL * (0.25 + 0.75 * ndlL);
        vec3 inlandWater = mix(waterLitL, skyColL, fresL * 0.85) + vec3(1.0, 0.95, 0.85) * specL * ndlL;
        inlandWater += waterBaseL * 0.05;                             // ambient floor (never black)
        // blend over the lit ground: full water in deep parts, terrain shows through at the shoreline.
        float waterCover = sub;                                      // 0 at waterline -> 1 deep
        color = mix(color, inlandWater, waterCover);
    }
    // WIREFRAME OVERLAY (uWireframe=1): draw the per-quad mesh-cell grid lines so the LOD
    // tessellation is visible. Uses the parametric vGrid coord (0..1 per tile, GRID cells) with an
    // fwidth-based anti-aliased line at each cell boundary -> crisp wireframe at any distance,
    // independent of the triangle topology (WebGL2 has no glPolygonMode). Drawn pre-tonemap so the
    // lines tonemap with the surface.
    if (uWireframe > 0.5) {
        vec2 g = vGrid * 16.0;                 // GRID cells per tile edge (gridMeshSize 16; was 24, FPS lever)
        vec2 gf = abs(fract(g) - 0.5);         // distance to nearest cell line in cell units
        vec2 gw = fwidth(g) * 1.2;             // line half-width in cell units (AA)
        float line = 1.0 - min(smoothstep(0.0, gw.x, gf.x), smoothstep(0.0, gw.y, gf.y));
        color = mix(color, vec3(0.05, 1.0, 0.4), clamp(line, 0.0, 1.0) * 0.85);
    }
    // MACRO GLOBE SHADING (user 2026-06-03: at the DEFAULT ORBIT view the planet read as a FLAT MAP --
    // a luma scan across the disc was ~uniform, no terminator/limb darkening, because the OCEAN bypasses
    // the sun term and the high exposure pushed land into the tonemap's flat bright shoulder). Apply the
    // SPHERE sun angle to EVERY fragment (land + ocean) so the globe reads 3D: bright at the sub-solar
    // point, falling to dark across the day side into the terminator. dayShade = smoothstep over the
    // surface-normal . sun, with a small ambient floor (0.10) so the night/terminator isn't pure black.
    float macroMu = dot(normalize(vWorld), sunDir);
    // SOFT FLOORED TERMINATOR (Real-World Look): a wider, smoother twilight band (uTermWidth) with a
    // night floor (uNightFloor) so framed night longitudes keep faint detail instead of crushing to
    // black (defect #4). The old hard 0.10..-0.12,0.55 band was too steep + too dark.
    float dayShade = mix(uNightFloor, 1.0, smoothstep(-uTermWidth, uTermWidth, macroMu));
    // LIMB DARKENING: grazing-view fragments (the disc EDGE from orbit) darken, giving the globe its
    // rounded 3D form instead of a flat disc. viewGraze = surface-normal . view-dir, ->1 facing the
    // camera (disc centre), ->0 at the limb. Only bites near the limb so it does not dim the main face.
    vec3 viewDir = normalize(camWorld - vWorld);
    float viewGraze = max(dot(normalize(vWorld), viewDir), 0.0);
    float limb = 0.45 + 0.55 * smoothstep(0.0, 0.45, viewGraze);   // 0.45 at the limb -> 1.0 facing camera
    // NIGHT / SHADOW FILL (user 2026-06-09: 'we want night lights in the SHADOW areas, not city lights'):
    // a UNIFORM dim fill that lifts the unlit hemisphere out of pure black -- a cool ambient night light,
    // no clusters, no warm dots, no land gate (it lifts every dark fragment, ocean + land). uNightLights
    // is the live intensity dial (0 = off, back to near-black night). The day-lit side (dayShade->1) is
    // unaffected because the fill is weighted by (1-dayShade). Raised from the old earthshine 0.012/0.018/
    // 0.032 (still read as black) to a clearly-visible dim blue so framed night terrain stays legible.
    vec3 nightFill = vec3(0.06, 0.075, 0.11) * uNightLights;
    vec3 color2 = (color * dayShade + nightFill * (1.0 - dayShade));
    // HDR -> SDR: ACES tonemap. Exposure 1.7 -> 1.25 so the N.sun + dayShade gradient SPREADS across the
    // tonemap's linear range instead of clipping flat to the bright shoulder (the orbit-flatness root).
    vec3 c = color2 * uExposure;   // exposure 1.25->uExposure(1.0): stop the bright ACES-shoulder wash
    vec3 mapped = clamp((c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14), 0.0, 1.0);
    // POST-ACES LOOK (Real-World Look): restore saturation + contrast DELIBERATELY in display-linear so
    // the result is darker-but-saturated, not bright-but-pastel. Pull saturation up (uLookSat) + an
    // S-curve contrast about mid-grey (uLookContrast), then gamma. Live-tunable.
    float lum = dot(mapped, vec3(0.2126, 0.7152, 0.0722));
    mapped = mix(vec3(lum), mapped, uLookSat);
    mapped = clamp((mapped - 0.5) * uLookContrast + 0.5, 0.0, 1.0);
    fragColor = vec4(pow(mapped, vec3(1.0/2.2)), 1.0);
}
#endif

// ---- HEIGHT PROBE (collision): compute the EXACT rendered terrain height for one world dir,
// reusing the same hpfSample + broadShapeM the mesh VS uses, so the free-fly collision floor
// can never diverge from the rendered surface (user-chosen: read px from the GPU, not a CPU
// mirror). Paired with a 1-point VS in gl-render.js; writes height (metres) to R32F.
#ifdef _PROBE_
uniform vec3 probeDir;     // world direction under the camera (normalized)
out vec4 probeOut;
void main(){
    vec3 dir0 = normalize(probeDir);
    // THC-Normal W1/W7: the collision height is now the SAME composeHeight() the geometry bake uses,
    // so sampleGroundM returns EXACTLY the rendered surface -- no parallel mirror to drift (was a
    // hand-copied carve sequence that OMITTED vtxDisplace = the camera-stops-short gap). Reconstruct
    // the face-local metres from the probe world dir (max-axis cube projection -> warped face metres)
    // so vtxDisplace samples the identical micro-relief field the VS does. tileM = a deck-leaf size so
    // the Nyquist fade keeps every octave the close-up geometry shows (collision == visible surface).
    int face; vec2 uv; hpfFaceUV(dir0, face, uv);
    // hpfFaceUV INVERTS the cube projection that built dir0, so (uv*2-1)*defRadius IS ALREADY the
    // warped face-local metres the mesh VS passes to composeHeight (terrain.glsl:667: faceWarp(...)).
    // The old line wrapped this in faceWarp() a SECOND time = a double-warp -> the probe sampled a
    // SHIFTED vtxDisplace noise-lattice position than the VS = a per-probe micro-relief height delta
    // (collision-elev-facelocal-parity, workflow wb4syopmo). Drop the extra warp: probe lattice == VS lattice.
    highp vec2 faceLocal = (uv * 2.0 - 1.0) * defRadius;             // W7: warped face-local metres ~6.4e6 -> highp (matches VS:667)
    // tileM is INERT in vtxDisplace (terrain.glsl:495-501 has no tileM term; PURE per-patch-step fix) so
    // composeHeight is bit-identical for any tileM -> the hardcoded value below is NOT a divergence (verified).
    highp float h = composeHeight(dir0, faceLocal, 64.0);            // W7: metres (tileM inert; collision == rendered surface)
    // RAW height out (2026-06-11): the old smoothstep(-60,60) flat-ocean clamp made the probe a
    // CLAMPED GAUGE -- sampleGroundM could never report real bathymetry (witnessed: a 2000-dir
    // sweep found 'no ocean below -100m' on a planet with km-deep basins), silently corrupting
    // every depth-based witness. Collision still floats on the water: BOTH movement consumers
    // clamp at use (planet.html Math.max(0, sampleGroundM(...)) at the move-step and ground-track
    // sites), which is where a USE-specific policy belongs -- the instrument now reports truth.
    probeOut = vec4(h, 0.0, 0.0, 1.0);
}
#endif

#ifdef _HEIGHTBAKE_
// THC HEIGHT-CACHE BAKE (2026-06-14): render ONE tile's composeHeight into a height-pool texel grid.
// Each fragment = one parametric (u,v) of the tile; the dir + faceLocal mapping is IDENTICAL to the
// mesh VS (faceWarp(vertex.xy*defOffset.z+defOffset.xy), normalize(frame*vec3(faceLocal,defRadius)))
// so the baked height matches the procedural geometry exactly. NON-DESTRUCTIVE: this is a separate
// program; the live VS still computes composeHeight until thc-vs-sample switches it to a pool sample.
uniform mat3 uBakeFrame;    // face-local -> world for this tile's face (== faceFrame(face))
uniform vec4 uBakeOffset;   // tile (ox, oy, l, level) face-local metres (== iOffset)
uniform float uBakeRes;     // bake grid resolution (texels per edge)
out vec4 bakeOut;
void main(){
    highp vec2 uv = (gl_FragCoord.xy - 0.5) / max(uBakeRes - 1.0, 1.0);   // texel -> parametric [0,1]
    highp vec2 faceLocal = faceWarp(uv * uBakeOffset.z + uBakeOffset.xy);
    highp vec3 dir0 = normalize(uBakeFrame * vec3(faceLocal, defRadius));
    highp float h = composeHeight(dir0, faceLocal, uBakeOffset.z);
    bakeOut = vec4(h, 0.0, 0.0, 1.0);
}
#endif
