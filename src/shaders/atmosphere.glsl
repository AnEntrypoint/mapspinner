// src/shaders/atmosphere.glsl  (#version 300 es is prepended by the JS loader)
// -----------------------------------------------------------------------------
// WebGL2 analytic Rayleigh/Mie single-scatter atmosphere for the planet.
//
// PORT NOTE — what this is vs. the WebGPU path:
//   planet.html runs the AUTHORITATIVE Bruneton model: it BAKES the transmittance/
//   scattering/irradiance LUTs at runtime with WebGPU COMPUTE shaders (3D textures,
//   multiple-scattering orders 2..4) and then samples them with a_GetSkyRadiance().
//   WebGL2 has NO compute shaders, and the baked LUTs live in a *separate* WebGPU
//   GPUDevice/context that cannot be shared into the WebGL2 context. Re-baking the
//   full 3D multiple-scattering LUTs with WebGL2 fragment passes is a large, fragile
//   undertaking. Per the task's explicit fallback, this file implements a FAITHFUL
//   analytic single-scatter atmosphere instead: same atmosphere geometry (bottom
//   6360 km, top 6420 km), the same Rayleigh/Mie scattering coefficients and phase
//   functions and solar irradiance Bruneton uses, and an analytic optical-depth
//   transmittance. It produces a real limb/halo, sky color gradient, a sun disc,
//   and sun+sky irradiance on the terrain. SIMPLIFIED vs. baked Bruneton:
//     - single scattering only (no orders 2..4 multiple scattering),
//     - analytic exponential optical depth (chapman-ish) instead of LUT transmittance,
//     - sky ambient from a hemispheric Rayleigh approximation instead of the irradiance LUT.
//
// Units: positions passed in are ATMOSPHERE-SPACE kilometres (surface at ATM_BOTTOM
// = 6360). atmPos() maps the meter-scale render world (R = 6360000 m) into this space.
// -----------------------------------------------------------------------------

const float ATM_PI = 3.14159265358979;

// Atmosphere geometry (km) — matches ATMO_DEF used to bake the WebGPU LUTs.
const float ATM_BOTTOM = 6360.0;
// Visible shell thickened to ~140 km (vs Bruneton's 60 km) so the limb/halo reads
// as a real atmosphere ring from orbit instead of a sub-pixel sliver. SIMPLIFICATION
// vs. the baked Bruneton LUTs (which use 6420). Scale heights raised to match so the
// density still falls off smoothly across the thicker shell.
const float ATM_TOP    = 6500.0;
const float ATM_RAYLEIGH_H = 18.0;   // Rayleigh scale height (km), inflated for the halo
const float ATM_MIE_H      = 4.0;    // Mie scale height (km)
const float ATM_MIE_G      = 0.8;

// Bruneton scattering coefficients (per-km) and solar irradiance (W/m^2/nm-ish).
const vec3  ATM_RAYLEIGH = vec3(0.005802, 0.013558, 0.0331);
const vec3  ATM_MIE      = vec3(0.003996, 0.003996, 0.003996);
const vec3  ATM_MIE_EXT  = vec3(0.003996) * (1.0 / 0.9); // Mie extinction = scat/ssa, ssa~0.9
const vec3  ATM_SOLAR_IRRADIANCE = vec3(1.474, 1.8504, 1.91198);
const float ATM_SUN_ANGULAR_RADIUS = 0.004675;

// Map a render-space world position (meters, surface radius R_m) into atmosphere
// space (km, surface at ATM_BOTTOM). Preserves altitude fractions across the shell.
// W7: worldMeters ~6.4e6 m + R_m ~6.4e6 m both overflow fp16 -> highp (terrain.glsl injects this file
// into the FS, whose global default precision is now mediump). The km-scale result narrows naturally.
vec3 atmPos(highp vec3 worldMeters, highp float R_m) {
    return worldMeters * (ATM_BOTTOM / R_m);
}

float atm_rayleighPhase(float nu) { return (3.0/(16.0*ATM_PI)) * (1.0 + nu*nu); }
float atm_miePhase(float g, float nu) {
    float k = 3.0/(8.0*ATM_PI) * (1.0-g*g)/(2.0+g*g);
    return k * (1.0+nu*nu) / pow(max(1.0 + g*g - 2.0*g*nu, 1e-4), 1.5);
}

// Distance from a point at radius r along direction with cosine mu to the top shell.
// Returns -1 if the ray escapes without entering the atmosphere from outside.
// W7: r ~6360 km so r*r ~4e7 OVERFLOWS fp16 (max 65504) -> r + the disc quadratic MUST be highp.
float atm_distToTop(highp float r, float mu) {
    highp float disc = r*r*(mu*mu - 1.0) + ATM_TOP*ATM_TOP;
    if (disc < 0.0) return -1.0;
    return max(-r*mu + sqrt(disc), 0.0);
}
// Distance to ground sphere (ATM_BOTTOM), or -1 if no hit ahead.
float atm_distToGround(highp float r, float mu) {
    highp float disc = r*r*(mu*mu - 1.0) + ATM_BOTTOM*ATM_BOTTOM;
    if (disc < 0.0) return -1.0;
    highp float d = -r*mu - sqrt(disc);
    return d >= 0.0 ? d : -1.0;
}

// Analytic densities at radius r (altitude above ATM_BOTTOM).
// W7: r ~6360 (km-scale planet radius) and the (r - ATM_BOTTOM) altitude cancellation are highp -- at
// mediump fp16 the ~6360 radius resolves to ~4 km steps and the density profile (scale-height tens of
// km) would band/collapse. The exp() RESULT narrows to the mediump default.
float atm_rayleighDensity(highp float r) { return exp(-(r - ATM_BOTTOM)/ATM_RAYLEIGH_H); }
float atm_mieDensity(highp float r)      { return exp(-(r - ATM_BOTTOM)/ATM_MIE_H); }

// Optical depth (Rayleigh,Mie line integral) from point p0 along dir for distance d.
// Cheap fixed-step trapezoid; cheap enough for a fullscreen pass.
void atm_opticalDepth(highp vec3 p0, vec3 dir, float d, out float odR, out float odM) {   // W7: km-scale p0 -> highp
    const int N = 8;
    float dt = d / float(N);
    odR = 0.0; odM = 0.0;
    for (int i = 0; i < N; i++) {
        highp vec3 p = p0 + dir * (dt * (float(i) + 0.5));   // W7: km-scale march point
        highp float r = length(p);
        odR += atm_rayleighDensity(r) * dt;
        odM += atm_mieDensity(r) * dt;
    }
}

// Transmittance over a segment of length d from p0 along dir.
vec3 atm_transmittanceSeg(highp vec3 p0, vec3 dir, float d) {   // W7: km-scale p0 -> highp
    float odR, odM;
    atm_opticalDepth(p0, dir, d, odR, odM);
    return exp(-(ATM_RAYLEIGH * odR + ATM_MIE_EXT * odM));
}

// Transmittance from point p to the top of the atmosphere along sun direction.
vec3 atm_transmittanceToSun(highp vec3 p, vec3 sun) {   // W7: km-scale p -> highp
    highp float r = length(p);
    float mu = dot(p, sun) / r;
    float d = atm_distToTop(r, mu);
    if (d < 0.0) return vec3(1.0);
    // SOFT sea-level horizon (user 2026-06-11 'shading indicates depth not slope' -- THE math bug):
    // the old branch returned a BINARY vec3(0) whenever the sun ray intersected the ATM_BOTTOM
    // sphere. ATM_BOTTOM is SEA LEVEL, so at low sun the direct light became a hard step function
    // of ELEVATION (every point below a sun-dependent altitude got zero sun, every point above got
    // full sun) -- slope-blind, elevation-keyed dark patches that read as 'depth shading' + dark
    // rocky blobs. Replace with a smooth attenuation over ~2 deg of sun dip below the sea horizon:
    // night stays dark, the elevation step is gone, slopes drive the shading again.
    float muHoriz = -sqrt(max(0.0, 1.0 - (ATM_BOTTOM * ATM_BOTTOM) / (r * r)));
    float soft = smoothstep(muHoriz - 0.035, muHoriz + 0.005, mu);
    if (soft <= 0.0) return vec3(0.0);
    return atm_transmittanceSeg(p, sun, d) * soft;
}

// Sky in-scatter radiance + view transmittance along a view ray from cameraIn.
// camera/view in atmosphere-space km. Single-scatter Rayleigh+Mie integration.
vec3 atm_skyRadiance(highp vec3 cameraIn, vec3 viewRay, vec3 sun, out vec3 transmittance) {   // W7: km-scale camera -> highp
    highp vec3 camera = cameraIn;
    highp float r = length(camera);
    float mu = dot(camera, viewRay) / r;

    // March-start: if outside the atmosphere, advance to the top shell.
    if (r > ATM_TOP) {
        float dt = atm_distToTop(r, mu);
        if (dt < 0.0) { transmittance = vec3(1.0); return vec3(0.0); } // misses atmosphere
        camera = camera + viewRay * dt;
        r = length(camera);
        mu = dot(camera, viewRay) / r;
    }

    // March-end: ground hit, else exit at top shell.
    float dGround = atm_distToGround(r, mu);
    float dTop = atm_distToTop(r, mu);
    bool ground = dGround > 0.0;
    float dEnd = ground ? dGround : max(dTop, 0.0);
    if (dEnd <= 0.0) { transmittance = vec3(1.0); return vec3(0.0); }

    const int N = 16;
    float dt = dEnd / float(N);
    vec3 inscatR = vec3(0.0);
    vec3 inscatM = vec3(0.0);
    float odR = 0.0, odM = 0.0; // accumulated optical depth from camera to sample

    for (int i = 0; i < N; i++) {
        highp vec3 p = camera + viewRay * (dt * (float(i) + 0.5));   // W7: km-scale march point -> highp
        highp float pr = length(p);
        float dR = atm_rayleighDensity(pr) * dt;
        float dM = atm_mieDensity(pr) * dt;
        odR += dR; odM += dM;
        // transmittance camera->sample
        vec3 tView = exp(-(ATM_RAYLEIGH * odR + ATM_MIE_EXT * odM));
        // transmittance sample->sun
        vec3 tSun = atm_transmittanceToSun(p, sun);
        vec3 t = tView * tSun;
        inscatR += t * dR;
        inscatM += t * dM;
    }
    float nu = dot(viewRay, sun);
    transmittance = exp(-(ATM_RAYLEIGH * odR + ATM_MIE_EXT * odM));
    if (ground) transmittance = vec3(0.0);

    vec3 radiance = ATM_SOLAR_IRRADIANCE * (
        inscatR * ATM_RAYLEIGH * atm_rayleighPhase(nu) +
        inscatM * ATM_MIE      * atm_miePhase(ATM_MIE_G, nu));
    return radiance;
}

// Sun + sky irradiance reaching a surface point with given normal.
// Returns DIRECT sun irradiance (already * N.L); sky_irradiance is the ambient term.
// DIFFUSE WRAP (user 2026-06-10 'looking down the horizon at grass it gets darker'): pure Lambert
// kills slopes tilted a few degrees off the sun, so a downward view (which preferentially shows
// ripple backsides) reads darker than the grazing view of the same field. Wrap lifts away-facing
// gentle slopes (N.L -> (N.L+w)/(1+w)) -- soft-knee terrain diffuse, view-angle gradient gone.
// Unset by programs that do not bind it (GL zero-default) -> plain Lambert there (sky pass).
uniform float uDiffWrap;   // 0 = pure Lambert; ~0.35 = soft wrap (terrain render program)
vec3 atm_sunSkyIrradiance(highp vec3 point, vec3 normal, vec3 sun, out vec3 sky_irradiance) {   // W7: km-scale point -> highp
    highp float r = length(point);
    vec3 up = point / r;
    float muS = dot(up, sun);
    // ELEVATION-INDEPENDENT direct sun (user 2026-06-11 'normals appear elevation based again' --
    // the recurring class, previously narrowed by 1263c51 but not killed): BOTH the sea-horizon cut
    // (muHoriz is a function of r) and the sun-path optical depth key on the POINT'S RADIUS, so at
    // low sun the direct light is a smooth function of terrain ELEVATION -- slope-blind brightness
    // gradients across whole regions that read as 'height-based normals'. Evaluate the sun
    // transmittance on a FIXED shell (sea level + 0.5 km): day/night and sunset reddening still
    // follow the sun's local altitude (mu via `up`), but terrain elevation can no longer modulate
    // direct light -- slope (N.sun) owns the shading. The sky/AP marches keep their per-point
    // physics (airborne sample points genuinely differ); only the SURFACE direct term is pinned.
    vec3 tSun = atm_transmittanceToSun(up * (ATM_BOTTOM + 0.5), sun);
    vec3 direct = ATM_SOLAR_IRRADIANCE * tSun * clamp((dot(normal, sun) + uDiffWrap) / (1.0 + uDiffWrap), 0.0, 1.0);
    // Sky (ambient) dome: Rayleigh-tinted, scaled by how much sky the surface sees
    // and by daylight (smoothstep over the terminator). Hemispheric weight (1+N.up)/2.
    float day = smoothstep(-0.10, 0.25, muS);
    // Desaturate the sky-ambient tint toward white: raw Rayleigh (b~4x r) washed lit land
    // fully blue. mostly-white ambient lights terrain without recoloring it blue.
    vec3 rayTint = ATM_RAYLEIGH / (ATM_RAYLEIGH.x);
    vec3 skyTint = mix(vec3(1.0), rayTint, 0.4);
    // SUN-vs-SKY BALANCE (user 2026-06-02: 'normals arent affecting the lit view properly'). The
    // sky ambient is weighted by N.up (near-directionless), so a too-large ambient washes out the
    // directional N.sun sun term and opposing slopes barely differ -> normals look like they dont
    // shade. Cut the ambient coefficient (0.14 -> 0.075) so the DIRECT sun (which carries the normal
    // via N.sun) dominates and slopes self-shade; the (1+N.up)/2 dome weight stays so it still fills
    // shadowed faces without flattening them. An ambient floor downstream keeps shadows non-black.
    sky_irradiance = ATM_SOLAR_IRRADIANCE * 0.075 * day * skyTint
                     * (0.5 * (1.0 + dot(normal, up)));
    return direct;
}
