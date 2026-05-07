
    precision highp float;

    varying vec3 vPosition;
    varying vec3 vWorld;
    varying float vHeight;
    varying float vBiomeIdx;
    varying float vSnowline;
    varying float vRockSlope;
    varying vec3 vNormal;

    uniform float uHeight;
    uniform int uDebugMode;
    uniform float uTime;
    uniform float uTimeOfDay;
    uniform float uSeason;
    uniform vec3 uSampleCameraPos;
    uniform sampler2D uTGeom;
    uniform sampler2D uTMat;
    uniform float uMaxR;
    uniform float uConcentration;

    vec2 worldToBakeUV(vec2 worldXZ) {
        vec2 dxz = worldXZ - uSampleCameraPos.xz;
        float r = length(dxz);
        if (r < 1e-6) return vec2(0.5);
        float rNorm = clamp(r / uMaxR, 0.0, 1.0);
        float t = pow(rNorm, 1.0 / max(uConcentration, 1e-3));
        vec2 dir = dxz / r;
        return clamp(0.5 + 0.5 * t * dir, vec2(0.0), vec2(1.0));
    }

    vec3 hsv2rgb(vec3 c) {
        vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
        vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
        return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
    }
    vec3 viridis(float t) {
        t = clamp(t, 0.0, 1.0);
        vec3 c0 = vec3(0.267, 0.005, 0.329);
        vec3 c1 = vec3(0.127, 0.566, 0.551);
        vec3 c2 = vec3(0.993, 0.906, 0.144);
        return mix(mix(c0, c1, smoothstep(0.0, 0.5, t)), c2, smoothstep(0.5, 1.0, t));
    }

    // Cheap value-noise hash for material variation (not needing lockstep).
    float hash21(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
    }
    // Match JS-side _hash21 used by rocks/trees placement (rocks.js _vnoise2).
    // Used ONLY for biotic AO so the dark patches sit on the same lattice as spawns.
    float hash21Spawn(vec2 p) {
        float h = sin(p.x * 127.1 + p.y * 311.7) * 43758.5453;
        return fract(h);
    }
    float vnoiseSpawn(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        float a = hash21Spawn(i);
        float b = hash21Spawn(i + vec2(1.0, 0.0));
        float c = hash21Spawn(i + vec2(0.0, 1.0));
        float d = hash21Spawn(i + vec2(1.0, 1.0));
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    }
    float vnoise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        float a = hash21(i);
        float b = hash21(i + vec2(1.0, 0.0));
        float c = hash21(i + vec2(0.0, 1.0));
        float d = hash21(i + vec2(1.0, 1.0));
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    }
    // Per-octave rotation breaks axis-aligned grid banding when sampling vnoise.
    mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

    const float PI = 3.14159265359;

    // Sun direction from time-of-day. t=0 midnight, 0.25 sunrise, 0.5 noon, 0.75 sunset.
    vec3 sunDirFromTOD(float t) {
        float el = sin((t - 0.25) * 2.0 * PI); // -1 midnight, +1 noon
        float az = t * 2.0 * PI;
        float horiz = sqrt(max(1.0 - el * el, 0.0));
        return normalize(vec3(cos(az) * horiz, el, sin(az) * horiz));
    }
    vec3 moonDirFromTOD(float t) { return -sunDirFromTOD(t); }
    // Sun light color: warm at horizon, white at noon, none below horizon.
    vec3 sunColorFromTOD(float t) {
        vec3 sd = sunDirFromTOD(t);
        float above = clamp(sd.y, 0.0, 1.0);
        vec3 warm = vec3(1.30, 0.62, 0.28);
        vec3 mid  = vec3(1.20, 0.95, 0.78);
        vec3 white= vec3(1.10, 1.00, 0.90);
        vec3 c = mix(warm, mid, smoothstep(0.0, 0.18, sd.y));
        c = mix(c, white, smoothstep(0.18, 0.55, sd.y));
        float gate = smoothstep(-0.05, 0.08, sd.y);
        return c * gate;
    }
    // Sky color along a view direction. Blends day blue, twilight orange, night dark.
    vec3 skyColorDir(vec3 d) {
        vec3 sd = sunDirFromTOD(uTimeOfDay);
        float sunEl = sd.y;
        float dayAmt   = smoothstep(-0.05, 0.30, sunEl);
        float twiAmt   = (1.0 - dayAmt) * smoothstep(-0.30, 0.05, sunEl);
        float nightAmt = 1.0 - smoothstep(-0.15, 0.05, sunEl);
        float t = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
        // Day palette
        vec3 dHor = vec3(0.78, 0.84, 0.92);
        vec3 dMid = vec3(0.55, 0.72, 0.90);
        vec3 dZen = vec3(0.20, 0.42, 0.74);
        vec3 day = mix(mix(dHor, dMid, smoothstep(0.0, 0.45, t)), dZen, smoothstep(0.45, 1.0, t));
        // Twilight: orange/pink near horizon strongest opposite/toward sun azimuth
        float sunAlign = max(dot(normalize(vec3(d.x, 0.0, d.z)), normalize(vec3(sd.x, 0.0, sd.z))), 0.0);
        vec3 tHor = mix(vec3(0.55, 0.32, 0.40), vec3(1.05, 0.55, 0.30), sunAlign);
        vec3 tMid = vec3(0.40, 0.32, 0.46);
        vec3 tZen = vec3(0.10, 0.14, 0.32);
        vec3 twi = mix(mix(tHor, tMid, smoothstep(0.0, 0.45, t)), tZen, smoothstep(0.45, 1.0, t));
        // Night: dark blue zenith, slightly lighter horizon, with faint star-tint speckle
        vec3 nHor = vec3(0.04, 0.06, 0.12);
        vec3 nMid = vec3(0.02, 0.03, 0.08);
        vec3 nZen = vec3(0.005, 0.01, 0.04);
        vec3 night = mix(mix(nHor, nMid, smoothstep(0.0, 0.45, t)), nZen, smoothstep(0.45, 1.0, t));
        // Hashed jittered starfield with magnitude+color variation
        vec2 starUV = d.xz / max(0.001, abs(d.y) + 0.15) * 4.0;
        vec3 starCol = vec3(0.0);
        for (int si = 0; si < 3; si++) {
            float scl = exp2(float(si));
            vec2 cell = floor(starUV * scl);
            vec2 frac = fract(starUV * scl) - 0.5;
            float h1 = fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
            float h2 = fract(sin(dot(cell, vec2(269.5, 183.3))) * 43758.5453);
            float h3 = fract(sin(dot(cell, vec2(419.2, 371.9))) * 43758.5453);
            float h4 = fract(sin(dot(cell, vec2(57.13, 219.7))) * 43758.5453);
            vec2 starPos = (vec2(h1, h2) - 0.5) * 0.6;
            float magnitude = pow(h3, 8.0);
            float dd = length(frac - starPos);
            float bright = magnitude * smoothstep(0.04, 0.0, dd) / scl;
            vec3 tint = mix(vec3(0.85, 0.92, 1.10), vec3(1.10, 0.95, 0.80), h4);
            starCol += bright * tint;
        }
        starCol *= smoothstep(-0.05, 0.10, d.y) * nightAmt;
        night += starCol * 1.4;
        return day * dayAmt + twi * twiAmt + night * nightAmt;
    }

    // Pure-diffuse lighting with subsurface wrap. No specular, no fresnel.
    vec3 pbrLight(vec3 N, vec3 V, vec3 L, vec3 albedo, vec3 lightCol, float sssAmt) {
        float NdotL = max(dot(N, L), 0.0);
        float wrap = max(0.0, dot(N, L) + sssAmt * 0.3) / (1.0 + sssAmt * 0.3);
        float diffTerm = mix(NdotL, wrap, sssAmt);
        return albedo * diffTerm * lightCol / PI;
    }

    void main() {
        // Sample baked textures: tGeom (R=packed h, GB=normal xz), tMat (rock/snow/grass/forest weights)
        vec2 texUV = worldToBakeUV(vWorld.xz);
        vec4 geom = texture2D(uTGeom, texUV);
        vec4 mat  = texture2D(uTMat, texUV, -0.5);
        float bakedNx = geom.g * 2.0 - 1.0;
        float bakedNz = geom.b * 2.0 - 1.0;
        float bakedNy = sqrt(max(1.0 - bakedNx*bakedNx - bakedNz*bakedNz, 1e-4));
        vec3 N = normalize(vec3(bakedNx, bakedNy, bakedNz));
        if (N.y < 0.0) N = -N;
        float bakedRockMix   = mat.r;
        float bakedSnowMix   = mat.g;
        float bakedGrassMix  = mat.b;
        float bakedForestMix = mat.a;
        // Keep dpx/dpy for edge-length debug only.
        vec3 dpx = dFdx(vPosition);
        vec3 dpy = dFdy(vPosition);
        float slope = 1.0 - N.y;
        float invNy = 1.0 / max(N.y, 1e-4);
        vec2 gradLocal = vec2(-N.x * invNy, -N.z * invNy);

        vec3 color;
        if (uDebugMode == 2) {
            color = viridis(clamp(slope * 2.5, 0.0, 1.0));
        } else if (uDebugMode == 3) {
            float edgeLen = 0.5 * (length(dpx) + length(dpy));
            float dist = length(vPosition - cameraPosition);
            float norm = edgeLen / max(dist * 0.01, 0.001);
            color = viridis(clamp(norm * 0.5, 0.0, 1.0));
        } else if (uDebugMode == 4) {
            float ang = atan(gradLocal.y, gradLocal.x);
            float hue = (ang / 6.2831853) + 0.5;
            float mag = clamp(length(gradLocal) * 1.5, 0.0, 1.0);
            color = hsv2rgb(vec3(hue, 0.9, 0.3 + 0.7 * mag));
        } else if (uDebugMode == 5) {
            color = viridis(clamp(length(gradLocal) * 1.2, 0.0, 1.0));
        } else {
            vec2 wp = vPosition.xz;
            // Real slope from vNormal (sin of slope angle): 0=flat, 1=vertical.
            // Tied to the visible terrain — material thresholds follow actual mountainsides.
            float realSlope = sqrt(max(1.0 - N.y * N.y, 0.0));
            float realSlopeForMat = realSlope * 0.65;
            float n_xl = vnoise(wp * 0.0035);
            float n_lg = vnoise(wp * 0.018);
            float n_md = vnoise(wp * 0.075);
            float n_sm = vnoise(wp * 0.42);
            float n_micro = vnoise(wp * 1.6);
            float n_var = vnoise(wp * 8.0); // small-scale albedo variance
            float dither = (n_sm - 0.5) * 0.05 + (n_micro - 0.5) * 0.02 + (n_var - 0.5) * 0.07;

            // Season weights
            float wSpring = max(1.0 - abs(uSeason - 0.0)   * 4.0, 0.0);
            float wSummer = max(1.0 - abs(uSeason - 0.25)  * 4.0, 0.0);
            float wAutumn = max(1.0 - abs(uSeason - 0.5)   * 4.0, 0.0);
            float wWinter = max(1.0 - abs(uSeason - 0.75)  * 4.0, 0.0);
            float wSum = wSpring + wSummer + wAutumn + wWinter + 1e-5;
            wSpring /= wSum; wSummer /= wSum; wAutumn /= wSum; wWinter /= wSum;
            // Realistic muted palettes
            vec3 sand    = mix(vec3(0.86, 0.76, 0.56), vec3(0.94, 0.86, 0.66), n_md);
            sand        *= 0.92 + 0.16 * n_lg;
            sand        += vec3(0.03, 0.01, -0.01) * (n_xl - 0.5);
            // grass: per-season tint
            vec3 grassA  = vec3(0.22, 0.42, 0.14);
            vec3 grassB  = vec3(0.34, 0.58, 0.20);
            vec3 grassC  = vec3(0.30, 0.46, 0.16);
            vec3 grass   = mix(grassA, grassB, n_md);
            grass        = mix(grass, grassC, smoothstep(0.55, 0.85, n_lg) * 0.45);
            grass       *= 0.82 + 0.28 * n_xl;
            vec3 grassSpring = grass * vec3(0.85, 1.15, 0.80);
            vec3 grassSummer = grass;
            vec3 grassAutumn = grass * vec3(1.35, 0.95, 0.55);
            vec3 grassWinter = mix(grass, vec3(0.78, 0.76, 0.70), 0.75);
            grass = grassSpring*wSpring + grassSummer*wSummer + grassAutumn*wAutumn + grassWinter*wWinter;
            // forest: per-season tint
            vec3 forestA = vec3(0.08, 0.26, 0.10);
            vec3 forestB = vec3(0.16, 0.40, 0.16);
            vec3 forestC = vec3(0.20, 0.32, 0.12);
            vec3 forest  = mix(forestA, forestB, n_md);
            forest       = mix(forest, forestC, smoothstep(0.6, 0.9, n_lg) * 0.5);
            forest      *= 0.85 + 0.25 * n_xl;
            vec3 forestSpring = forest * vec3(0.90, 1.15, 0.85);
            vec3 forestSummer = forest;
            vec3 forestAutumn = forest * vec3(1.55, 1.00, 0.50);
            vec3 forestWinter = mix(forest, vec3(0.40, 0.42, 0.40), 0.55);
            forest = forestSpring*wSpring + forestSummer*wSummer + forestAutumn*wAutumn + forestWinter*wWinter;
            // rock: gray with iron-oxide warm streaks
            vec3 rockBase= mix(vec3(0.44, 0.40, 0.36), vec3(0.62, 0.57, 0.50), n_md);
            vec3 rockWarm= vec3(0.56, 0.40, 0.28);
            float oxide  = smoothstep(0.55, 0.85, n_lg);
            vec3 rock    = mix(rockBase, rockWarm, oxide * 0.35);
            rock        *= 0.72 + 0.36 * n_xl;
            vec3 darkRock= mix(vec3(0.20, 0.19, 0.19), vec3(0.32, 0.30, 0.28), n_md);
            // snow: bluish in shadow, cream in sun
            vec3 snowSun = vec3(1.00, 0.99, 0.95);
            vec3 snowSh  = vec3(0.84, 0.88, 0.96);
            vec3 snow    = mix(snowSh, snowSun, smoothstep(0.4, 0.9, N.y));
            snow         = mix(snow, snow * 1.02, n_sm * 0.4);

            // Use BAKED material weights (computed once per 10m snap) instead of per-pixel thresholds
            float grassBand = bakedGrassMix;
            float forestMix = bakedForestMix;
            float rockMix   = bakedRockMix;
            float snowMix   = bakedSnowMix;
            float cliff     = smoothstep(0.55, 0.78, realSlopeForMat);

            vec3 albedo = sand;
            albedo = mix(albedo, grass, grassBand);
            albedo = mix(albedo, forest, forestMix * 0.55);
            albedo = mix(albedo, rock, rockMix);
            albedo = mix(albedo, snow, snowMix);
            albedo = mix(albedo, darkRock, cliff);

            // Concave mask reused for snow tint — derived from real slope.
            float concav = 1.0 - smoothstep(0.0, 0.6, realSlopeForMat);
            // Snow: subtle blue tint in concave depressions
            albedo = mix(albedo, albedo * vec3(0.82, 0.88, 1.05), snowMix * concav * 0.55);
            float sandWeightEarly = (1.0 - grassBand) * (1.0 - forestMix) * (1.0 - rockMix) * (1.0 - snowMix) * (1.0 - cliff);
            float sandRipple = vnoise(wp * 0.05 + vec2(n_lg, n_md) * 1.5);
            albedo *= 1.0 + (sandRipple - 0.5) * 0.06 * sandWeightEarly;

            // Wetness near water: surfaces within ~3m of water level (y=0) get darker
            float wetness = (1.0 - smoothstep(0.0, 3.5, vHeight)) * (1.0 - cliff);
            albedo *= mix(1.0, 0.65, wetness);

            // Biotic AO: dark cool-green shadow tone keyed to the same density
            // field rocks/trees cluster on. Use vnoiseSpawn — same lattice as
            // JS _vnoise2 in rocks.js / scene.js _densityNoise — so AO patches
            // coincide with where placements actually land.
            float densityField = vnoiseSpawn(wp * 0.05);
            // Boost vegDensity floor: even when biome material weights are low
            // (e.g. trees parked on plains where forestMix is small), spawns are
            // still real — let the density-noise carry the AO.
            float vegDensity = clamp(forestMix * 1.5 + rockMix * 0.8 + grassBand * 0.4 + 0.35, 0.0, 1.0);
            float vegMask = (1.0 - cliff) * (1.0 - snowMix);
            float biotic = vegDensity * vegMask * densityField * 0.92;
            // Stronger AO darkening — both as albedo tint AND as a multiplicative
            // shadow factor so it survives the filmic shoulder downstream.
            albedo = mix(albedo, albedo * vec3(0.40, 0.45, 0.32), clamp(biotic * 0.95, 0.0, 1.0));

            // Per-biome high-frequency DETAIL TEXTURE — visible up close, faded by distance
            // Distance fade: full amplitude under 40m, zero past 120m (kills aliasing).
            float distToCam = length(vWorld - cameraPosition);
            float detailFade = 1.0 - smoothstep(40.0, 120.0, distToCam);
            // Triplanar projection — sample detail noise on XZ, YZ, XY planes
            // weighted by abs(N). Without this, steep slopes stretch the XZ
            // projection into long streaks along the slope-fall direction.
            // Wrap each plane into a 1024m tile so the hash keeps float precision.
            vec3 tpW = mod(vWorld, vec3(1024.0));
            vec2 wpW_xz = tpW.xz;
            vec2 wpW_yz = tpW.yz;
            vec2 wpW_xy = tpW.xy;
            vec2 wpD_xz = wpW_xz + vec2(vnoise(wpW_xz * 0.3) - 0.5, vnoise(wpW_xz * 0.3 + vec2(7.1, 13.2)) - 0.5) * 0.6;
            vec2 wpD_yz = wpW_yz + vec2(vnoise(wpW_yz * 0.3) - 0.5, vnoise(wpW_yz * 0.3 + vec2(7.1, 13.2)) - 0.5) * 0.6;
            vec2 wpD_xy = wpW_xy + vec2(vnoise(wpW_xy * 0.3) - 0.5, vnoise(wpW_xy * 0.3 + vec2(7.1, 13.2)) - 0.5) * 0.6;
            // Triplanar weights from world-space normal. Sharpen with pow so flat
            // terrain (|N.y|~1) keeps the existing XZ look exactly.
            vec3 tpN = pow(abs(normalize(vNormal)), vec3(4.0));
            tpN /= max(tpN.x + tpN.y + tpN.z, 1e-4);
            // Convenience: sample three planes on a frequency-scaled vnoise call.
            #define TP_VN(F) (vnoise((F) * wpD_yz) * tpN.x + vnoise((F) * wpD_xz) * tpN.y + vnoise((F) * wpD_xy) * tpN.z)
            #define TP_VNR(R, F) (vnoise((R) * (F) * wpD_yz) * tpN.x + vnoise((R) * (F) * wpD_xz) * tpN.y + vnoise((R) * (F) * wpD_xy) * tpN.z)
            // Backward-compat aliases so the existing per-biome blocks below still compile.
            vec2 wpW = wpW_xz;
            vec2 wpD = wpD_xz;
            // Grass: speckle + fine strands + grain + occasional bright clumps (per-octave rotation)
            float ga = vnoise(rot2(0.3) * wpD * 8.0);
            float gb = vnoise(rot2(0.7) * wpD * 25.0);
            float gc = vnoise(rot2(1.4) * wpD * 60.0);
            float gClump = step(0.7, vnoise(rot2(2.1) * wpD * 4.0));
            float grassDetail = (ga - 0.5) * 0.2 + (gb - 0.5) * 0.5 + (gc - 0.5) * 0.4 + gClump * 0.05;
            // Rock: triplanar fracture + blocks + grain — kills slope streaks.
            float rFract = TP_VNR(rot2(0.5), 3.0);
            float rBlocks = step(0.5, TP_VNR(rot2(1.1), 1.5)) * 0.10;
            float rGrain = TP_VNR(rot2(1.9), 40.0);
            float rockDetail = (rFract - 0.5) * 0.25 + rBlocks - 0.05 + (rGrain - 0.5) * 0.4;
            // Sand: meandering ripples — direction wobbles via low-freq noise so iso-lines curve.
            vec2 sandDir = normalize(vec2(1.0, 0.7));
            float sandPerturb = (vnoise(wpW * 0.04) - 0.5) * 1.2;
            vec2 sandWp = rot2(sandPerturb) * wpD;
            // Cross two rotated sin's for non-straight ripples.
            float sRipplesA = sin(dot(sandWp, sandDir) * 3.0 + vnoise(wpD * 0.5) * 1.2);
            float sRipplesB = sin(dot(rot2(0.6) * sandWp, sandDir) * 2.3);
            float sRipples = (sRipplesA * 0.6 + sRipplesB * 0.4) * 0.5 + 0.5;
            float sGrain = vnoise(rot2(2.4) * wpD * 50.0);
            float sandDetail = (sRipples - 0.5) * 0.4 + (sGrain - 0.5) * 0.25;
            // Snow: smooth coarse + fine grain (per-octave rotation)
            float swCoarse = vnoise(rot2(0.9) * wpD * 6.0);
            float swFine = vnoise(rot2(1.7) * wpD * 20.0);
            float snowDetail = (swCoarse - 0.5) * 0.10 + (swFine - 0.5) * 0.2;
            // Sand weight = leftover (also matches earlier sandWeightEarly)
            float sandWeight = (1.0 - grassBand) * (1.0 - forestMix) * (1.0 - rockMix) * (1.0 - snowMix) * (1.0 - cliff);
            float dGrass = grassBand * grassDetail;
            float dRock = max(rockMix, cliff) * rockDetail;
            float dSand = sandWeight * sandDetail;
            float dSnow = snowMix * snowDetail;
            float detail = (dGrass + dRock + dSand + dSnow) * detailFade;
            albedo *= (1.0 + detail * 0.35);

            // Slope-direction tint (stronger — gradient direction from baked normal G,B)
            vec2 gradXZ = vec2(bakedNx, bakedNz);
            float gradLen = length(gradXZ);
            if (gradLen > 1e-4) {
                float gradAng = atan(gradXZ.y, gradXZ.x);
                float warmth = sin(gradAng);
                vec3 dirTint = vec3(1.0 + 0.16 * warmth, 1.0, 1.0 - 0.16 * warmth);
                albedo *= dirTint;
            }

            albedo += dither;

            // Micro-detail normal perturbation: 2 octaves of high-freq noise gradients
            float md_eps = 0.04;
            mat2 mdR1 = rot2(1.1), mdR2 = rot2(2.3);
            vec2 mwp1 = mdR1 * wpD * 5.0;
            vec2 mwp2 = mdR2 * wpD * 15.0;
            // Triplanar micro-bump gradients — sample on YZ/XZ/XY weighted by abs(N).
            mat2 mdR1m = mdR1, mdR2m = mdR2;
            #define MD_TPX(R, P, F) ( (vnoise((R)*(F)*wpD_yz + vec2(md_eps,0.0)) - vnoise((R)*(F)*wpD_yz - vec2(md_eps,0.0))) * tpN.x \
                                    + (vnoise((R)*(F)*wpD_xz + vec2(md_eps,0.0)) - vnoise((R)*(F)*wpD_xz - vec2(md_eps,0.0))) * tpN.y \
                                    + (vnoise((R)*(F)*wpD_xy + vec2(md_eps,0.0)) - vnoise((R)*(F)*wpD_xy - vec2(md_eps,0.0))) * tpN.z )
            #define MD_TPZ(R, P, F) ( (vnoise((R)*(F)*wpD_yz + vec2(0.0,md_eps)) - vnoise((R)*(F)*wpD_yz - vec2(0.0,md_eps))) * tpN.x \
                                    + (vnoise((R)*(F)*wpD_xz + vec2(0.0,md_eps)) - vnoise((R)*(F)*wpD_xz - vec2(0.0,md_eps))) * tpN.y \
                                    + (vnoise((R)*(F)*wpD_xy + vec2(0.0,md_eps)) - vnoise((R)*(F)*wpD_xy - vec2(0.0,md_eps))) * tpN.z )
            float md1_x = MD_TPX(mdR1m, 0, 5.0);
            float md1_z = MD_TPZ(mdR1m, 0, 5.0);
            float md2_x = MD_TPX(mdR2m, 0, 15.0);
            float md2_z = MD_TPZ(mdR2m, 0, 15.0);
            // Per-biome strength: subtle micro-bumps only — slope normal dominates lighting.
            float micStr = mix(0.08, 0.20, rockMix);           // rock
            micStr = mix(micStr, 0.10, grassBand);              // grass
            micStr = mix(micStr, 0.05, snowMix);                // snow
            micStr = mix(micStr, 0.25, cliff);                  // cliff
            vec3 microN = normalize(vec3(-(md1_x + md2_x*0.5)*micStr, 1.0, -(md1_z + md2_z*0.5)*micStr));
            // Tangent-space-ish blend: rotate N toward microN — small perturbation on smooth vNormal
            N = normalize(N + microN * 0.10 - vec3(0.0, 0.10, 0.0));

            // curvature AO via real slope — concave dips darken slightly
            float ao = 1.0 - 0.12 * (1.0 - smoothstep(0.0, 0.6, realSlopeForMat));
            ao *= 0.85 + 0.15 * n_lg;
            // Multiply biotic AO into lighting AO so dense-spawn lattice patches
            // darken even after filmic shoulder. Range: biotic ∈ [0,0.92] → ao
            // factor ∈ [0.45, 1.0].
            ao *= 1.0 - clamp(biotic * 0.6, 0.0, 0.55);

            // Subsurface amount — strong on snow, medium on grass/forest, none on rock
            float sssAmt = snowMix * 0.7 + grassBand * 0.4 + forestMix * 0.3 * (1.0 - rockMix);
            sssAmt = clamp(sssAmt, 0.0, 1.0);

            // Soften slope shading: lerp the shading normal 15% toward up — slopes retain ~50% more
            // contrast than the previous 30% flatten while still smoothing out micro-bake harshness.
            vec3 Nshade = normalize(mix(N, vec3(0.0, 1.0, 0.0), 0.0));

            // Three-light studio rig (key sun + sky hemisphere fill + warm ground bounce)
            vec3 sunDir = sunDirFromTOD(uTimeOfDay);
            vec3 sunCol = sunColorFromTOD(uTimeOfDay);
            vec3 V = normalize(cameraPosition - vPosition);
            // KEY: sun (pure diffuse) — warm photo-film tint on lit side
            vec3 keyTint = vec3(1.0, 0.99, 0.97);
            vec3 lit = pbrLight(Nshade, V, sunDir, albedo, sunCol * keyTint, sssAmt) * 0.30 * 3.0;
            // Subsurface tint pass for snow/grass — adds slight albedo-tinted forward-scatter on key
            float backLight = max(0.0, dot(-Nshade, sunDir));
            vec3 sssTint = albedo * sunCol * backLight * sssAmt * 0.18;
            lit += sssTint;

            // Moon key at night
            vec3 moonDir = moonDirFromTOD(uTimeOfDay);
            float nightAmt = 1.0 - smoothstep(-0.05, 0.18, sunDir.y);
            vec3 moonCol = vec3(0.18, 0.22, 0.32) * nightAmt * max(moonDir.y, 0.0);
            lit += pbrLight(Nshade, V, normalize(moonDir + vec3(0.001)), albedo, moonCol, sssAmt);

            // FILL: sky hemisphere — cool blue from above, biased toward shadow side
            vec3 skyOverhead = skyColorDir(vec3(0.0, 1.0, 0.0));
            float dayAmb   = smoothstep(-0.05, 0.30, sunDir.y);
            float twiAmb   = (1.0 - dayAmb) * smoothstep(-0.30, 0.05, sunDir.y);
            float ambStrength = mix(0.06, 0.32, dayAmb) + 0.18 * twiAmb;
            vec3 ambSky = mix(skyColorDir(Nshade), vec3(1.0), 0.3);
            vec3 fillSkyOverhead = mix(skyOverhead, vec3(1.0), 0.3);
            vec3 fillTint = vec3(0.96, 0.98, 1.02);
            vec3 fill = albedo * mix(ambSky, fillSkyOverhead, 0.4) * fillTint * (0.4 + 0.5 * Nshade.y) * ambStrength * 1.3;
            // BOUNCE: warm ground bounce from below, picks up albedo
            vec3 groundBounceCol = albedo * vec3(0.85, 0.72, 0.55) * (0.25 + 0.75 * dayAmb);
            vec3 bounce = groundBounceCol * max(0.0, -dot(Nshade, vec3(0.0, 1.0, 0.0))) * 0.25
                        + groundBounceCol * (1.0 - max(Nshade.y, 0.0)) * 0.30 * dayAmb;
            lit += fill + bounce;
            lit *= ao;
            // Ambient floor — guarantee minimum lighting so nothing goes black
            lit = max(lit, albedo * 0.06);

            // Filmic shoulder (gentler) — shoulder kicks in at higher luminance, mid-tones stay dark.
            // Formula: x' = x / (x + 0.45) * 1.45
            lit = lit / (lit + vec3(0.55)) * 1.55;

            color = lit;
            if (uDebugMode == 1) color *= 0.35;
            if (uDebugMode == 99) color = vec3(clamp(biotic, 0.0, 1.0));
        }

        // Time-of-day atmospheric haze
        float dist = length(vPosition - cameraPosition);
        vec3 viewDir = normalize(vPosition - cameraPosition);
        float t = uTimeOfDay;
        float morningMist = smoothstep(0.18, 0.27, t) * (1.0 - smoothstep(0.27, 0.35, t));
        float sunsetHaze  = smoothstep(0.65, 0.75, t) * (1.0 - smoothstep(0.75, 0.85, t));
        float nightDip    = smoothstep(0.85, 1.0, t) + (1.0 - smoothstep(0.0, 0.15, t));
        float baseDensity = 0.0035;
        float density = baseDensity * (1.0 + morningMist * 1.5 + sunsetHaze * 0.6 - 0.3 * nightDip);
        density = max(density, 0.0008);
        float heightFactor = 1.0 - clamp(vWorld.y / (uHeight * 1.5), 0.0, 0.6);
        float hazeAmount = (1.0 - exp(-dist * density * heightFactor)) * 0.95;
        vec3 skyAtView = skyColorDir(viewDir);
        vec3 hazeTint;
        if (sunsetHaze > 0.0 || morningMist > 0.0) {
            vec3 warmHaze = vec3(1.05, 0.92, 0.78);
            hazeTint = mix(vec3(1.0), warmHaze, clamp(sunsetHaze + morningMist, 0.0, 1.0));
        } else if (nightDip > 0.5) {
            hazeTint = vec3(0.55, 0.65, 0.85);
        } else {
            hazeTint = vec3(0.95, 0.98, 1.04);
        }
        vec3 hazeColor = skyAtView * hazeTint;
        gl_FragColor = vec4(mix(color, hazeColor, hazeAmount), 1.0);
    }
