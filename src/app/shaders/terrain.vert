
    #define MAX_ANCHORS 64
    #define MAX_BIOMES 12
    #define POLAR_MAX_R 2200.0

    uniform vec3 uSampleCameraPos;
    uniform float uHeight;
    uniform float uConcentration;
    uniform int uAnchorCount;
    uniform float uFalloff;
    // Anchor: xy = world pos, z = elevation, w = biome index (float)
    uniform vec4 uAnchors[MAX_ANCHORS];
    // Biome packed in 2 vec4:
    //   slot0: amp, freq, elevation, continentMix
    //   slot1: ridgeMix, rollMix, snowline, rockSlope
    uniform vec4 uBiomes0[MAX_BIOMES];
    uniform vec4 uBiomes1[MAX_BIOMES];

    varying vec3 vPosition;
    varying vec3 vWorld;
    varying float vHeight;
    varying float vBiomeIdx;
    varying float vSnowline;
    varying float vRockSlope;
    varying vec3 vNormal;

    // ============ snoiseD: simplex2 with analytical derivatives ============
    vec3 permute3(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
    vec3 snoiseD(vec2 v) {
        const vec4 C = vec4(0.211324865405187, 0.366025403784439, -0.577350269189626, 0.024390243902439);
        vec2 i  = floor(v + dot(v, C.yy));
        vec2 x0 = v - i + dot(i, C.xx);
        vec2 i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
        vec4 x12 = x0.xyxy + C.xxzz;
        x12.xy -= i1;
        i = mod(i, 289.0);
        vec3 p = permute3(permute3(i.y + vec3(0.0, i1.y, 1.0)) + i.x + vec3(0.0, i1.x, 1.0));
        vec3 x = 2.0 * fract(p * C.www) - 1.0;
        vec3 h = abs(x) - 0.5;
        vec3 ox = floor(x + 0.5);
        vec3 a0 = x - ox;
        vec2 g0 = vec2(a0.x, h.x);
        vec2 g1 = vec2(a0.y, h.y);
        vec2 g2 = vec2(a0.z, h.z);
        vec3 norm = 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
        g0 *= norm.x; g1 *= norm.y; g2 *= norm.z;
        vec3 t = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
        vec3 t2 = t*t; vec3 t4 = t2*t2;
        float gd0 = dot(g0, x0), gd1 = dot(g1, x12.xy), gd2 = dot(g2, x12.zw);
        float n = 130.0 * dot(t4, vec3(gd0, gd1, gd2));
        return vec3(n, 0.0, 0.0);
    }
    float snoise(vec2 v) { return snoiseD(v).x; }

    // ============ FBM: simple 3-octave with offset-salt ============
    // Salt is a vec2 added before sampling; per octave we rotate-and-shift the salt
    // so octaves decorrelate. Identical operation order on JS.
    float fbm3(vec2 p, float freq, vec2 salt) {
        float h = 0.0;
        float a = 1.0;
        float f = freq;
        vec2 o = salt;
        // octave 0
        h += a * snoise(p * f + o);
        a *= 0.5; f *= 2.0; o = vec2(o.y, -o.x) + vec2(11.1, 7.3);
        // octave 1
        h += a * snoise(p * f + o);
        a *= 0.5; f *= 2.0; o = vec2(o.y, -o.x) + vec2(11.1, 7.3);
        // octave 2
        h += a * snoise(p * f + o);
        return h;
    }
    float fbm2(vec2 p, float freq, vec2 salt) {
        float h = 0.0;
        float a = 1.0;
        float f = freq;
        vec2 o = salt;
        h += a * snoise(p * f + o);
        a *= 0.5; f *= 2.0; o = vec2(o.y, -o.x) + vec2(11.1, 7.3);
        h += a * snoise(p * f + o);
        return h;
    }
    float ridgeFbm(vec2 p, float freq, vec2 salt) {
        // Softened ridge: pow(1-|n|, 0.7) per octave breaks knife-edge crease.
        float h = 0.0;
        float a = 1.0;
        float f = freq;
        vec2 o = salt;
        h += a * pow(1.0 - abs(snoise(p * f + o)), 0.7);
        a *= 0.5; f *= 2.0; o = vec2(o.y, -o.x) + vec2(11.1, 7.3);
        h += a * pow(1.0 - abs(snoise(p * f + o)), 0.7);
        a *= 0.5; f *= 2.0; o = vec2(o.y, -o.x) + vec2(11.1, 7.3);
        h += a * pow(1.0 - abs(snoise(p * f + o)), 0.7);
        return h - 0.875;
    }

    // ============ Biome erosion strength (rock>grass>dunes) ============
    float biomeErosionStrength(int bi) {
        // 0 snow_peak, 1 alpine_rock, 2 forest_hills, 3 grassland, 4 plains,
        // 5 desert_dunes, 6 river_valley, 7 river_channel, 8 lakeshore
        if (bi == 0) return 1.20;
        if (bi == 1) return 1.30;
        if (bi == 2) return 0.85;
        if (bi == 3) return 0.55;
        if (bi == 4) return 0.45;
        if (bi == 5) return 0.10;
        if (bi == 6) return 0.65;
        if (bi == 7) return 0.50;
        if (bi == 8) return 0.40;
        return 0.7;
    }
    // Shared erosion + roughness layer applied to ALL biomes.
    // Returns delta to add to height contribution (typically negative for channels).
    float applyErosionDelta(vec2 p, int bi) {
        float em = fbm3(p, 0.005, vec2(3.7, 5.1));
        float ed = fbm2(p, 0.02,  vec2(8.3, 1.9));
        float channels = (1.0 - abs(em)) * (1.0 - abs(ed)) * 0.7;
        float rough = fbm2(p, 0.08, vec2(17.7, 23.1)) * 0.10;
        float strength = biomeErosionStrength(bi);
        // Output is in same units as biomeContribution (pre-uHeight scale).
        // Treat 20.0 = nominal terrain amplitude (uHeight default).
        return (-channels * 0.15 + rough * 0.15) * strength * 20.0;
    }
    // ============ Biome height contribution ============
    float biomeContribution(vec2 p, int bi) {
        vec4 b0 = uBiomes0[bi];
        float amp = b0.x;
        float freq = b0.y;
        float elev = b0.z;
        float continentMix = b0.w;
        vec4 b1 = uBiomes1[bi];
        float ridgeMix = b1.x;
        float rollMix = b1.y;
        // Salts seeded from biome index so different biomes use different noise.
        float bf = float(bi);
        vec2 saltC = vec2(13.0 + bf * 7.1, 29.0 - bf * 3.3);
        vec2 saltR = vec2(91.0 - bf * 5.7, 41.0 + bf * 2.9);
        vec2 saltL = vec2(57.0 + bf * 11.0, 17.0 - bf * 6.4);
        float cont  = fbm3(p, freq,        saltC);
        float ridge = ridgeFbm(p, freq * 2.0, saltR);
        float roll  = fbm2(p, freq * 0.5,  saltL);
        // Inner ridge erosion: secondary noise carves the ridge crease.
        float erosion = fbm2(p, freq * 3.0, saltL + vec2(31.7, 19.1)) * 0.3;
        float broadGate = clamp(cont * 0.5 + 0.5, 0.0, 1.0);
        ridge = ridge * (1.0 - erosion) * broadGate;
        float h = elev + amp * (cont * continentMix + ridge * ridgeMix + roll * rollMix);
        h += applyErosionDelta(p, bi);
        return h;
    }

    // ============ Anchor-blended height (LOCKSTEP) ============
    // weight = exp(-d/falloff) * smoothstep(3*falloff, 2*falloff, d)
    // GLSL smoothstep(edge0, edge1, x) with edge0 > edge1 yields the "fade out" curve.
    float oceanFloorHeight(vec2 p) {
        return -25.0 + 4.0 * fbm2(p, 0.003, vec2(2.1, 4.7));
    }
    float anchoredHeight(vec2 p) {
        float falloff = uFalloff;
        int n = uAnchorCount;
        float ws[MAX_ANCHORS];
        float sum = 0.0;
        for (int i = 0; i < MAX_ANCHORS; i++) {
            if (i >= n) break;
            vec2 dv = p - uAnchors[i].xy;
            float d = sqrt(dot(dv, dv));
            float w = exp(-d / falloff);
            ws[i] = w;
            sum += w;
        }
        float inv = 1.0 / max(sum, 1e-6);
        float y = 0.0;
        for (int i = 0; i < MAX_ANCHORS; i++) {
            if (i >= n) break;
            float w = ws[i];
            int bi = int(uAnchors[i].w + 0.5);
            y += (w * inv) * (biomeContribution(p, bi) + uAnchors[i].z);
        }
        float anchorInfluence = smoothstep(0.3, 2.5, sum);
        return mix(oceanFloorHeight(p), y, anchorInfluence);
    }

    // Per-fragment biome metadata: weighted-avg snowline & rockSlope.
    void anchoredMeta(vec2 p, out float snowline, out float rockSlope, out float dominantIdx) {
        float falloff = uFalloff;
        int n = uAnchorCount;
        float sum = 0.0;
        float sn = 0.0, rs = 0.0;
        float bestW = 0.0;
        int bestI = 0;
        for (int i = 0; i < MAX_ANCHORS; i++) {
            if (i >= n) break;
            vec2 dv = p - uAnchors[i].xy;
            float d = sqrt(dot(dv, dv));
            float w = exp(-d / falloff);
            if (w > bestW) { bestW = w; bestI = i; }
            sum += w;
            int bi = int(uAnchors[i].w + 0.5);
            sn += w * uBiomes1[bi].z;
            rs += w * uBiomes1[bi].w;
        }
        snowline = sn / max(sum, 1e-6);
        rockSlope = rs / max(sum, 1e-6);
        dominantIdx = float(bestI);
    }

    // ============ Polar-grid (ringT, theta) -> world XZ ============
    // position.x = radial param t in [0..1] (0 only for center vertex)
    // position.z = angle theta in [0..2pi]
    // Bias: rNorm = pow(t, p). p<1 inner-dense, p>1 outer-dense, p=1 uniform.
    vec3 polarToWorld(float t, float theta) {
        float p = uConcentration;
        float rNorm = (t <= 0.0) ? 0.0 : pow(t, p);
        float r = rNorm * POLAR_MAX_R;
        vec2 xz = uSampleCameraPos.xz + r * vec2(cos(theta), sin(theta));
        return vec3(xz.x, 0.0, xz.y);
    }

    // Combined: height + biome metadata in one anchor pass.
    void heightAndMeta(vec2 p, out float height, out float snowline, out float rockSlope, out float dominantIdx) {
        float falloff = uFalloff;
        int n = uAnchorCount;
        float ws[MAX_ANCHORS];
        float sum = 0.0;
        float bestW = 0.0; int bestI = 0;
        float sn = 0.0, rs = 0.0;
        for (int i = 0; i < MAX_ANCHORS; i++) {
            if (i >= n) break;
            vec2 dv = p - uAnchors[i].xy;
            float d = sqrt(dot(dv, dv));
            float w = exp(-d / falloff);
            ws[i] = w;
            sum += w;
            if (w > bestW) { bestW = w; bestI = i; }
            int bi = int(uAnchors[i].w + 0.5);
            sn += w * uBiomes1[bi].z;
            rs += w * uBiomes1[bi].w;
        }
        float inv = 1.0 / max(sum, 1e-6);
        snowline = sn * inv; rockSlope = rs * inv; dominantIdx = float(bestI);
        float y = 0.0;
        for (int i = 0; i < MAX_ANCHORS; i++) {
            if (i >= n) break;
            float w = ws[i];
            int bi = int(uAnchors[i].w + 0.5);
            y += (w * inv) * (biomeContribution(p, bi) + uAnchors[i].z);
        }
        float anchorInfluence = smoothstep(0.3, 2.5, sum);
        height = mix(oceanFloorHeight(p), y, anchorInfluence);
    }

    void main() {
        vec3 worldPos = polarToWorld(position.x, position.z);
        float h, snowline, rockSlope, dominantIdx;
        heightAndMeta(worldPos.xz, h, snowline, rockSlope, dominantIdx);
        float scaled = h * (uHeight / 20.0);
        worldPos.y = scaled;
        // Forward-difference gradient for smooth-shaded normal (2 extra height samples).
        // eps scales with viewing distance so far-LOD faces stay smooth without spiking gradient.
        float dist = length(worldPos.xz - uSampleCameraPos.xz);
        float eps = max(1.5, dist * 0.012);
        float hxp = anchoredHeight(worldPos.xz + vec2(eps, 0.0));
        float hzp = anchoredHeight(worldPos.xz + vec2(0.0, eps));
        float scl = uHeight / 20.0;
        float gx = (hxp - h) / eps * scl;
        float gz = (hzp - h) / eps * scl;
        vNormal = normalize(vec3(-gx, 1.0, -gz));
        vPosition = worldPos;
        vWorld = worldPos;
        vHeight = scaled;
        vSnowline = snowline;
        vRockSlope = rockSlope;
        vBiomeIdx = dominantIdx;
        gl_Position = projectionMatrix * viewMatrix * vec4(worldPos, 1.0);
    }
