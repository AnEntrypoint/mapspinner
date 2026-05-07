
    precision highp float;
    #define MAX_ANCHORS 64
    #define MAX_BIOMES 12
    varying vec2 vUv;
    uniform vec3 uSampleCameraPos;
    uniform float uMaxR;
    uniform float uConcentration;
    uniform float uHeight;
    uniform float uSeason;
    uniform int uAnchorCount;
    uniform float uFalloff;
    uniform vec4 uAnchors[MAX_ANCHORS];
    uniform vec4 uBiomes0[MAX_BIOMES];
    uniform vec4 uBiomes1[MAX_BIOMES];

    vec2 bakeUVToWorld(vec2 uv) {
        vec2 d = uv * 2.0 - 1.0;
        float len = length(d);
        if (len < 1e-6) return uSampleCameraPos.xz;
        float rNorm = pow(len, uConcentration);
        return uSampleCameraPos.xz + (rNorm * uMaxR) * (d / len);
    }
    vec2 worldToBakeUV(vec2 worldXZ) {
        vec2 dxz = worldXZ - uSampleCameraPos.xz;
        float r = length(dxz);
        if (r < 1e-6) return vec2(0.5);
        float rNorm = clamp(r / uMaxR, 0.0, 1.0);
        float t = pow(rNorm, 1.0 / max(uConcentration, 1e-3));
        vec2 dir = dxz / r;
        return clamp(0.5 + 0.5 * t * dir, vec2(0.0), vec2(1.0));
    }

    vec3 permute3(vec3 x) { return mod(((x * 34.0) + 1.0) * x, 289.0); }
    float snoise(vec2 v) {
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
        vec3 norm = 1.79284291400159 - 0.85373472095314 * (a0*a0 + h*h);
        vec2 g0 = vec2(a0.x, h.x) * norm.x;
        vec2 g1 = vec2(a0.y, h.y) * norm.y;
        vec2 g2 = vec2(a0.z, h.z) * norm.z;
        vec3 t = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
        vec3 t4 = t*t; t4 = t4*t4;
        float gd0 = dot(g0, x0), gd1 = dot(g1, x12.xy), gd2 = dot(g2, x12.zw);
        return 130.0 * dot(t4, vec3(gd0, gd1, gd2));
    }
    float fbm3(vec2 p, float freq, vec2 salt) {
        float h=0.0,a=1.0,f=freq; vec2 o=salt;
        h+=a*snoise(p*f+o); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
        h+=a*snoise(p*f+o); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
        h+=a*snoise(p*f+o); return h;
    }
    float fbm2(vec2 p, float freq, vec2 salt) {
        float h=0.0,a=1.0,f=freq; vec2 o=salt;
        h+=a*snoise(p*f+o); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
        h+=a*snoise(p*f+o); return h;
    }
    float ridgeFbm(vec2 p, float freq, vec2 salt) {
        float h=0.0,a=1.0,f=freq; vec2 o=salt;
        h+=a*pow(1.0-abs(snoise(p*f+o)),0.7); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
        h+=a*pow(1.0-abs(snoise(p*f+o)),0.7); a*=0.5; f*=2.0; o=vec2(o.y,-o.x)+vec2(11.1,7.3);
        h+=a*pow(1.0-abs(snoise(p*f+o)),0.7); return h-0.875;
    }
    float biomeErosionStrength(int bi) {
        if (bi==0) return 1.20; if (bi==1) return 1.30; if (bi==2) return 0.85;
        if (bi==3) return 0.55; if (bi==4) return 0.45; if (bi==5) return 0.10;
        if (bi==6) return 0.65; if (bi==7) return 0.50; if (bi==8) return 0.40;
        return 0.7;
    }
    float applyErosionDelta(vec2 p, int bi) {
        float em = fbm3(p, 0.005, vec2(3.7, 5.1));
        float ed = fbm2(p, 0.02,  vec2(8.3, 1.9));
        float channels = (1.0 - abs(em)) * (1.0 - abs(ed)) * 0.7;
        float rough = fbm2(p, 0.08, vec2(17.7, 23.1)) * 0.10;
        return (-channels * 0.15 + rough * 0.15) * biomeErosionStrength(bi) * 20.0;
    }
    float biomeContribution(vec2 p, int bi) {
        vec4 b0 = uBiomes0[bi]; vec4 b1 = uBiomes1[bi];
        float amp=b0.x, freq=b0.y, elev=b0.z, continentMix=b0.w;
        float ridgeMix=b1.x, rollMix=b1.y;
        float bf = float(bi);
        vec2 saltC = vec2(13.0+bf*7.1, 29.0-bf*3.3);
        vec2 saltR = vec2(91.0-bf*5.7, 41.0+bf*2.9);
        vec2 saltL = vec2(57.0+bf*11.0, 17.0-bf*6.4);
        float cont = fbm3(p,freq,saltC);
        float ridge = ridgeFbm(p,freq*2.0,saltR);
        float roll = fbm2(p,freq*0.5,saltL);
        float erosion = fbm2(p, freq*3.0, saltL+vec2(31.7,19.1)) * 0.3;
        float broadGate = clamp(cont*0.5+0.5, 0.0, 1.0);
        ridge = ridge * (1.0 - erosion) * broadGate;
        float h = elev + amp*(cont*continentMix + ridge*ridgeMix + roll*rollMix);
        h += applyErosionDelta(p, bi);
        return h;
    }
    // Snowline-only anchor blend (cheap: no biomeContribution calls)
    // Bare exponential weight, no fadeIn/fadeOut cutoff -> C-infinity smooth, no seam.
    float snowlineAt(vec2 p) {
        float falloff = uFalloff;
        int n = uAnchorCount;
        float sum = 0.0, sn = 0.0;
        for (int i=0;i<MAX_ANCHORS;i++) {
            if (i>=n) break;
            vec2 dv = p - uAnchors[i].xy;
            float d = sqrt(dot(dv,dv));
            float w = exp(-d / falloff);
            sum += w;
            int bi = int(uAnchors[i].w + 0.5);
            sn += w * uBiomes1[bi].z;
        }
        return sn / max(sum, 1e-6);
    }
    uniform sampler2D uTHeight;
    uniform float uBakeSize;
    void main() {
        vec2 worldXZ = bakeUVToWorld(vUv);
        // Sample baked height (already scaled meters: r*300 - 100)
        float height = texture2D(uTHeight, vUv).r * 300.0 - 100.0;
        // Slope from fixed world-space finite-diff on baked tHeight
        const float WORLD_STEP = 2.0;
        vec2 uvXp = worldToBakeUV(worldXZ + vec2( WORLD_STEP, 0.0));
        vec2 uvXm = worldToBakeUV(worldXZ + vec2(-WORLD_STEP, 0.0));
        vec2 uvZp = worldToBakeUV(worldXZ + vec2(0.0,  WORLD_STEP));
        vec2 uvZm = worldToBakeUV(worldXZ + vec2(0.0, -WORLD_STEP));
        float hxp = texture2D(uTHeight, uvXp).r * 300.0 - 100.0;
        float hxm = texture2D(uTHeight, uvXm).r * 300.0 - 100.0;
        float hzp = texture2D(uTHeight, uvZp).r * 300.0 - 100.0;
        float hzm = texture2D(uTHeight, uvZm).r * 300.0 - 100.0;
        float dhdx = (hxp - hxm) / (2.0 * WORLD_STEP);
        float dhdz = (hzp - hzm) / (2.0 * WORLD_STEP);
        float slopeMag = sqrt(dhdx*dhdx + dhdz*dhdz);
        float realSlope = slopeMag / sqrt(slopeMag*slopeMag + 1.0);
        float realSlopeForMat = realSlope * 0.65;

        float snowline = snowlineAt(worldXZ);

        float n_lg = snoise(worldXZ * 0.018) * 0.5 + 0.5;
        float n_md = snoise(worldXZ * 0.075) * 0.5 + 0.5;
        float blendJitter = (n_lg - 0.5) * 6.0 + (n_md - 0.5) * 2.5;

        // Season weights
        float wSpring = max(1.0 - abs(uSeason - 0.0)   * 4.0, 0.0);
        float wSummer = max(1.0 - abs(uSeason - 0.25)  * 4.0, 0.0);
        float wAutumn = max(1.0 - abs(uSeason - 0.5)   * 4.0, 0.0);
        float wWinter = max(1.0 - abs(uSeason - 0.75)  * 4.0, 0.0);
        float wSum = wSpring + wSummer + wAutumn + wWinter + 1e-5;
        wSpring /= wSum; wSummer /= wSum; wAutumn /= wSum; wWinter /= wSum;
        float seasonSnowOffset = wWinter * (-25.0) + wSummer * 12.0 + wSpring * 6.0 + wAutumn * (-4.0);
        float effSnowline = snowline + seasonSnowOffset;

        float grassBand = smoothstep(3.0, 12.0, height + blendJitter) * (1.0 - smoothstep(effSnowline - 30.0, effSnowline - 5.0, height));
        float forestMix = smoothstep(16.0, 42.0, height + blendJitter * 1.4) * (1.0 - smoothstep(effSnowline - 20.0, effSnowline + 5.0, height));
        float rockMix   = smoothstep(0.35, 0.62, realSlopeForMat + 0.18 * n_md);
        float snowAlt   = smoothstep(effSnowline - 6.0, effSnowline + 6.0, height + 4.0 * (n_lg - 0.5));
        float snowSlope = 1.0 - smoothstep(0.30, 0.55, realSlopeForMat);
        float snowMix   = snowAlt * snowSlope;

        gl_FragColor = vec4(rockMix, snowMix, grassBand, forestMix);
    }
