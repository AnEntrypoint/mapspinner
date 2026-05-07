
    precision highp float;
    varying vec2 vUv;
    uniform vec3 uSampleCameraPos;
    uniform float uMaxR;
    uniform float uConcentration;
    uniform float uSeason;
    uniform sampler2D uTMat;

    vec2 bakeUVToWorld(vec2 uv) {
        vec2 d = uv * 2.0 - 1.0;
        float len = length(d);
        if (len < 1e-6) return uSampleCameraPos.xz;
        float rNorm = pow(len, uConcentration);
        return uSampleCameraPos.xz + (rNorm * uMaxR) * (d / len);
    }

    float hash21(vec2 p) {
        p = fract(p * vec2(123.34, 456.21));
        p += dot(p, p + 45.32);
        return fract(p.x * p.y);
    }
    float hash21Spawn(vec2 p) {
        float h = sin(p.x * 127.1 + p.y * 311.7) * 43758.5453;
        return fract(h);
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
    float vnoiseSpawn(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        float a = hash21Spawn(i);
        float b = hash21Spawn(i + vec2(1.0, 0.0));
        float c = hash21Spawn(i + vec2(0.0, 1.0));
        float d = hash21Spawn(i + vec2(1.0, 1.0));
        vec2 u = f * f * (3.0 - 2.0 * f);
        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
    }

    void main() {
        vec2 wp = bakeUVToWorld(vUv);
        vec4 mat = texture2D(uTMat, vUv);
        float bakedRockMix   = mat.r;
        float bakedSnowMix   = mat.g;
        float bakedGrassMix  = mat.b;
        float bakedForestMix = mat.a;

        float n_xl = vnoise(wp * 0.0035);
        float n_lg = vnoise(wp * 0.018);
        float n_md = vnoise(wp * 0.075);

        float wSpring = max(1.0 - abs(uSeason - 0.0)  * 4.0, 0.0);
        float wSummer = max(1.0 - abs(uSeason - 0.25) * 4.0, 0.0);
        float wAutumn = max(1.0 - abs(uSeason - 0.5)  * 4.0, 0.0);
        float wWinter = max(1.0 - abs(uSeason - 0.75) * 4.0, 0.0);
        float wSum = wSpring + wSummer + wAutumn + wWinter + 1e-5;
        wSpring /= wSum; wSummer /= wSum; wAutumn /= wSum; wWinter /= wSum;

        vec3 sand = mix(vec3(0.86, 0.76, 0.56), vec3(0.94, 0.86, 0.66), n_md);
        sand *= 0.92 + 0.16 * n_lg;
        sand += vec3(0.03, 0.01, -0.01) * (n_xl - 0.5);

        vec3 grassA = vec3(0.22, 0.42, 0.14);
        vec3 grassB = vec3(0.34, 0.58, 0.20);
        vec3 grassC = vec3(0.30, 0.46, 0.16);
        vec3 grass  = mix(grassA, grassB, n_md);
        grass = mix(grass, grassC, smoothstep(0.55, 0.85, n_lg) * 0.45);
        grass *= 0.82 + 0.28 * n_xl;
        vec3 grassSpring = grass * vec3(0.85, 1.15, 0.80);
        vec3 grassSummer = grass;
        vec3 grassAutumn = grass * vec3(1.35, 0.95, 0.55);
        vec3 grassWinter = mix(grass, vec3(0.78, 0.76, 0.70), 0.75);
        grass = grassSpring*wSpring + grassSummer*wSummer + grassAutumn*wAutumn + grassWinter*wWinter;

        vec3 forestA = vec3(0.08, 0.26, 0.10);
        vec3 forestB = vec3(0.16, 0.40, 0.16);
        vec3 forestC = vec3(0.20, 0.32, 0.12);
        vec3 forest  = mix(forestA, forestB, n_md);
        forest = mix(forest, forestC, smoothstep(0.6, 0.9, n_lg) * 0.5);
        forest *= 0.85 + 0.25 * n_xl;
        vec3 forestSpring = forest * vec3(0.90, 1.15, 0.85);
        vec3 forestSummer = forest;
        vec3 forestAutumn = forest * vec3(1.55, 1.00, 0.50);
        vec3 forestWinter = mix(forest, vec3(0.40, 0.42, 0.40), 0.55);
        forest = forestSpring*wSpring + forestSummer*wSummer + forestAutumn*wAutumn + forestWinter*wWinter;

        vec3 rockBase = mix(vec3(0.44, 0.40, 0.36), vec3(0.62, 0.57, 0.50), n_md);
        vec3 rockWarm = vec3(0.56, 0.40, 0.28);
        float oxide   = smoothstep(0.55, 0.85, n_lg);
        vec3 rock     = mix(rockBase, rockWarm, oxide * 0.35);
        rock *= 0.72 + 0.36 * n_xl;

        vec3 snow = mix(vec3(0.84, 0.88, 0.96), vec3(1.00, 0.99, 0.95), 0.6);

        vec3 albedo = sand;
        albedo = mix(albedo, grass, bakedGrassMix);
        albedo = mix(albedo, forest, bakedForestMix * 0.55);
        albedo = mix(albedo, rock, bakedRockMix);
        albedo = mix(albedo, snow, bakedSnowMix);

        float densityField = vnoiseSpawn(wp * 0.05);
        float vegDensity = clamp(bakedForestMix * 1.5 + bakedRockMix * 0.8 + bakedGrassMix * 0.4 + 0.35, 0.0, 1.0);
        float biotic = vegDensity * (1.0 - bakedSnowMix) * densityField * 0.92;
        albedo = mix(albedo, albedo * vec3(0.40, 0.45, 0.32), clamp(biotic * 0.95, 0.0, 1.0));

        gl_FragColor = vec4(albedo, 1.0);
    }
