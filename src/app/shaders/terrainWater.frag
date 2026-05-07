
    precision highp float;
    #define MAX_ANCHORS 64
    #define MAX_BIOMES 12
    varying vec3 vWorld;
    uniform float uTime;
    uniform float uTimeOfDay;
    uniform float uSeason;
    uniform float uHeight;
    uniform int uAnchorCount;
    uniform float uFalloff;
    uniform vec4 uAnchors[MAX_ANCHORS];
    uniform vec4 uBiomes0[MAX_BIOMES];
    uniform vec4 uBiomes1[MAX_BIOMES];
    uniform sampler2D uWaterSim;
    uniform vec2 uWaterSimCenter;
    uniform vec2 uWaterSimTexel;
    uniform float uWaterSimWorldSize;
    uniform float uWaterSimAmp;
    const float WPI = 3.14159265359;
    float hash21(vec2 p) { p = fract(p*vec2(123.34,456.21)); p += dot(p,p+45.32); return fract(p.x*p.y); }
    float vnoise(vec2 p) {
        vec2 i = floor(p), f = fract(p);
        float a = hash21(i), b = hash21(i+vec2(1,0)), c = hash21(i+vec2(0,1)), d = hash21(i+vec2(1,1));
        vec2 u = f*f*(3.0-2.0*f);
        return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
    }
    // Cheap depth proxy: nearest anchor only (single 64-iter find-min loop).
    // Avoids the full weighted-blend cost of the terrain shader.
    float waterTerrainProxy(vec2 p) {
        float falloff = uFalloff;
        int n = uAnchorCount;
        float bestDSq = 1e18; float anchorElev = 0.0; float baseElev = 0.0;
        for (int i = 0; i < MAX_ANCHORS; i++) {
            if (i >= n) break;
            vec2 dv = p - uAnchors[i].xy;
            float dSq = dot(dv, dv);
            if (dSq < bestDSq) {
                bestDSq = dSq;
                anchorElev = uAnchors[i].z;
                int bi = int(uAnchors[i].w + 0.5);
                baseElev = uBiomes0[bi].z;
            }
        }
        float dn = sqrt(bestDSq) / falloff;
        float w = exp(-dn);
        return (anchorElev * w + baseElev * (1.0 - w * 0.5)) * (uHeight / 20.0);
    }
    vec3 sunDirFromTOD(float tt) {
        float el = sin((tt - 0.25) * 2.0 * WPI);
        float az = tt * 2.0 * WPI;
        float horiz = sqrt(max(1.0 - el*el, 0.0));
        return normalize(vec3(cos(az)*horiz, el, sin(az)*horiz));
    }
    vec3 skyColorDir(vec3 d) {
        vec3 sd = sunDirFromTOD(uTimeOfDay);
        float sunEl = sd.y;
        float dayAmt   = smoothstep(-0.05, 0.30, sunEl);
        float twiAmt   = (1.0 - dayAmt) * smoothstep(-0.30, 0.05, sunEl);
        float nightAmt = 1.0 - smoothstep(-0.15, 0.05, sunEl);
        float t = clamp(d.y*0.5+0.5, 0.0, 1.0);
        vec3 dHor = vec3(0.78,0.84,0.92), dMid = vec3(0.55,0.72,0.90), dZen = vec3(0.20,0.42,0.74);
        vec3 day = mix(mix(dHor, dMid, smoothstep(0.0,0.45,t)), dZen, smoothstep(0.45,1.0,t));
        float sunAlign = max(dot(normalize(vec3(d.x,0.0,d.z)), normalize(vec3(sd.x,0.0,sd.z))), 0.0);
        vec3 tHor = mix(vec3(0.55,0.32,0.40), vec3(1.05,0.55,0.30), sunAlign);
        vec3 tMid = vec3(0.40,0.32,0.46), tZen = vec3(0.10,0.14,0.32);
        vec3 twi = mix(mix(tHor,tMid,smoothstep(0.0,0.45,t)), tZen, smoothstep(0.45,1.0,t));
        vec3 nHor = vec3(0.04,0.06,0.12), nMid = vec3(0.02,0.03,0.08), nZen = vec3(0.005,0.01,0.04);
        vec3 night = mix(mix(nHor,nMid,smoothstep(0.0,0.45,t)), nZen, smoothstep(0.45,1.0,t));
        return day*dayAmt + twi*twiAmt + night*nightAmt;
    }
    void main() {
        vec2 wp = vWorld.xz;
        // Distance-attenuated waves — fewer octaves at distance
        float dCam = length(vWorld - cameraPosition);
        float lod = smoothstep(40.0, 240.0, dCam); // 0=close (full octaves), 1=far (drop micro-chop)
        float eps = 0.6;
        // octave 1: large slow swells
        vec2 a1 = wp * 0.035 + vec2(uTime * 0.07, uTime * 0.04);
        float L1 = vnoise(a1);
        float L1x = vnoise(a1 + vec2(eps, 0.0)) - L1;
        float L1z = vnoise(a1 + vec2(0.0, eps)) - L1;
        // octave 2: medium ripples
        vec2 a2 = wp * 0.13 + vec2(-uTime * 0.18, uTime * 0.11);
        float L2 = vnoise(a2);
        float L2x = vnoise(a2 + vec2(eps, 0.0)) - L2;
        float L2z = vnoise(a2 + vec2(0.0, eps)) - L2;
        // octave 3: fast micro-chop — only when close
        float L3 = 0.0, L3x = 0.0, L3z = 0.0;
        if (lod < 0.95) {
            vec2 a3 = wp * 0.55 + vec2(uTime * 0.55, -uTime * 0.40);
            L3 = vnoise(a3);
            L3x = vnoise(a3 + vec2(eps, 0.0)) - L3;
            L3z = vnoise(a3 + vec2(0.0, eps)) - L3;
        }
        float micro = (1.0 - lod);
        float Nx = L1x * 1.6 + L2x * 1.0 + L3x * 0.55 * micro;
        float Nz = L1z * 1.6 + L2z * 1.0 + L3z * 0.55 * micro;
        // Wave-equation sim ripple contribution (raindrops, etc.) — sampled as a
        // world-anchored tile of size uWaterSimWorldSize centred on uWaterSimCenter.
        vec2 simUv = (wp - uWaterSimCenter) / uWaterSimWorldSize + 0.5;
        // Wrap so the tile repeats outside its bounds (negligible visual artefact at sea scale).
        simUv = fract(simUv);
        float simH  = texture2D(uWaterSim, simUv).r;
        float simHx = texture2D(uWaterSim, simUv + vec2(uWaterSimTexel.x, 0.0)).r;
        float simHz = texture2D(uWaterSim, simUv + vec2(0.0, uWaterSimTexel.y)).r;
        float Sx = (simHx - simH) * uWaterSimAmp;
        float Sz = (simHz - simH) * uWaterSimAmp;
        Nx += Sx;
        Nz += Sz;
        vec3 N = normalize(vec3(-Nx, 1.0, -Nz));
        vec3 V = normalize(cameraPosition - vWorld);
        vec3 L = sunDirFromTOD(uTimeOfDay);
        vec3 H = normalize(L + V);
        float NdotV = max(dot(N, V), 0.0);
        float NdotH = max(dot(N, H), 0.0);
        // Strong fresnel — grazing fully reflective, normal incidence shows water color
        float F0 = 0.02;
        float F = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
        // Time-of-day water color: sunrise/sunset warm, day teal, night dark
        float sunEl = L.y;
        float dayAmt   = smoothstep(-0.05, 0.30, sunEl);
        float twiAmt   = (1.0 - dayAmt) * smoothstep(-0.30, 0.05, sunEl);
        float nightAmt = 1.0 - smoothstep(-0.15, 0.05, sunEl);
        vec3 deepDay    = vec3(0.02, 0.10, 0.16);
        vec3 shallowDay = vec3(0.10, 0.34, 0.40);
        vec3 deepTwi    = vec3(0.18, 0.10, 0.12);
        vec3 shallowTwi = vec3(0.55, 0.32, 0.24);
        vec3 deepNight  = vec3(0.01, 0.02, 0.05);
        vec3 shallowNight=vec3(0.03, 0.05, 0.10);
        vec3 deep    = deepDay*dayAmt + deepTwi*twiAmt + deepNight*nightAmt;
        vec3 shallow = shallowDay*dayAmt + shallowTwi*twiAmt + shallowNight*nightAmt;
        // Underwater color via terrain height proxy — approximates depth at this xz
        float terrainH = waterTerrainProxy(wp);
        float depth = max(0.0, -terrainH); // how far the bed is below water plane
        float depthMix = 1.0 - exp(-depth * 0.18);
        vec3 refracted = mix(shallow, deep, depthMix);
        // Slight ripple variation
        refracted += (L1 + L2 - 1.0) * vec3(0.02, 0.04, 0.05);
        // Winter: shift toward icy gray/white
        float wWinter = max(1.0 - abs(uSeason - 0.75) * 4.0, 0.0);
        refracted = mix(refracted, vec3(0.65, 0.70, 0.74), wWinter * 0.7);
        // Reflected sky
        vec3 R = reflect(-V, N);
        if (R.y < 0.0) R.y = -R.y * 0.3; // clamp downward reflection back up
        vec3 reflected = skyColorDir(R);
        vec3 col = mix(refracted, reflected, F);
        // Tight specular sun glint — high-power Phong on perturbed N
        float glint = pow(NdotH, 320.0) * 2.4;
        // soften glint at night
        vec3 glintCol = mix(vec3(0.6, 0.7, 1.0), vec3(1.0, 0.97, 0.86), dayAmt + twiAmt);
        col += glintCol * glint * (dayAmt + twiAmt * 0.7 + nightAmt * 0.15);
        // Foam at shore — where bed is near water surface
        float foamMask = 1.0 - smoothstep(-0.5, 1.5, depth);
        float foamPattern = smoothstep(0.45, 0.85, vnoise(wp * 0.9 + vec2(uTime * 0.4, 0.0)));
        float foam = foamMask * (0.4 + 0.6 * foamPattern);
        col = mix(col, vec3(0.92, 0.95, 0.96), clamp(foam, 0.0, 0.85));
        // Time-of-day atmospheric haze (matches terrain)
        float dist = length(vWorld - cameraPosition);
        vec3 viewDirH = normalize(vWorld - cameraPosition);
        float th = uTimeOfDay;
        float mMist = smoothstep(0.18, 0.27, th) * (1.0 - smoothstep(0.27, 0.35, th));
        float sHaze = smoothstep(0.65, 0.75, th) * (1.0 - smoothstep(0.75, 0.85, th));
        float nDip  = smoothstep(0.85, 1.0, th) + (1.0 - smoothstep(0.0, 0.15, th));
        float densityW = 0.0035 * (1.0 + mMist * 1.5 + sHaze * 0.6 - 0.3 * nDip);
        densityW = max(densityW, 0.0008);
        float heightFactorW = 1.0 - clamp(vWorld.y / (uHeight * 1.5), 0.0, 0.6);
        float atm = (1.0 - exp(-dist * densityW * heightFactorW)) * 0.95;
        vec3 skyAtViewW = skyColorDir(viewDirH);
        vec3 hazeTintW;
        if (sHaze > 0.0 || mMist > 0.0) {
            hazeTintW = mix(vec3(1.0), vec3(1.05, 0.92, 0.78), clamp(sHaze + mMist, 0.0, 1.0));
        } else if (nDip > 0.5) {
            hazeTintW = vec3(0.55, 0.65, 0.85);
        } else {
            hazeTintW = vec3(0.95, 0.98, 1.04);
        }
        vec3 skyHere = skyAtViewW * hazeTintW;
        col = mix(col, skyHere, atm);
        float alpha = mix(0.72, 0.95, F);
        alpha = max(alpha, foam * 0.9);
        gl_FragColor = vec4(col, alpha);
    }
