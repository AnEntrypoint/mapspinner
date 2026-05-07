
    precision highp float;
    #define MAX_ANCHORS 64
    #define MAX_BIOMES 12
    varying vec2 vUv;
    uniform vec3 uSampleCameraPos;
    uniform float uMaxR;
    uniform float uConcentration;
    uniform float uHeight;
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
        vec3 t2 = t*t; vec3 t4 = t2*t2;
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
    float anchoredHeight(vec2 p) {
        float falloff = uFalloff;
        int n = uAnchorCount;
        float ws[MAX_ANCHORS]; float sum = 0.0;
        for (int i=0;i<MAX_ANCHORS;i++) {
            if (i>=n) break;
            vec2 dv = p - uAnchors[i].xy;
            float d = sqrt(dot(dv,dv));
            float w = exp(-d / falloff);
            ws[i] = w; sum += w;
        }
        float inv = 1.0 / max(sum, 1e-6);
        float y = 0.0;
        for (int i=0;i<MAX_ANCHORS;i++) {
            if (i>=n) break;
            float w = ws[i];
            int bi = int(uAnchors[i].w + 0.5);
            y += (w * inv) * (biomeContribution(p, bi) + uAnchors[i].z);
        }
        float anchorInfluence = smoothstep(0.3, 2.5, sum);
        float oceanFloor = -25.0 + 4.0 * fbm2(p, 0.003, vec2(2.1, 4.7));
        return mix(oceanFloor, y, anchorInfluence);
    }
    void main() {
        // Pass A: write only the scaled height (single anchored loop) — cheap.
        // Pass B reads this and finishes tGeom via dFdx/dFdy.
        vec2 worldXZ = bakeUVToWorld(vUv);
        float scl = uHeight / 20.0;
        float h0 = anchoredHeight(worldXZ) * scl;
        float packedH = clamp((h0 + 100.0) / 300.0, 0.0, 1.0);
        gl_FragColor = vec4(packedH, 0.5, 0.5, 1.0);
    }
