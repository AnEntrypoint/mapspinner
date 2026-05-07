
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uTHeight;
    uniform vec3 uSampleCameraPos;
    uniform float uMaxR;
    uniform float uConcentration;
    uniform float uBakeSize; // texel count along one axis
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
    void main() {
        const float WORLD_STEP = 2.0;
        vec2 wC = bakeUVToWorld(vUv);
        vec2 uvXp = worldToBakeUV(wC + vec2( WORLD_STEP, 0.0));
        vec2 uvXm = worldToBakeUV(wC + vec2(-WORLD_STEP, 0.0));
        vec2 uvZp = worldToBakeUV(wC + vec2(0.0,  WORLD_STEP));
        vec2 uvZm = worldToBakeUV(wC + vec2(0.0, -WORLD_STEP));
        // 4-tap on packed height at fixed world-space offsets (unpack: m = R*300 - 100)
        float hC = texture2D(uTHeight, vUv).r * 300.0 - 100.0;
        float hxp = texture2D(uTHeight, uvXp).r * 300.0 - 100.0;
        float hxm = texture2D(uTHeight, uvXm).r * 300.0 - 100.0;
        float hzp = texture2D(uTHeight, uvZp).r * 300.0 - 100.0;
        float hzm = texture2D(uTHeight, uvZm).r * 300.0 - 100.0;
        float dhdx = (hxp - hxm) / (2.0 * WORLD_STEP);
        float dhdz = (hzp - hzm) / (2.0 * WORLD_STEP);
        vec3 N = normalize(vec3(-dhdx, 1.0, -dhdz));
        float lap = (hxp + hxm + hzp + hzm - 4.0 * hC) / (WORLD_STEP * WORLD_STEP);
        float curv = clamp(lap * 5.0 + 0.5, 0.0, 1.0);
        gl_FragColor = vec4(texture2D(uTHeight, vUv).r, N.x * 0.5 + 0.5, N.z * 0.5 + 0.5, curv);
    }
