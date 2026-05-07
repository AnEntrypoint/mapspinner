
    precision highp float;
    varying vec2 vUv;
    uniform sampler2D uPrev;     // R = h(t-1), G = h(t-2)
    uniform vec2 uTexel;         // 1/size
    uniform float uDamping;
    uniform vec4 uImpulse;       // xy = uv center, z = radius (uv units), w = strength (0 = no impulse)
    void main() {
        vec4 c = texture2D(uPrev, vUv);
        float h = c.r;
        float hPrev = c.g;
        // Laplacian (5-tap)
        float hL = texture2D(uPrev, vUv + vec2(-uTexel.x, 0.0)).r;
        float hR = texture2D(uPrev, vUv + vec2( uTexel.x, 0.0)).r;
        float hD = texture2D(uPrev, vUv + vec2(0.0, -uTexel.y)).r;
        float hU = texture2D(uPrev, vUv + vec2(0.0,  uTexel.y)).r;
        float lap = (hL + hR + hD + hU) - 4.0 * h;
        // Standard discrete wave eq (c^2 dt^2 / dx^2 absorbed into 0.5 factor for stability with 5-tap).
        float hNew = (2.0 * h - hPrev) + 0.5 * lap;
        hNew *= uDamping;
        // Optional impulse (raindrop).
        if (uImpulse.w > 0.0) {
            vec2 d = vUv - uImpulse.xy;
            float r = length(d);
            float fall = exp(-(r*r) / (uImpulse.z * uImpulse.z));
            hNew += uImpulse.w * fall;
        }
        gl_FragColor = vec4(hNew, h, 0.0, 1.0);
    }
