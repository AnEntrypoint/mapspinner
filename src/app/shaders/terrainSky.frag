
    precision highp float;
    varying vec2 vUv;
    uniform float uTimeOfDay;
    const float PI = 3.14159265359;
    vec3 sunDirFromTOD(float tt) {
        float el = sin((tt - 0.25) * 2.0 * PI);
        float az = tt * 2.0 * PI;
        float horiz = sqrt(max(1.0 - el*el, 0.0));
        return normalize(vec3(cos(az)*horiz, el, sin(az)*horiz));
    }
    void main() {
        // vUv.y is screen, treat as elevation 0..1; build a fake view-dir on the sun azimuth plane
        float t = clamp(vUv.y, 0.0, 1.0);
        vec3 sd = sunDirFromTOD(uTimeOfDay);
        float sunEl = sd.y;
        float dayAmt   = smoothstep(-0.05, 0.30, sunEl);
        float twiAmt   = (1.0 - dayAmt) * smoothstep(-0.30, 0.05, sunEl);
        float nightAmt = 1.0 - smoothstep(-0.15, 0.05, sunEl);
        vec3 dHor = vec3(0.78, 0.84, 0.92);
        vec3 dMid = vec3(0.55, 0.72, 0.90);
        vec3 dZen = vec3(0.20, 0.42, 0.74);
        vec3 day = mix(mix(dHor, dMid, smoothstep(0.0, 0.45, t)), dZen, smoothstep(0.45, 1.0, t));
        // For background plane, fake horizontal alignment with screen-x
        float sunAlign = clamp(0.5 + (vUv.x - 0.5) * (sd.x > 0.0 ? 1.0 : -1.0), 0.0, 1.0);
        vec3 tHor = mix(vec3(0.55, 0.32, 0.40), vec3(1.05, 0.55, 0.30), sunAlign);
        vec3 tMid = vec3(0.40, 0.32, 0.46);
        vec3 tZen = vec3(0.10, 0.14, 0.32);
        vec3 twi = mix(mix(tHor, tMid, smoothstep(0.0, 0.45, t)), tZen, smoothstep(0.45, 1.0, t));
        vec3 nHor = vec3(0.04, 0.06, 0.12);
        vec3 nMid = vec3(0.02, 0.03, 0.08);
        vec3 nZen = vec3(0.005, 0.01, 0.04);
        vec3 night = mix(mix(nHor, nMid, smoothstep(0.0, 0.45, t)), nZen, smoothstep(0.45, 1.0, t));
        // Hashed jittered starfield (matches terrain fragment)
        vec2 starUV = (vUv - 0.5) * vec2(8.0, 4.0);
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
        starCol *= smoothstep(0.30, 0.55, t) * nightAmt;
        night += starCol * 1.4;
        vec3 c = day*dayAmt + twi*twiAmt + night*nightAmt;
        gl_FragColor = vec4(c, 1.0);
    }
