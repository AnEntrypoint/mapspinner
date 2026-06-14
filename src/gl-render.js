// WebGL2 terrain RENDER layer: compiles and executes src/shaders/terrain.glsl
// (spherical deformation VS + CLOD blend + lit FS) per frame. Per-quad deformation
// uniforms (screenQuadCorners C / verticals N / cornerNorms L / offset / camera /
// blending / localToWorld) are computed in JS. No WebGPU.

// ---- minimal column-major mat4 helpers (no gl-matrix dep) ----------------------------
const M4 = {
  mul(a, b) { // a*b, column-major (OpenGL convention)
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
      let s = 0; for (let k = 0; k < 4; k++) s += a[k*4+r] * b[c*4+k];
      o[c*4+r] = s;
    }
    return o;
  },
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy/2), nf = 1/(near-far);
    return new Float32Array([ f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0 ]);
  },
  // column-major translation matrix translate(t): maps p -> p + t
  translate(t) {
    return new Float32Array([ 1,0,0,0, 0,1,0,0, 0,0,1,0, t[0],t[1],t[2],1 ]);
  },
  // lookAt view matrix (world->camera), column-major
  lookAt(eye, center, up) {
    const z0=eye[0]-center[0], z1=eye[1]-center[1], z2=eye[2]-center[2];
    let zl=Math.hypot(z0,z1,z2); const zx=z0/zl, zy=z1/zl, zz=z2/zl;
    let x0=up[1]*zz-up[2]*zy, x1=up[2]*zx-up[0]*zz, x2=up[0]*zy-up[1]*zx;
    let xl=Math.hypot(x0,x1,x2);
    if (xl < 1e-4) {                 // up parallel to view dir (poles) -> pick another up
      const au=[0,0,1]; x0=au[1]*zz-au[2]*zy; x1=au[2]*zx-au[0]*zz; x2=au[0]*zy-au[1]*zx;
      xl=Math.hypot(x0,x1,x2);
      if (xl < 1e-4){ x0=zy*1-zz*0; x1=zz*0-zx*1; x2=zx*0-zy*0; xl=Math.hypot(x0,x1,x2); } // up=[1,0,0]
    }
    xl = xl || 1; x0/=xl; x1/=xl; x2/=xl;
    const y0=zy*x2-zz*x1, y1=zz*x0-zx*x2, y2=zx*x1-zy*x0;
    return new Float32Array([
      x0,y0,zx,0, x1,y1,zy,0, x2,y2,zz,0,
      -(x0*eye[0]+x1*eye[1]+x2*eye[2]), -(y0*eye[0]+y1*eye[1]+y2*eye[2]), -(zx*eye[0]+zy*eye[1]+zz*eye[2]), 1
    ]);
  },
};

export async function initMapspinnerRender(gl, opts = {}) {
  const R = opts.radius || 6360000.0;
  const TILE_W = opts.tileW || 25;         // mesh-coord tile width (was producer.TILE_W; producer gone)
  // GRID 24 -> 16 (FPS lever, measured browser-18: pxPerPoly median 2.4px@40km / 0.45px@8km at GRID 24
  // = SUB-PIXEL over-tessellation, only 40%/24% in the 4-50px band). GRID 16 cuts verts/quad 676->324
  // (-52%) and tris/quad 1152->512 (-55%), so the per-vertex 14-oct broadShapeM VS (browser-9: 95% of
  // the low-alt frame) runs on ~half the vertices. median scales ~24/16 -> ~3.6px, far closer to the
  // band; the fine relief is carried per-pixel by the FS dFdx normal, not the mesh tessellation.
  const GRID = opts.gridMeshSize || 11;    // mesh quads per edge. 16->11 (user 2026-06-14): FPS is TRIANGLE-THROUGHPUT bound, not broadShapeM ALU (octMax 12->3 left frame time flat; GRID is ~linear). GRID 8 was faster (-50%) but made BIOME CROSSOVER LINES JAGGED (climate varying interpolated across coarse triangles steps along edges) -- reverted to 11 (-37%). Proper fix to reclaim GRID 8 = per-pixel biome sampling in the FS. Override via ?grid=N.
  // Expose the LIVE mesh grid so screen-space-error diagnostics (planet.html __diag.pxPerPoly)
  // divide by the real polys/tile instead of a stale literal. Any future GRID change self-corrects
  // the metric (the 24->16 lever left pxPerPoly defaulting to 24 = 1.5x wrong band fraction).
  if (typeof window !== 'undefined') window.__glGrid = GRID;
  const BORDER = 2;
  const USABLE = TILE_W - 2*BORDER;        // 21 interior samples spanned by the mesh

  // HPF (hierarchical parameter field) continental texture -- set by the orchestrator via
  // setHpf(). The terrain VS samples it by world dir for the continental elevation bias
  // (seaBias), replacing the old hardcoded lobe. null until set (VS falls back to 0 bias).
  let _hpfTex = null, _hpfTex2 = null, _hpfRes = 0;   // _hpfTex RG16F(seaBias,elevAmp), _hpfTex2 RG8(temp,humid) -- W12 pack

  // ---- compile terrain.glsl ----
  let src = await (await fetch('./src/shaders/terrain.glsl')).text();
  // Analytic Bruneton-style atmosphere helpers, shared by terrain FS + sky pass.
  let atmoSrc = await (await fetch('./src/shaders/atmosphere.glsl')).text();

  // NON-BLOCKING COMPILE (user 2026-06-02: 'startup takes really long'). The terrain shader's
  // first (cold-cache) compile can take tens of seconds; querying COMPILE_STATUS/LINK_STATUS
  // BLOCKS the main thread until the driver finishes -> the page freezes for the whole compile.
  // KHR_parallel_shader_compile lets the driver compile on a worker thread; we poll the
  // non-blocking COMPLETION_STATUS_KHR and yield to the event loop between polls, so the page
  // stays responsive (and can show a loading state) during a cold compile instead of freezing.
  const _parExt = gl.getExtension('KHR_parallel_shader_compile');
  const COMPLETION_STATUS_KHR = 0x91B1;
  // Await a program's link completion without blocking the main thread. With the parallel ext we
  // poll COMPLETION_STATUS_KHR (true once the driver is done); without it we fall back to one
  // yield then the (blocking) status read. Throws on compile/link failure, same as before.
  async function awaitProgramLink(p, vs, fs, label){
    if (_parExt) {
      // poll until the driver reports completion, yielding each tick. NOT rAF when the tab is hidden
      // (2026-06-12: background tabs throttle rAF to ~1/min, so a backgrounded compile sat at 'init'
      // for 10+ minutes -- the recurring stuck-at-init mechanism); setTimeout keeps polling at ~8ms.
      const yield_ = () => new Promise(res => (typeof requestAnimationFrame !== 'undefined'
        && typeof document !== 'undefined' && !document.hidden
        ? requestAnimationFrame(() => res()) : setTimeout(res, 8)));
      while (!gl.getProgramParameter(p, COMPLETION_STATUS_KHR)) { await yield_(); }
    }
    // now the status reads return immediately (compile/link already finished)
    if (vs && !gl.getShaderParameter(vs, gl.COMPILE_STATUS)) throw new Error(label+' vs: '+gl.getShaderInfoLog(vs));
    if (fs && !gl.getShaderParameter(fs, gl.COMPILE_STATUS)) throw new Error(label+' fs: '+gl.getShaderInfoLog(fs));
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(label+' link: '+gl.getProgramInfoLog(p));
  }
  // PRECISION: global default HIGHP float. The mediump default (a speculative mobile-ALU lever) kept
  // causing recurring UV SCRAMBLES -- any world-scale noise UV (normalize(worldPos)*freq, freq up to
  // ~9000) whose snoise3 arg was evaluated in mediump (fp16 mantissa ~2048) lost lattice precision and
  // scrambled at close range, and chasing every per-site highp island kept missing sites (multiple
  // commits: f8550b2 et al). HIGHP-DEFAULT eliminates the entire class in one line (P2 simplicity +
  // P8 make-misuse-impossible: a mediump world-scale UV can no longer be reintroduced by omission).
  // Float WIDTH is not our measured frontier (octave count + LOD vertex count are), so the ALU cost is
  // acceptable; correctness + simplicity win over a micro-optimization that keeps breaking. The explicit
  // highp islands left in the shader are now redundant-but-harmless. int + sampler2DArray stay highp.
  const hdr = '#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2DArray;\n';
  // Build (or rebuild) the terrain program from the current src/atmoSrc. Factored so the
  // shader can be HOT-RELOADED in place (recompile()) without a page reload -- the biggest
  // single cut to the shader-edit debug loop. On compile/link failure it throws WITHOUT
  // disturbing the live program, so a bad edit is reported inline and the old shader keeps
  // running (no broken page).
  // Kick off compile+link WITHOUT reading status (non-blocking with KHR_parallel_shader_compile).
  // Returns {p, vs, fs}; the caller awaits awaitProgramLink() to validate once the driver is done.
  // fsDefs lets the caller add FS-only #defines (e.g. _DEBUGVIEW_ for the lazy debug program that
  // carries the diagnostic displayModes). The render program passes '' so the diagnostic blocks are
  // #ifdef'd OUT (the 7132-char / 25% cold-compile cut, browser-1590); the debug program passes
  // ' _DEBUGVIEW_' to compile them in. The VS is identical for both (no debug branches in the VS).
  function buildTerrainProgram(terrainSrc, atmo, fsDefs){
    fsDefs = fsDefs || '';   // space-separated extra FS defines, e.g. '_DEBUGVIEW_'
    function shader(type, def){ const s=gl.createShader(type);
      // Inject atmosphere.glsl into the FRAGMENT stage only (it's pure functions; the VS
      // doesn't need it). It must appear before terrain.glsl's FS uses the helpers.
      const body = (type===gl.FRAGMENT_SHADER) ? (atmo+'\n'+terrainSrc) : terrainSrc;
      // Each token gets its OWN `#define` line -- a single `#define _FRAGMENT_ _DEBUGVIEW_` would make
      // _FRAGMENT_ a macro that EXPANDS to _DEBUGVIEW_ (and never DEFINE _DEBUGVIEW_), so the debug
      // blocks stayed #ifdef'd out (witnessed browser-1609: debugFS==renderFS). Split into lines.
      const tokens = [def].concat(
        (type===gl.FRAGMENT_SHADER && fsDefs) ? fsDefs.trim().split(/\s+/) : []);
      const defLines = tokens.map(t => '#define '+t+'\n').join('');
      gl.shaderSource(s, hdr+defLines+body); gl.compileShader(s); return s; }
    const vs = shader(gl.VERTEX_SHADER,'_VERTEX_'), fs = shader(gl.FRAGMENT_SHADER,'_FRAGMENT_');
    const p = gl.createProgram();
    gl.attachShader(p, vs); gl.attachShader(p, fs);
    gl.bindAttribLocation(p, 0, 'vertex');
    gl.linkProgram(p);                              // kicks off the (parallel) link; do NOT read status here
    return { p, vs, fs };
  }
  // COLD COMPILE: only the render program is built on the cold startup path now. The collision PROBE
  // program is LAZY (ensureProbe, built on first sampleGroundM) and the DEBUG program is lazy
  // (ensureDebug) -- both off the cold path. KHR_parallel_shader_compile keeps the render link
  // non-blocking so the page shows a loading state instead of freezing.
  let _b = buildTerrainProgram(src, atmoSrc);
  // LAZY PROBE (build-time pivot 2026-06-09): the collision-height probe program is NO LONGER built on
  // the cold startup path. Measured: the probe (composeHeight + 5 carves FS) was a co-equal cold-compile
  // pole, but sampleGroundM only runs on free-fly collision NEAR GROUND -- never at startup. So it is now
  // built on first sampleGroundM() call (ensureProbe, mirroring the lazy debug program). Removes the probe
  // VS+FS from the cold compile with ZERO functionality loss (collision still GPU-exact, just compiled the
  // first time the user needs it). sampleGroundM returns null until the first build finishes (caller falls
  // back to no-collision, same as the long-standing probe-unavailable path).
  await awaitProgramLink(_b.p, _b.vs, _b.fs, 'terrain');   // non-blocking poll, then validate (render only now)
  let prog = _b.p;
  // LAZY DEBUG PROGRAM: the diagnostic displayModes (1,5,6,7,8,9,10,11,12) live behind _DEBUGVIEW_,
  // compiled into this SEPARATE program only when the user first selects such a mode -- it is NEVER
  // on the cold startup path (the render program above excludes them). Built on demand by ensureDebug();
  // null until then. Its own uniform-location cache (_dbgUloc) since locations are per-program.
  let debugProg = null, _dbgBuilding = null;
  const _dbgUloc = new Map();
  // The diagnostic-only modes that REQUIRE the debug program. Modes 0 (lit), 2 (albedo), 4 (biome
  // ramp) render correctly in the hot program, so they never trigger a debug-program build.
  const DEBUG_MODES = new Set([1,5,6,7,8,9,10,11,12]);
  function ensureDebug(){
    if (debugProg || _dbgBuilding) return;          // already built or in-flight
    // VISIBLE STATE (2026-06-12 'total clarity' tooling): a failed/slow debug compile used to fall
    // back to the lit view FOREVER with no signal (witnessed: displayMode 11 silently rendered lit;
    // a whole diagnostic session trusted a view that never engaged). __debugProgState is the witness:
    // 'compiling' -> 'ready' | 'failed: <log>'; planet.html shows it in the HUD while a debug mode
    // is requested but not yet served.
    if (typeof window !== 'undefined') window.__debugProgState = 'compiling';
    _dbgBuilding = (async () => {
      try {
        const nb = buildTerrainProgram(src, atmoSrc, ' _DEBUGVIEW_');
        await awaitProgramLink(nb.p, nb.vs, nb.fs, 'debug');
        debugProg = nb.p; _dbgUloc.clear();
        if (typeof window !== 'undefined') window.__debugProgState = 'ready';
      } catch(e){ try { if(typeof window!=='undefined') { window.__debugProgErr = String(e.message||e); window.__debugProgState = 'failed: ' + String(e.message||e).slice(0,120); } } catch(_){} }
      finally { _dbgBuilding = null; }
    })();
  }
  // ACTIVE PROGRAM indirection: U() resolves locations against whichever program is bound this frame
  // (render prog by default; the debug prog while a diagnostic displayMode is active). Each program
  // keeps its own location cache. _activeProg/_activeUloc are swapped in render() per frame.
  let _activeProg = null, _activeUloc = null;
  function setActiveProgram(p, cache){ _activeProg = p; _activeUloc = cache; }
  // MEMOIZE uniform locations: U() was calling gl.getUniformLocation EVERY time, and the
  // per-quad path (setQuadUniforms + 3x setTileCoords) hit it ~15x per quad per frame ->
  // ~3000 synchronous driver round-trips/frame at 200 quads = the ~4fps stall. Cache by
  // name; getUniformLocation is then called once per name. Cleared on recompile().
  const _uloc = new Map();
  const U = n => { const cache = _activeUloc || _uloc; const p = _activeProg || prog;
    let l = cache.get(n); if (l === undefined) { l = gl.getUniformLocation(p, n); cache.set(n, l); } return l; };
  // PROBE uniform-location cache (ESE 2026-06-10): sampleGroundM ran ~18 synchronous
  // gl.getUniformLocation(probeProg,...) per call (8 inline + ~10 via setComposeHeightUniforms),
  // hit once/frame on the near-ground collision path = ~18 driver round-trips/frame where it hurts
  // most. Mirror _uloc: memoize per name, cleared when the probe program is (re)built.
  let _probeUloc = new Map();
  const PU = n => { let l = _probeUloc.get(n); if (l === undefined) { l = gl.getUniformLocation(probeProg, n); _probeUloc.set(n, l); } return l; };
  // HOT-RELOAD: re-fetch both shader files (cache-busted), rebuild the terrain program,
  // and swap it in atomically. Returns {ok:true} or {ok:false, error} -- never leaves the
  // renderer in a broken state (a failed build throws before `prog` is reassigned).
  async function recompile(){
    try {
      const ns = await (await fetch('./src/shaders/terrain.glsl?t='+(performance.now()|0))).text();
      const na = await (await fetch('./src/shaders/atmosphere.glsl?t='+(performance.now()|0))).text();
      const nb = buildTerrainProgram(ns, na);
      await awaitProgramLink(nb.p, nb.vs, nb.fs, 'terrain');   // throws on compile/link error
      const newProg = nb.p;
      const old = prog; prog = newProg; src = ns; atmoSrc = na; _uloc.clear();
      gl.deleteProgram(old);
      // invalidate the lazy debug program so it rebuilds from the new source on the next debug-mode frame.
      if (debugProg) { gl.deleteProgram(debugProg); debugProg = null; _dbgUloc.clear(); }
      // invalidate the lazy probe program too (perf sweep 2026-06-11): it was leaked AND kept running
      // the OLD shader source after a hot-reload -- collision silently diverged from the new geometry.
      if (probeProg) { gl.deleteProgram(probeProg); probeProg = null; _probeUloc.clear(); }
      return { ok: true };
    } catch (e) { return { ok: false, error: String(e.message || e) }; }
  }

  // ---- HEIGHT PROBE program (collision): render the EXACT terrain height for ONE world dir
  // to a 1x1 R32F target, then readPixels it. The free-fly collision floor reads this so it can
  // never diverge from the rendered surface (user-chosen GPU readback, not a CPU mirror). The
  // probe FS (#define _PROBE_) reuses terrain.glsl's hpfSample + broadShapeM; the VS emits one
  // point at clip (0,0). Tiny 4-byte readback per call (collision once/frame).
  let probeProg = null, probeFbo = null, probeTex = null, _probeBuilding = null;
  // ensureProbe(): build the collision-height probe program + its 1x1 R32F FBO on first need (lazy).
  // Idempotent + in-flight-guarded (mirrors ensureDebug). Off the cold startup path.
  function ensureProbe(){
    if (probeProg || _probeBuilding) return;
    _probeBuilding = (async () => {
      try {
        const pvs = hdr + 'void main(){ gl_Position = vec4(0.0,0.0,0.0,1.0); gl_PointSize = 1.0; }';
        const pfs = hdr + atmoSrc + '\n#define _PROBE_\n' + src;
        const pv = gl.createShader(gl.VERTEX_SHADER); gl.shaderSource(pv, pvs); gl.compileShader(pv);
        const pf = gl.createShader(gl.FRAGMENT_SHADER); gl.shaderSource(pf, pfs); gl.compileShader(pf);
        const pp = gl.createProgram(); gl.attachShader(pp, pv); gl.attachShader(pp, pf); gl.linkProgram(pp);
        await awaitProgramLink(pp, pv, pf, 'probe');
        const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, 1, 1);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        _probeUloc.clear();   // stale locations from any prior probe program are invalid for the new one
        probeTex = tex; probeFbo = fbo; probeProg = pp;   // assign LAST so a half-built probe is never used
      } catch(e){ probeProg = null; try { if(typeof window!=='undefined') window.__probeErr = String(e.message||e); } catch(_){} }
      finally { _probeBuilding = null; }
    })();
  }
  const probeVao = gl.createVertexArray();
  // sampleGroundM(dir): rendered terrain height (metres) at world direction dir. Returns null if
  // the probe is unavailable (caller falls back). One 4-byte readPixels.
  function sampleGroundM(dir) {
    if (!probeProg) { ensureProbe(); return null; }   // lazy: kick off the build on first need, fall back to null until ready
    const pl = Math.hypot(dir[0],dir[1],dir[2])||1;
    gl.bindFramebuffer(gl.FRAMEBUFFER, probeFbo);
    gl.viewport(0,0,1,1);
    gl.useProgram(probeProg);
    gl.bindVertexArray(probeVao);
    if (_hpfTex) { gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D_ARRAY, _hpfTex); gl.uniform1i(PU('hpfPool'),3); }
    if (_hpfTex2) { gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY, _hpfTex2); gl.uniform1i(PU('hpfPool2'),5); }
    gl.uniform1i(PU('hasHpf'), _hpfTex?1:0);
    // SAME shape-control + HPF-sampler congruence as render() (setComposeHeightUniforms): the probe runs
    // composeHeight for sampleGroundM (collision/camera height) so collision matches the rendered surface.
    // If uHiFreqCut/vtxDetail were unset (0.0) here the probe's height would omit all fine relief and
    // diverge from the rendered geometry = the camera stops short of the visible surface. Match render().
    setComposeHeightUniforms(PU);
    gl.uniform3f(PU('probeDir'), dir[0]/pl, dir[1]/pl, dir[2]/pl);
    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.POINTS, 0, 1);
    const out = new Float32Array(4);
    gl.readPixels(0,0,1,1, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
    return out[0];
  }

  // ===== THC HEIGHT-CACHE BAKE (2026-06-14, NON-DESTRUCTIVE) =====
  // A separate program renders composeHeight for one tile into an R32F grid (a fullscreen tri; each
  // fragment = one tile parametric texel). Used FIRST as a readback witness (bake vs procedural
  // sampleGroundM) to prove the bake matches the geometry; the pool/LRU + the VS-sample switch are
  // later DAG nodes. _faceFrames mirror terrain.glsl faceFrame() columns (column-major mat3).
  const THC_BAKE_RES = 130;
  const _faceFrames = [
    [0,0,-1, 0,1,0, 1,0,0], [0,0,1, 0,1,0, -1,0,0],
    [1,0,0, 0,0,-1, 0,1,0], [1,0,0, 0,0,1, 0,-1,0],
    [1,0,0, 0,1,0, 0,0,1], [-1,0,0, 0,1,0, 0,0,-1],
  ];
  let bakeProg=null, bakeTex=null, bakeFbo=null, _bakeBuilding=null; const _bakeUloc=new Map();
  const BU = n => { let l=_bakeUloc.get(n); if(l===undefined){ l=gl.getUniformLocation(bakeProg,n); _bakeUloc.set(n,l);} return l; };
  function ensureBake(){
    if (bakeProg || _bakeBuilding) return;
    _bakeBuilding = (async () => {
      try {
        const bvs = hdr + 'void main(){ vec2 p=vec2((gl_VertexID==1)?3.0:-1.0,(gl_VertexID==2)?3.0:-1.0); gl_Position=vec4(p,0.0,1.0); }';
        const bfs = hdr + '\n#define _HEIGHTBAKE_\n' + src;
        const bv=gl.createShader(gl.VERTEX_SHADER); gl.shaderSource(bv,bvs); gl.compileShader(bv);
        const bf=gl.createShader(gl.FRAGMENT_SHADER); gl.shaderSource(bf,bfs); gl.compileShader(bf);
        const bp=gl.createProgram(); gl.attachShader(bp,bv); gl.attachShader(bp,bf); gl.linkProgram(bp);
        await awaitProgramLink(bp, bv, bf, 'bake');
        const tex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, THC_BAKE_RES, THC_BAKE_RES);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        const fbo=gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        _bakeUloc.clear(); bakeTex=tex; bakeFbo=fbo; bakeProg=bp;
      } catch(e){ bakeProg=null; try{ if(typeof window!=='undefined') window.__bakeErr=String(e.message||e); }catch(_){} }
      finally { _bakeBuilding=null; }
    })();
  }
  const bakeVao = gl.createVertexArray();
  // bake ONE tile into bakeTex + read it back (Float32Array of THC_BAKE_RES^2 heights). Returns null
  // until the program is built (lazy). NON-DESTRUCTIVE: does not touch the live render path.
  function bakeTileReadback(face, ox, oy, l, level){
    if (!bakeProg){ ensureBake(); return null; }
    gl.bindFramebuffer(gl.FRAMEBUFFER, bakeFbo);
    gl.viewport(0,0,THC_BAKE_RES,THC_BAKE_RES);
    gl.useProgram(bakeProg);
    gl.bindVertexArray(bakeVao);
    if (_hpfTex){ gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D_ARRAY,_hpfTex); gl.uniform1i(BU('hpfPool'),3); }
    if (_hpfTex2){ gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY,_hpfTex2); gl.uniform1i(BU('hpfPool2'),5); }
    gl.uniform1i(BU('hasHpf'), _hpfTex?1:0);
    setComposeHeightUniforms(BU);
    gl.uniform1f(BU('defRadius'), R);
    gl.uniformMatrix3fv(BU('uBakeFrame'), false, new Float32Array(_faceFrames[face|0]));
    gl.uniform4f(BU('uBakeOffset'), ox, oy, l, level);
    gl.uniform1f(BU('uBakeRes'), THC_BAKE_RES);
    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const buf = new Float32Array(THC_BAKE_RES*THC_BAKE_RES*4);
    gl.readPixels(0,0,THC_BAKE_RES,THC_BAKE_RES, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
    const out = new Float32Array(THC_BAKE_RES*THC_BAKE_RES);
    for (let i=0;i<out.length;i++) out[i]=buf[i*4];
    let dbg=null; try{ dbg={ offLoc: BU('uBakeOffset')!=null, resLoc: BU('uBakeRes')!=null, frameLoc: BU('uBakeFrame')!=null,
      offRead: BU('uBakeOffset')?Array.from(gl.getUniform(bakeProg, BU('uBakeOffset'))):null,
      resRead: BU('uBakeRes')?gl.getUniform(bakeProg, BU('uBakeRes')):null }; }catch(e){ dbg={err:String(e)}; }
    return { heights: out, res: THC_BAKE_RES, dbg };
  }
  if (typeof window !== 'undefined') { window.__thcBakeReadback = bakeTileReadback; window.__thcEnsureBake = ensureBake; }

  // ===== THC HEIGHT POOL + LRU (the VS-sample consumer; the FPS win) =====
  // The VS samples a baked per-tile height (O(1) texture fetch) instead of composeHeight 5x/vertex,
  // when window.__thc is on. A 2D-array pool holds one BAKE_RES^2 R32F layer per live tile; a leaf
  // gets a layer (baked once) on first sight, LRU-evicted when the pool is full. Default OFF -> the
  // live render is unchanged (composeHeight), so this is safe to ship behind the toggle.
  const THC_POOL_LAYERS = 512;
  let heightPool=null, poolFbo=null;
  const _tcMap = new Map();                                   // tileKey -> layer
  const _tcLayerKey = new Array(THC_POOL_LAYERS).fill(null);  // layer -> tileKey (evict bookkeeping)
  const _tcUsed = new Int32Array(THC_POOL_LAYERS);            // layer -> last-used frame
  let _tcFrame = 0, _tcNextFree = 0, _tcBakesThisFrame = 0;
  // BAKE-ON-EDIT: terraform/HPF changes make every baked layer stale -> drop the whole map so each
  // visible tile re-bakes on next sight (synchronously, before its draw -> no black/stale frame).
  function invalidatePool(){ _tcMap.clear(); _tcLayerKey.fill(null); _tcNextFree = 0; }
  function ensurePool(){
    if (heightPool) return;
    heightPool = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D_ARRAY, heightPool);
    gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 1, gl.R32F, THC_BAKE_RES, THC_BAKE_RES, THC_POOL_LAYERS);
    const lin = _halfFloatLinearOK ? gl.LINEAR : gl.NEAREST;   // R32F LINEAR needs OES_texture_float_linear; else VS does manual bilinear
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, lin); gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, lin);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    poolFbo = gl.createFramebuffer();
  }
  function bakeTileToLayer(face,ox,oy,l,level,layer){
    gl.bindFramebuffer(gl.FRAMEBUFFER, poolFbo);
    gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, heightPool, 0, layer);
    gl.viewport(0,0,THC_BAKE_RES,THC_BAKE_RES);
    gl.useProgram(bakeProg); gl.bindVertexArray(bakeVao);
    if (_hpfTex){ gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D_ARRAY,_hpfTex); gl.uniform1i(BU('hpfPool'),3); }
    if (_hpfTex2){ gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY,_hpfTex2); gl.uniform1i(BU('hpfPool2'),5); }
    gl.uniform1i(BU('hasHpf'), _hpfTex?1:0);
    setComposeHeightUniforms(BU);
    gl.uniform1f(BU('defRadius'), R);
    gl.uniformMatrix3fv(BU('uBakeFrame'), false, new Float32Array(_faceFrames[face|0]));
    gl.uniform4f(BU('uBakeOffset'), ox, oy, l, level);
    gl.uniform1f(BU('uBakeRes'), THC_BAKE_RES);
    gl.disable(gl.DEPTH_TEST);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    _tcBakesThisFrame++;
  }
  // pool layer for a tile, baked on first sight; LRU-evicts when full. Returns -1 if not yet bakeable.
  function ensureTileLayer(face,ox,oy,l,level){
    const key = face+':'+ox+':'+oy+':'+l;
    let layer = _tcMap.get(key);
    if (layer === undefined){
      if (_tcNextFree < THC_POOL_LAYERS){ layer = _tcNextFree++; }
      else { let lru=0, lruF=_tcUsed[0]; for(let k=1;k<THC_POOL_LAYERS;k++) if(_tcUsed[k]<lruF){lruF=_tcUsed[k];lru=k;} layer=lru; const old=_tcLayerKey[lru]; if(old!=null) _tcMap.delete(old); }
      _tcMap.set(key, layer); _tcLayerKey[layer]=key;
      bakeTileToLayer(face,ox,oy,l,level,layer);
    }
    _tcUsed[layer]=_tcFrame;
    return layer;
  }
  // THC active = toggle on AND both programs/pool ready. Builds them lazily; returns false until ready
  // so the first frames fall back to composeHeight (uThc=0) with no garbage.
  let _tcInvSeen = 0;
  function thcActive(){
    if (typeof window==='undefined' || !window.__thc) return false;
    if (!bakeProg){ ensureBake(); return false; }
    ensurePool();
    // live re-bake hook: window.__thcInvalidate() bumps __thcInval; any composeHeight-shaping edit
    // (e.g. __gen biome/relief dials) should call it so the baked pool refreshes.
    const inv = (window.__thcInval|0);
    if (inv !== _tcInvSeen){ _tcInvSeen = inv; invalidatePool(); }
    return !!heightPool;
  }
  if (typeof window !== 'undefined') window.__thcInvalidate = () => { window.__thcInval = (window.__thcInval|0) + 1; };

  // FLOAT-LINEAR FORMAT PROBE (NOT a quality tier): OES_texture_float_linear lets the HPF atlas pools
  // filter LINEAR in hardware -> hpfSample collapses to one texture() call. 0 = manual 4-tap fallback.
  const _halfFloatLinearOK = !!gl.getExtension('OES_texture_float_linear') || !!gl.getExtension('OES_texture_half_float_linear');
  // Diagnostics-only readout (NOT a branch): exposes the float-linear probe outcome for a witness/CLI.
  try { if (typeof window !== 'undefined') window.__terrainConfig = { floatLinearOK: _halfFloatLinearOK }; } catch(_){}

  // ONE SOURCE OF TRUTH for composeHeight's shape-control + HPF-sampler uniforms: every program that runs
  // composeHeight (render, _PROBE_) calls this with its own uniform-locator so they CANNOT diverge.
  function setComposeHeightUniforms(loc) {
    const g = (n,d)=> (typeof window!=='undefined' && window['__'+n]!=null) ? +window['__'+n] : d;
    gl.uniform1f(loc('uHiFreqCut'),     g('hiFreqCut', 0.25));   // DECISIVE: ungated *= at terrain.glsl fine octaves; 0.5->0.25 (2026-06-10 'blotchy': the 4x fine band read as leopard dapple at altitude -- live-isolated, hiFreqCut=0 removed it entirely)
    gl.uniform1f(loc('uDetailOverlay'), g('detailOverlay', 6.0));  // perlin-everywhere ELEVATION term in composeHeight -- probe must match the VS or collision diverges
    gl.uniform1f(loc('vtxDetail'),      g('vtxDetail', 1.0));    // DECISIVE: vtxDisplace strength (early-return on 0)
    gl.uniform1f(loc('canyonDepthMul'), g('canyonDepth', 1.0));
    gl.uniform1f(loc('uVsCheap'),       (typeof window!=='undefined' && window.__vsCheap) ? 1.0 : 0.0);   // VS carve-cost profiling A/B
    gl.uniform1f(loc('uBeachShelfM'),   g('beachShelf', 0.0));   // land coastal shelf (geometry); probe MUST match render
    gl.uniform1f(loc('uLandBias'),      g('landBias', 0.0));       // +650m hypsometry bias = ~+30% land:sea (measured: landFrac 0.041 -> 0.054 over a 700-dir sphere grid, user 2026-06-14). window.__landBias dials it live.
    gl.uniform1f(loc('cliffAmt'),       g('cliffAmt', 1.0));
    gl.uniform1i(loc('uFloatLinearOK'), _halfFloatLinearOK ? 1 : 0);
    // FXC unroll-defeat (2026-06-12 AMD d3d11 fix): runtime octave bound for broadShapeM; the shader
    // guards uOctMax<=0 -> 12, so this set is belt-and-braces. Live dial: window.__octMax.
    gl.uniform1i(loc('uOctMax'),        (typeof window!=='undefined' && window.__octMax!=null) ? (window.__octMax|0) : 12);
    gl.uniform1i(loc('uInciseRidgeOcts'), (typeof window!=='undefined' && window.__inciseRidgeOcts!=null) ? (window.__inciseRidgeOcts|0) : 4);
    gl.uniform1i(loc('uBroadLowOcts'),    (typeof window!=='undefined' && window.__broadLowOcts!=null) ? (window.__broadLowOcts|0) : 8);
    gl.uniform1i(loc('uPeakOcts'),        (typeof window!=='undefined' && window.__peakOcts!=null) ? (window.__peakOcts|0) : 3);
    gl.uniform1i(loc('uVtxBaseOcts'),     (typeof window!=='undefined' && window.__vtxBaseOcts!=null) ? (window.__vtxBaseOcts|0) : 6);
    gl.uniform1i(loc('uVtxErodeOcts'),    (typeof window!=='undefined' && window.__vtxErodeOcts!=null) ? (window.__vtxErodeOcts|0) : 4);
    gl.uniform1i(loc('uDetailFbmOcts'),   (typeof window!=='undefined' && window.__detailFbmOcts!=null) ? (window.__detailFbmOcts|0) : 3);
    gl.uniform1i(loc('uFSDetailOcts'),    (typeof window!=='undefined' && window.__fsDetailOcts!=null) ? (window.__fsDetailOcts|0) : 3);
    // FXC fold-defeat (2026-06-12, the rock-on-flat patches): the lit-normal FD step is uniform-fed
    // so d3d11/FXC cannot constant-fold the 150/R offset. Live dial: window.__nrmStepM.
    gl.uniform1f(loc('uNrmStepM'),      g('nrmStepM', 300.0));
    gl.uniform1f(loc('uGrid'),          GRID);
    gl.uniform1f(loc('uHpfInset'),      (typeof window!=='undefined' && window.__hpfInset === false) ? 0.0 : 1.0);   // SEAM FIX: inset sampler is the permanent default (matches bakeFace fu=x/(RES-1)); window.__hpfInset===false rolls back
    // ANCHOR-STEP A/B TOGGLES (per-area stairstep, wrxo0rr7a). Default 0 = current; set window.__<name>=1
    // to widen that anchor-keyed band. Set HERE so BOTH render and the _PROBE_ collision see them (parity).
    gl.uniform1f(loc('uMtnBandWide'),   g('mtnBandWide', 0.0));
    gl.uniform1f(loc('uClimateRelief'), g('climateRelief', 0.0));
    gl.uniform1f(loc('uIsleWide'),      g('isleWide', 0.0));
    gl.uniform1f(loc('uCarveWide'),     g('carveWide', 0.0));
  }


  // ---- SURFACE PHOTO-TEXTURES (user 2026-06-10): grass/rock/sand/snow color + displacement JPGs
  // from /textures, packed into two mipped sampler2DArrays. Normals are SOBEL-DERIVED from the
  // displacement at load (3x3, WRAPPED edges -- the textures tile, so the kernel must wrap or the
  // tile border gets a seam line). uSurfAlb = sRGB color (RGB) + displacement (A, linear alpha);
  // uSurfNrm = tangent normal xy 0.5-biased (RG) + displacement (B). Loaded ASYNC off the cold
  // startup path; uHasSurfTex stays 0 (procedural-only) until the upload lands.
  let _surfAlb = null, _surfNrm = null, _surfMeanL = [0.2, 0.2, 0.2, 0.5];
  // 8-bit -> linear LUT (perf sweep 2026-06-11): gamma-2.2 on an 8-bit input has exactly 256 values;
  // the per-pixel Math.pow de-shade/mean passes were ~30M transcendental calls = ~0.5s+ of main-thread
  // long tasks per load. The LUT is bit-identical for every 8-bit input.
  const LIN8 = new Float32Array(256);
  for (let v = 0; v < 256; v++) LIN8[v] = Math.pow(v / 255, 2.2);
  async function loadSurfaceTextures() {
    const MATS = ['grass', 'rock', 'sand', 'snow'];   // layer order: matches terrain.glsl splat
    const SZ = 1024;
    const img = (u) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = () => rej(new Error('load ' + u)); i.src = u; });
    const cv = document.createElement('canvas'); cv.width = SZ; cv.height = SZ;
    const cx = cv.getContext('2d', { willReadFrequently: true });
    const px = (im) => { cx.drawImage(im, 0, 0, SZ, SZ); return cx.getImageData(0, 0, SZ, SZ).data; };
    const albAll = new Uint8Array(SZ * SZ * 4 * MATS.length);
    const nrmAll = new Uint8Array(SZ * SZ * 4 * MATS.length);
    for (let m = 0; m < MATS.length; m++) {
      const [ci, di] = await Promise.all([img('./textures/' + MATS[m] + '-color.jpg'), img('./textures/' + MATS[m] + '-displacement.jpg')]);
      const c = px(ci), d = px(di);
      // DE-SHADE (user 2026-06-11 'flat, unangled bowls of rock'): the photos carry baked large-scale
      // shading (shadowed depressions), which at a 2.4km tile pastes bowl-shaped shadows onto geometry
      // with no matching shape. Divide each pixel by a wrapped-bilinear 32x32 blur of the photo's own
      // linear luminance (renormalized to the photo mean) -- kills the bowl-scale light, keeps detail.
      // PER-CHANNEL (user 2026-06-11 'rocky patches on flat ground, no steep slopes': the grass
      // photo carries large grey-brown bare-dirt patches that differ in CHROMA, not just luminance;
      // at the 2.4km tile they become hundred-metre grey blotches that read as rock. Per-channel
      // division flattens large-scale COLOR blotches too; fine grain untouched.)
      { const G = 32, cell = SZ / G, grid = new Float32Array(G * G * 3), cnt = cell * cell;
        const lin = (v) => LIN8[v], delin = (v) => Math.pow(v, 1 / 2.2) * 255;
        for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) {
          const i = (y * SZ + x) * 4, g = (((y / cell) | 0) * G + ((x / cell) | 0)) * 3;
          grid[g] += LIN8[c[i]]; grid[g + 1] += LIN8[c[i + 1]]; grid[g + 2] += LIN8[c[i + 2]];
        }
        const gMean = [0, 0, 0];
        for (let g = 0; g < G * G; g++) for (let ch = 0; ch < 3; ch++) { grid[g * 3 + ch] /= cnt; gMean[ch] += grid[g * 3 + ch]; }
        for (let ch = 0; ch < 3; ch++) gMean[ch] /= G * G;
        for (let y = 0; y < SZ; y++) for (let x = 0; x < SZ; x++) {
          const gx = x / cell - 0.5, gy = y / cell - 0.5;
          const x0 = (Math.floor(gx) + G) % G, y0 = (Math.floor(gy) + G) % G;
          const x1 = (x0 + 1) % G, y1 = (y0 + 1) % G;
          const fx = gx - Math.floor(gx), fy = gy - Math.floor(gy);
          const i = (y * SZ + x) * 4;
          for (let ch = 0; ch < 3; ch++) {
            const blur = (grid[(y0 * G + x0) * 3 + ch] * (1 - fx) + grid[(y0 * G + x1) * 3 + ch] * fx) * (1 - fy)
                       + (grid[(y1 * G + x0) * 3 + ch] * (1 - fx) + grid[(y1 * G + x1) * 3 + ch] * fx) * fy;
            // pow 0.8: stronger than the old 0.65 luma-only pass -- the patches must GO; fine
            // (sub-64px) structure is untouched by construction.
            const s = Math.min(2.5, Math.max(0.4, Math.pow(gMean[ch] / Math.max(blur, 1e-4), 0.8)));
            // FINE-CONTRAST RESTORE (user 2026-06-11 'grass texture... not on grassy areas'): the
            // de-shade flattened large blotches but also left flats reading textureless once the
            // shade-match lands the average on the macro color. Stretch the remaining (fine-grain)
            // deviation around the channel mean x1.35 so the texture stays visible on flat ground.
            const v = lin(c[i + ch]) * s;
            c[i + ch] = Math.min(255, Math.max(0, Math.round(delin(Math.max(0, gMean[ch] + (v - gMean[ch]) * 1.35)))));
          }
        }
      }
      // yield between the de-shade and Sobel passes too (perf sweep 2026-06-11): one material was a
      // single contiguous main-thread block; splitting halves the worst long task.
      await new Promise(res => setTimeout(res, 0));
      const base = m * SZ * SZ * 4;
      const S = 2.2;   // gradient gain: 1024px tile reads as believable relief at the default repeat
      for (let y = 0; y < SZ; y++) {
        const ym = (y + SZ - 1) % SZ, yp = (y + 1) % SZ;
        for (let x = 0; x < SZ; x++) {
          const xm = (x + SZ - 1) % SZ, xp = (x + 1) % SZ;
          const i = y * SZ + x, o = base + i * 4;
          const r = (X, Y) => d[(Y * SZ + X) * 4];
          const gx = (r(xp, ym) + 2 * r(xp, y) + r(xp, yp) - r(xm, ym) - 2 * r(xm, y) - r(xm, yp)) / (8 * 255);
          const gy = (r(xm, yp) + 2 * r(x, yp) + r(xp, yp) - r(xm, ym) - 2 * r(x, ym) - r(xp, ym)) / (8 * 255);
          // COARSE octave (user 2026-06-11 'normals seem missing for textures'): the 1px Sobel is
          // pure fine detail and mips to flat at any distance; a +/-6px central diff carries the
          // mid-scale relief through the mip chain.
          const x6p = (x + 6) % SZ, x6m = (x + SZ - 6) % SZ, y6p = (y + 6) % SZ, y6m = (y + SZ - 6) % SZ;
          const gx2 = (r(x6p, y) - r(x6m, y)) / (2 * 255), gy2 = (r(x, y6p) - r(x, y6m)) / (2 * 255);
          // LARGE octave (user 2026-06-11, repeated displacement->normals ask -- the gap was SPECTRAL:
          // a km-scale displacement feature spans hundreds of px, so its per-texel gradient (~0.004)
          // never registered in the 1px/6px diffs = the BIG relief had zero normal response, and the
          // fine detail that did register mips away at flight altitude. A +/-48px diff at gain 8
          // makes the large features tilt the normal AND survive the mip chain.)
          const x48p = (x + 48) % SZ, x48m = (x + SZ - 48) % SZ, y48p = (y + 48) % SZ, y48m = (y + SZ - 48) % SZ;
          const gx3 = (r(x48p, y) - r(x48m, y)) / (2 * 255), gy3 = (r(x, y48p) - r(x, y48m)) / (2 * 255);
          // gain 8 -> 2 + total tilt CAP (user: 'rock blotches reappear, normal data gone again' --
          // gain 8 saturated the tangential normal (gradients ~0.3 on ~100px features x8 = full
          // sideways tilt) = the scramble class, self-inflicted). Cap |xy| at 0.9 so no texel ever
          // encodes a >42deg tilt regardless of octave stacking.
          let nx = -(gx * S + gx2 * 2.5 + gx3 * 2.0), ny = -(gy * S + gy2 * 2.5 + gy3 * 2.0);
          const tm = Math.hypot(nx, ny);
          if (tm > 0.9) { nx *= 0.9 / tm; ny *= 0.9 / tm; }
          const il = 1 / Math.hypot(nx, ny, 1);
          albAll[o] = c[i * 4]; albAll[o + 1] = c[i * 4 + 1]; albAll[o + 2] = c[i * 4 + 2]; albAll[o + 3] = d[i * 4];
          nrmAll[o] = Math.round((nx * il * 0.5 + 0.5) * 255);
          nrmAll[o + 1] = Math.round((ny * il * 0.5 + 0.5) * 255);
          nrmAll[o + 2] = d[i * 4]; nrmAll[o + 3] = 255;
        }
      }
      // yield between materials so the ~4x 1M-px Sobel never blocks a whole frame budget at once
      await new Promise(res => setTimeout(res, 0));
    }
    const aniso = gl.getExtension('EXT_texture_filter_anisotropic');
    function mkArray(data, internal) {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
      gl.texStorage3D(gl.TEXTURE_2D_ARRAY, 11, internal, SZ, SZ, MATS.length);   // 11 = full 1024 mip chain
      gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, 0, SZ, SZ, MATS.length, gl.RGBA, gl.UNSIGNED_BYTE, data);
      gl.generateMipmap(gl.TEXTURE_2D_ARRAY);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.REPEAT);
      if (aniso) gl.texParameterf(gl.TEXTURE_2D_ARRAY, aniso.TEXTURE_MAX_ANISOTROPY_EXT,
        Math.min(8, gl.getParameter(aniso.MAX_TEXTURE_MAX_ANISOTROPY_EXT)));
      return t;
    }
    // mean LINEAR color of the rock photo (layer 1): the far-field macro bcRock defaults to this so
    // the >20km rock shade matches the near-field photo rock (no color pop across the fade).
    { let r = 0, g = 0, b = 0; const base = 1 * SZ * SZ * 4, n = SZ * SZ;
      for (let i = 0; i < n; i++) { r += albAll[base + i * 4]; g += albAll[base + i * 4 + 1]; b += albAll[base + i * 4 + 2]; }
      const lin = (v) => Math.pow(v / n / 255, 2.2);
      if (typeof window !== 'undefined') window.__surfRockMean = [lin(r), lin(g), lin(b)];
    }
    // mean LINEAR luminance per layer (user 2026-06-11 'terrain gets darker' + 'dont see grass/snow
    // textures'): the shader shade-matches the photo by dividing out its LAYER-MEAN luminance (not
    // per-pixel, which cancelled all structure; not raw photo, which shifted the shade).
    { const meanL = [0, 0, 0, 0];
      for (let m = 0; m < MATS.length; m++) {
        let s = 0; const base = m * SZ * SZ * 4, n = SZ * SZ;
        for (let i = 0; i < n; i++) {
          s += 0.2126 * LIN8[albAll[base + i * 4]] + 0.7152 * LIN8[albAll[base + i * 4 + 1]] + 0.0722 * LIN8[albAll[base + i * 4 + 2]];
        }
        meanL[m] = s / n;
      }
      _surfMeanL = meanL;
      if (typeof window !== 'undefined') window.__surfMeanL = meanL;
    }
    _surfAlb = mkArray(albAll, gl.SRGB8_ALPHA8);   // sRGB decode in hardware (color); A (displacement) stays linear
    _surfNrm = mkArray(nrmAll, gl.RGBA8);          // normals/displacement are data, NOT color -> linear
    if (typeof window !== 'undefined') window.__surfTexReady = true;
  }
  if (typeof document !== 'undefined') {
    loadSurfaceTextures().catch(e => { try { window.__surfTexErr = String(e.message || e); } catch (_) {} });
  }

  // ---- fullscreen SKY pass program (atmospheric limb/halo behind the terrain) ----
  // VS emits a fullscreen triangle; FS reconstructs the world-space view ray from the
  // inverse view-projection and calls atm_skyRadiance. Drawn before terrain (depth
  // writes off) so terrain overdraws where the planet is, leaving sky on the limb.
  const skyVsSrc = hdr + `out vec2 vNdc;
    void main(){ vec2 p = vec2((gl_VertexID==1)?3.0:-1.0, (gl_VertexID==2)?3.0:-1.0);
      vNdc = p; gl_Position = vec4(p, 1.0, 1.0); }`;
  const skyFsSrc = hdr + atmoSrc + `
    in vec2 vNdc;
    layout(location=0) out vec4 fragColor;
    uniform mat3 camRot;        // world<-view rotation (columns = view basis in world)
    uniform vec2 projDiag;      // (proj[0][0], proj[1][1]) for NDC->view-ray
    uniform vec3 skyCamWorld;   // camera world pos (meters)
    uniform vec3 skySunDir;     // world sun dir (normalized)
    uniform float skyR;         // sphere radius (meters)
    uniform float uSkyFade;     // 1 at surface, 0 at 100km
    void main(){
      // Reconstruct the world-space view ray from NDC, like the WebGPU skyFs: undo the
      // projection (divide by the proj diagonal) to get a view-space dir, then rotate
      // into world with the camera basis. Robust (no near-far matrix inverse).
      vec3 dirView = normalize(vec3(vNdc.x/projDiag.x, vNdc.y/projDiag.y, -1.0));
      vec3 viewRay = normalize(camRot * dirView);
      vec3 camAtm = atmPos(skyCamWorld, skyR);
      vec3 t;
      vec3 radiance = atm_skyRadiance(camAtm, viewRay, skySunDir, t);

      // ---- Explicit limb/halo glow (guarantees a visible atmosphere ring from orbit).
      // The physical single-scatter limb is sub-pixel thin at orbital range, so we add
      // an analytic glow keyed on the ray's IMPACT PARAMETER b = perpendicular distance
      // of the view ray from the planet centre. b in [BOTTOM, ~BOTTOM+halo] -> bright
      // blue rim that fades outward; lit only on the sun-facing side, scaled by a soft
      // forward-scatter term. This is a deliberate visual augmentation of the analytic
      // single-scatter model (documented simplification).
      {
        float rc = length(camAtm);
        float muc = dot(camAtm, viewRay) / rc;
        float b = rc * sqrt(max(1.0 - muc*muc, 0.0)); // impact parameter (km)
        // Only for rays passing in FRONT of the planet (muc<0) and outside the surface.
        float halo = 0.0;
        if (muc < 0.0) {
          float t0 = (b - ATM_BOTTOM) / (ATM_TOP - ATM_BOTTOM);  // 0 at surface -> 1 at top
          // Inner rim brightest, fading to the shell top; zero below surface / above top.
          halo = smoothstep(0.0, 0.06, t0) * (1.0 - smoothstep(0.25, 1.6, t0));
        }
        // Daylight side weighting from the sun's relation to the limb point direction.
        vec3 limbDir = normalize(camAtm + viewRay * (-rc*muc)); // closest-approach dir
        // Day-side rim brightest; keep a small floor so the whole ring stays visible.
        float lit = 0.25 + 0.75 * smoothstep(-0.5, 0.6, dot(limbDir, skySunDir));
        vec3 haloColor = vec3(0.32, 0.55, 1.0);  // Rayleigh-blue rim
        radiance += haloColor * (halo * lit) * 0.03;
      }
      // Sun disc through the view transmittance.
      float cosVS = dot(viewRay, skySunDir);
      if (cosVS > cos(ATM_SUN_ANGULAR_RADIUS)) {
        radiance += t * ATM_SOLAR_IRRADIANCE * 6.0;   // sun disc
      }
      // The analytic single-scatter radiance is HDR with small magnitudes; lift then
      // ACES tonemap (matches the WebGPU sky pass family). EXPOSURE tuned so the limb
      // glow + daylit sky read as an atmosphere without blowing out.
      vec3 c = radiance * 105.0;   // 120->105: trim exposure so the bright daytime sky stops clipping to white
      vec3 mapped = clamp((c*(2.51*c+0.03))/(c*(2.43*c+0.59)+0.14), 0.0, 1.0); // ACES
      // DAY SKY TOO WHITE (user 2026-06-14): ACES desaturates the bright daylit sky toward white. Push
      // saturation back up so the day sky reads BLUE (sun disc stays white -- it is already near-neutral).
      float skyLum = dot(mapped, vec3(0.2126, 0.7152, 0.0722));
      mapped = clamp(mix(vec3(skyLum), mapped, 1.35), 0.0, 1.0);
      fragColor = vec4(pow(mapped, vec3(1.0/2.2)) * uSkyFade, 1.0);
    }`;
  function rawShader(type, source){ const s=gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS)) throw new Error('sky '+type+': '+gl.getShaderInfoLog(s)); return s; }
  const skyProg = gl.createProgram();
  gl.attachShader(skyProg, rawShader(gl.VERTEX_SHADER, skyVsSrc));
  gl.attachShader(skyProg, rawShader(gl.FRAGMENT_SHADER, skyFsSrc));
  gl.linkProgram(skyProg);
  if(!gl.getProgramParameter(skyProg, gl.LINK_STATUS)) throw new Error('sky link: '+gl.getProgramInfoLog(skyProg));
  const _usloc = new Map();
  const SU = n => { let l = _usloc.get(n); if (l === undefined) { l = gl.getUniformLocation(skyProg, n); _usloc.set(n, l); } return l; };
  const skyVao = gl.createVertexArray();

  // ---- mesh grid: OVERLAP-RING tessellation (replaces the old dropped-skirt curtain).
  // The mesh spans (GRID+2) cells in each axis: the INTERIOR GRID cells cover the tile's
  // usable region in param coord [0,1] exactly as before, plus ONE EXTRA RING of cells on
  // every side reaching param coord [-1/GRID, 1+1/GRID]. The extra ring extends the surface
  // one cell INTO the neighbor tile's territory (a real, continuous part of the elevation
  // field -- the atlas carries BORDER=2 texels of valid margin, so uv just outside [0,1]
  // samples genuine neighbor-edge texels, NOT garbage). At a coarse/fine LOD T-junction the
  // coarse tile's overlap ring covers the crack the skirt used to hide; at a same-LOD seam
  // both neighbors overlap into each other and overdraw a COPLANAR surface (both compute
  // near-identical world height from the continuous field, so no z-fight). The neighbor's
  // own interior overdraws the overlap, so the visible surface still ends at the true tile
  // boundary -- the outer ring is the "hidden last ring". vertex.z is always 0 (no skirt).
  const g2 = GRID+2;              // cells per axis (GRID interior + 1 ring each side)
  const n2 = g2+1;               // verts per axis
  const du = 1.0/GRID;           // param step = one interior cell
  // SKIRT not OVERLAP (fix-visible-overlap-ring): the outer ring used to extend one cell INTO the
  // neighbor [-du, 1+du] and rasterize a FLAT flap there -> a visible band at every patch edge (user:
  // 'ring polys visible'). Instead, CLAMP each outer-ring vertex's xy to the true interior edge [0,1]
  // and flag it (z=1) as a SKIRT: the VS drops it radially below the surface, forming a near-vertical
  // curtain at the tile boundary. The skirt fills any T-junction crack (so no seam, unlike deleting
  // the ring) but is hidden behind the surface (so no visible flat band, unlike the overlap).
  const vlist = []; // x, y (param coord clamped to [0,1]), z = skirt flag (0 surface, 1 skirt)
  for (let y=0;y<n2;y++) for (let x=0;x<n2;x++){
    const isRing = (x===0 || x===n2-1 || y===0 || y===n2-1);
    const px = Math.min(Math.max((x-1)*du, 0.0), 1.0);   // clamp ring xy onto the true edge
    const py = Math.min(Math.max((y-1)*du, 0.0), 1.0);
    vlist.push(px, py, isRing ? 1.0 : 0.0);
  }
  const idx = []; for (let y=0;y<g2;y++) for (let x=0;x<g2;x++){ const a=y*n2+x,b=a+1,c=a+n2,d=c+1; idx.push(a,c,b, b,c,d); }
  const verts = new Float32Array(vlist);
  const indices = new Uint32Array(idx);
  const vbo=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,vbo); gl.bufferData(gl.ARRAY_BUFFER,verts,gl.STATIC_DRAW);
  const ibo=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ibo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,indices,gl.STATIC_DRAW);
  const instBuf=gl.createBuffer();   // per-instance [ox,oy,l,level,face] (filled per frame in render())
  // DATA-CONTINUITY CACHE (2026-06-14): terrain + water get their OWN persistent instance buffers so
  // neither clobbers the other (the shared-buffer clobber forced a re-upload every frame and was the
  // root of the prior 'water drawn as terrain' regression). On a STATIC frame (same quads array object)
  // the instance data is identical -> skip the Float32Array build + bufferData + water dedup Set-loop
  // and just rebind+draw. Pure CPU/GC win (GPU is vertex-bound, the upload is off the critical path).
  const instBufWater=gl.createBuffer();
  let _instQuadsRef=null, _instWaterRef=null, _instWaterN=0, _lastThc=false;

  // per-face local->world (cube face -> sphere local frame). Column-major mat3 packed
  // into a Float32Array(9). Matches localToWorld3 convention:
  // col0 = U/rs, col1 = faceCenter, col2 = V/rs. rootQuadSize=2 -> face spans [-1,1].
  function localToWorld3(face) {
    // face axes (cube): for face 3 (+Z) U=+X, V=+Y, center=+Z. Generic table:
    const F = [
      {c:[ 1,0,0], u:[0,0,-1], v:[0,1,0]}, // +X
      {c:[-1,0,0], u:[0,0, 1], v:[0,1,0]}, // -X
      {c:[0, 1,0], u:[1,0,0], v:[0,0,-1]}, // +Y
      {c:[0,-1,0], u:[1,0,0], v:[0,0, 1]}, // -Y
      {c:[0,0, 1], u:[1,0,0], v:[0,1,0]},  // +Z
      {c:[0,0,-1], u:[-1,0,0],v:[0,1,0]},  // -Z
    ][face];
    // local plane coords (ox,oy) in [-1,1]; the VS builds P=(ox',oy',R) then normalizes
    // *defLocalToWorld* P. So localToWorld maps the local (x,y,z=R) basis to the face.
    // col0<-U, col1<-V, col2<-center (z axis = outward). Column-major 3x3.
    return new Float32Array([ F.u[0],F.u[1],F.u[2],  F.v[0],F.v[1],F.v[2],  F.c[0],F.c[1],F.c[2] ]);
  }

  // Compute & set the per-quad deformation uniforms (SphericalDeformation::setScreenUniforms).
  // quad = {level, tx, ty, ox, oy, l}; localCam = camera in this face's local plane coords.
  // (setQuadUniforms DELETED 2026-06-11 dead-code sweep: the single instanced draw replaced the
  // per-quad uniform path -- defOffset/defLocalToWorld are VS locals from iOffset/iFace now, and
  // the defViewProjRel/defOffset/defLocalToWorld uniforms no longer exist in the shader.)

  // textureTile coords for a tile resident at `layer` of the elev/normal atlas.
  // vertex.xy in [0,1] must sweep the tile INTERIOR (skip the BORDER): base = border/W,
  // span = (USABLE)/W. pixelScale carried in .z (unused by the simple textureTile).
  // ANCESTOR-FALLBACK: an optional `sub` = {ox,oy,scale} restricts sampling to a
  // sub-rectangle of the (ancestor) tile's USABLE interior. The mesh uv [0,1] then sweeps
  // only [ox,ox+scale] x [oy,oy+scale] of the usable region: base shifts by ox/oy of the
  // usable span, span shrinks by `scale`. With sub=null this is the full-tile interior.
  function setTileCoords(prefix, pool, layer, sub) {
    const fullSpan = (USABLE-1)/TILE_W;
    // EDGE-INSET fix lever (window.__elevEdgeInset, default 0.0): the texel-center offset of the
    // mesh-edge sample inside the BORDER. At 0.5 the edge vertex sampled texel (BORDER+0.5) so the
    // LINEAR filter blended the last INTERIOR texel with the adjacent SEAM BORDER texel -- a faint
    // per-tile-edge height kink that projected into a streak at grazing angle (the user's 'subtle
    // mip-bleed at medium distance', visible only when a tile edge aligned near-parallel to the
    // view ray). At 0.0 the edge vertex lands on the integer interior edge texel (= BORDER), the
    // shared-edge value both neighbours agree on, so there is NO border blend -> the static streak
    // floor is removed. VALIDATED (browser-401/402, this session): at the oblique heading where the
    // streak appeared, subtleFrac 0.0113->0.0027 + bestVertRun 6->2 (3-4x drop); harmless (0->0) at
    // clean oblique + closeup nadir (411/412/413/414), so it does not regress the ff5c8ba dim-line
    // fix. Default is now 0.0; the global stays as a live A/B lever.
    const inset = (typeof window.__elevEdgeInset === 'number') ? window.__elevEdgeInset : 0.5;
    const base0 = (BORDER + inset) / TILE_W;
    if (sub) {
      gl.uniform3f(U(prefix+'.tileCoords'), base0 + sub.ox*fullSpan, base0 + sub.oy*fullSpan, layer);
      gl.uniform3f(U(prefix+'.tileSize'),   fullSpan*sub.scale,     fullSpan*sub.scale,     1.0/TILE_W);
    } else {
      gl.uniform3f(U(prefix+'.tileCoords'), base0,    base0,    layer);
      gl.uniform3f(U(prefix+'.tileSize'),   fullSpan, fullSpan, 1.0/TILE_W);
    }
  }

  // SINGLE SOURCE OF TRUTH for the camera-relative clip matrix + near/far. Both render()
  // and the orchestrator's frustum cull use this so the cull can never disagree with the
  // draw (a divergence would cull on-screen quads or keep off-screen ones).
  function cullMatrix(cam) {
    const aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
    const camDist = Math.hypot(cam.eye[0], cam.eye[1], cam.eye[2]);
    const alt = Math.max(0.0, camDist - R);
    const altAboveTerrain = Math.max(0.001, alt - R * (cam.surfElev || 0));
    // FAR-PLANE HORIZON RADIUS = R - 500m (user 2026-06-14: 'nearby mountains disappear at water level;
    // adjust that level to 500m under water'). The far plane tracks the sea-level horizon = sqrt(camDist^2
    // - R^2), which at the deck (camDist~=R) collapses to a few hundred metres -> coastal mountains a km
    // out fall beyond the far plane and vanish. Dropping the horizon reference radius 500m below sea level
    // extends the horizon to tens of km at low altitude so near-shore relief stays in view (negligible
    // depth-precision cost: 500m vs R~6.37e6). Both the cull and the draw use this (single source).
    const RHORIZON = R - 150.0;   // far brought in 500->250 (user 2026-06-14 'bring far plane in a bit'): deck horizon ~80km->~56km = more z-precision; still clears coastal mountains
    // UNDERWATER FAR-PLANE FIX (user 2026-06-14 'at -214m visible, at -500m it disappears'): when the
    // camera is more than 500m below sea level, camDist < RHORIZON so the sea-level horizon is imaginary
    // (-> 0) and alt is negative; the old max(horizon, alt*8) then collapsed the far plane to ~0 and the
    // whole scene vanished past -500m deep (= the 'ocean looks shallow/empty' when exploring). Floor the
    // far reach to 60km when submerged so the seabed + the underwater view stay visible.
    const horizon = (camDist > RHORIZON) ? Math.sqrt(camDist*camDist - RHORIZON*RHORIZON) : 60000.0;
    // MATCH render()'s near exactly (2026-06-14 jank fix): the cull frustum must use the SAME near
    // as the draw frustum, else behind-limb/screen-AABB culling diverges from what is actually drawn
    // at the deck (cull near was max(*0.1,0.1) while render used the <2m 0.05 branch).
    const near = altAboveTerrain < 2.0 ? 0.5 : Math.max(altAboveTerrain * 0.1, 0.5);   // near nudged out 0.05->0.25 (user 2026-06-14 'improve on-ground'): more z-precision on the deck
    // FAR PLANE: horizon distance tracks the visible ground edge; blends toward camDist
    // above 500km for orbital views so the full planet is visible.
    const _fBlend = Math.min(1.0, Math.max(0.0, (alt - 500000.0) / 4500000.0));
    const farGround = Math.max(horizon, alt * 8.0);
    const far = farGround * (1.0 - _fBlend) + camDist * _fBlend;
    const proj = M4.perspective(cam.fovy||0.785, aspect, near, far);
    const eye = cam.eye;
    const viewRel = M4.lookAt([0,0,0], [cam.center[0]-eye[0], cam.center[1]-eye[1], cam.center[2]-eye[2]], cam.up||[0,1,0]);
    const viewProjRel = M4.mul(M4.mul(proj, viewRel), M4.translate([-eye[0],-eye[1],-eye[2]]));
    // viewProjNoEye = proj*viewRel WITHOUT the translate(-eye). The frustum cull must feed it
    // corners ALREADY made camera-relative (corner-eye, subtracted in JS double precision) --
    // folding translate(-eye) into the matrix and feeding ABSOLUTE ~6.37e6 m corners suffers
    // fp32 cancellation at ground level (eye~=world), garbaging the projection and blanking the
    // footprint. Subtracting in JS doubles first keeps the cull's projection precise near ground.
    const viewProjNoEye = M4.mul(proj, viewRel);
    return { viewProjRel, viewProjNoEye, eye, near, far, proj, viewRel };
  }

  // Render a set of quads. quads: [{quad, face, elevLayer, normalLayer}], cam: {eye, center, up, fovy}
  function render(quads, cam, sunDir, time) {
    const aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
    // ADAPTIVE near/far (altitude-tied). A fixed near=1 / far=R*8 (~5e7) at a 50km eye
    // pushed ALL near-surface geometry to NDC z~=1 (the far-plane limit), collapsing depth
    // precision so most near quads z-fought / clamped off -> only one screen rectangle
    // survived. Tie the planes to altitude: near = alt*0.1 (naturally scales from 1m at
    // deck to 1200km at orbit), far = horizon distance blended toward camDist above 500km
    // for orbital views. From space the far widens out to ~R*8, preserving the full-globe
    // view. Clamped so near>=1 and far>near.
    const camDist = Math.hypot(cam.eye[0], cam.eye[1], cam.eye[2]);
    const alt = Math.max(0.0, camDist - R);
    const altAboveTerrain = Math.max(0.001, alt - R * (cam.surfElev || 0));
    // FAR-PLANE HORIZON RADIUS = R - 500m (user 2026-06-14: 'nearby mountains disappear at water level;
    // adjust that level to 500m under water'). The far plane tracks the sea-level horizon = sqrt(camDist^2
    // - R^2), which at the deck (camDist~=R) collapses to a few hundred metres -> coastal mountains a km
    // out fall beyond the far plane and vanish. Dropping the horizon reference radius 500m below sea level
    // extends the horizon to tens of km at low altitude so near-shore relief stays in view (negligible
    // depth-precision cost: 500m vs R~6.37e6). Both the cull and the draw use this (single source).
    const RHORIZON = R - 150.0;   // far brought in 500->250 (user 2026-06-14 'bring far plane in a bit'): deck horizon ~80km->~56km = more z-precision; still clears coastal mountains
    // UNDERWATER FAR-PLANE FIX (user 2026-06-14 'at -214m visible, at -500m it disappears'): when the
    // camera is more than 500m below sea level, camDist < RHORIZON so the sea-level horizon is imaginary
    // (-> 0) and alt is negative; the old max(horizon, alt*8) then collapsed the far plane to ~0 and the
    // whole scene vanished past -500m deep (= the 'ocean looks shallow/empty' when exploring). Floor the
    // far reach to 60km when submerged so the seabed + the underwater view stay visible.
    const horizon = (camDist > RHORIZON) ? Math.sqrt(camDist*camDist - RHORIZON*RHORIZON) : 60000.0;
    const near = altAboveTerrain < 2.0 ? 0.5 : Math.max(altAboveTerrain * 0.1, 0.5);   // near nudged out 0.05->0.25 (user 2026-06-14 'improve on-ground'): more z-precision on the deck
    const _fBlend = Math.min(1.0, Math.max(0.0, (alt - 500000.0) / 4500000.0));
    const farGround = Math.max(horizon, alt * 8.0);
    const far = farGround * (1.0 - _fBlend) + camDist * _fBlend;
    const proj = M4.perspective(cam.fovy||0.785, aspect, near, far);
    const view = M4.lookAt(cam.eye, cam.center, cam.up||[0,1,0]);
    const viewProj = M4.mul(proj, view);
    // CAMERA-RELATIVE projection path (fp32 precision fix). At close range the world
    // coords (~6.36e6 m) and the eye (~9.5e6 m) are huge & nearly equal; view*world
    // suffers catastrophic fp32 cancellation, throwing gl_Position off-screen and
    // blanking the terrain. Build the projection so geometry is expressed RELATIVE to
    // the eye: place the eye at the origin (lookAt center-eye) and pre-translate world
    // corners by -eye. Then view*world differences are computed in fp32 BEFORE the big
    // magnitudes appear, so the small near-camera coords keep their precision. The
    // atmosphere/lighting path (camWorld, vWorld) stays ABSOLUTE -- only gl_Position is
    // relative.
    const eye = cam.eye;
    const _cm = cullMatrix(cam);
    const viewProjRel = _cm.viewProjRel;   // same matrix the frustum cull uses
    const viewProjNoEye = _cm.viewProjNoEye;   // proj*viewRel WITHOUT folded translate(-eye) -- for the
    // camera-relative VS path (vertex-jitter fix): the VS forms a SMALL camera-relative position so the
    // big ~6.4e6 radial magnitude never enters fp32 -> no ~0.5m quantization step = no vertex jitter.
    const _camDist = Math.hypot(eye[0], eye[1], eye[2]) || 1;
    const camDir = [eye[0]/_camDist, eye[1]/_camDist, eye[2]/_camDist];
    const camAlt = _camDist - R;
    // Expose the ACTUAL draw matrix + a finite-check for the motion debug probes. A NaN
    // viewProj (degenerate lookAt: fwd parallel up) is the classic disappear-on-move
    // signature -- all gl_Position go NaN and nothing draws. The probe reads __lastVP /
    // __lastVPFinite instead of guessing from a black screenshot.
    if (typeof window !== 'undefined') {
      window.__lastVP = viewProjRel;
      window.__lastVPFinite = viewProjRel.every(v => Number.isFinite(v));
      window.__deviceLost = gl.isContextLost();
    }

    gl.viewport(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight);
    gl.clearColor(0.0,0.0,0.0,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);

    // SKY/ATMOSPHERE PASS: atmospheric limb/halo behind the terrain. Fades in below
    // 100km (full at surface, transparent above 100km). Depth off so terrain overdraws.
    {
      const skyFade = Math.max(0.0, 1.0 - camAlt / 100000.0);
      if (skyFade > 0.001) {
        gl.useProgram(skyProg);
        gl.uniformMatrix3fv(SU('camRot'), false, new Float32Array([
          _cm.viewRel[0], _cm.viewRel[4], _cm.viewRel[8],
          _cm.viewRel[1], _cm.viewRel[5], _cm.viewRel[9],
          _cm.viewRel[2], _cm.viewRel[6], _cm.viewRel[10]
        ]));
        gl.uniform2f(SU('projDiag'), _cm.proj[0], _cm.proj[5]);
        gl.uniform3f(SU('skyCamWorld'), eye[0], eye[1], eye[2]);
        gl.uniform3f(SU('skySunDir'), sunDir[0], sunDir[1], sunDir[2]);
        gl.uniform1f(SU('skyR'), R);
        gl.uniform1f(SU('uSkyFade'), skyFade);
        gl.disable(gl.DEPTH_TEST);
        gl.bindVertexArray(skyVao);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
        gl.enable(gl.DEPTH_TEST);
      }
    }

    gl.enable(gl.DEPTH_TEST);
    // Back-face culling: the globe's FAR (back) hemisphere quads were drawing over the
    // near hemisphere (black faces that scramble on pan, worse at grazing angles). Cull
    // back-facing triangles so only the near hemisphere renders. windingCull selects
    // which winding is "front" (set from the witness; the deformed mesh + face frames
    // can flip it). Sky pass above is left unculled (fullscreen triangle).
    // Back-face cull mode. DEFAULT 'none': the GL winding-based cull is UNRELIABLE for the
    // spherically-deformed mesh -- the near-hemisphere patch winds GL-front at orbit but the
    // winding INVERTS at very low altitude, so a fixed cullFace(FRONT) renders the terrain
    // directly under the camera BLACK on descent / forward-flight (witnessed: cullFront cov
    // 1.0 at orbit but 0.17 at 2.5km; cullBack the opposite; cull NONE 1.0 at BOTH). The
    // depth buffer (enabled above) already resolves the far hemisphere correctly -- the near
    // surface always has smaller depth -- so no winding cull is needed. Override via
    // window.__cullMode = 'front'|'back' for diagnostics.
    const cm = window.__cullMode || 'none';
    if (cm === 'none') { gl.disable(gl.CULL_FACE); }
    else { gl.enable(gl.CULL_FACE);
      gl.cullFace((cm === 'back') ? gl.BACK : gl.FRONT);
      gl.frontFace(gl.CCW); }
    // ACTIVE PROGRAM select: a diagnostic displayMode needs the lazily-built debug program (which
    // carries the _DEBUGVIEW_ blocks). Build it on first request; until it finishes linking, fall
    // back to the render program (the lit view) for that frame -- no black flash, just one frame of
    // lit before the debug view appears. Modes 0/2/4 always use the render program.
    const _dm = cam.displayMode||0;
    if (DEBUG_MODES.has(_dm)) {
      ensureDebug();
      if (debugProg) setActiveProgram(debugProg, _dbgUloc); else setActiveProgram(prog, _uloc);
    } else { setActiveProgram(prog, _uloc); }
    gl.useProgram(_activeProg);
    gl.uniform3f(U('camWorld'), cam.eye[0], cam.eye[1], cam.eye[2]);
    gl.uniform1f(U('terrainR'), R);
    // camera-relative VS projection uniforms (vertex-jitter fix): the VS builds vRel = (dir0-camDir)*R
    // + dir0*h - camDir*camAlt (no 6.4e6 intermediate) and projects with defViewProjNoEye.
    gl.uniformMatrix4fv(U('defViewProjNoEye'), false, viewProjNoEye);
    gl.uniform3f(U('defCamDir'), camDir[0], camDir[1], camDir[2]);
    gl.uniform1f(U('defCamAlt'), camAlt);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
    gl.vertexAttribDivisor(0, 0);   // per-vertex
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    // No elevation/normal/ortho atlas: terrain shape+normal+material come from the GPU fractal +
    // biome ramp (the atlas producer is removed). Only the HPF continental field is sampled.
    // HPF continental field (TEXTURE3): sampled in the VS by world dir for the continental
    // elevation bias. hasHpf=0 -> VS uses 0 bias (graceful fallback before setHpf()).
    const hasHpf = !!_hpfTex;
    if (hasHpf) { gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D_ARRAY, _hpfTex); gl.uniform1i(U('hpfPool'), 3); }
    if (hasHpf && _hpfTex2) { gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D_ARRAY, _hpfTex2); gl.uniform1i(U('hpfPool2'), 5); }
    gl.uniform1i(U('hasHpf'), hasHpf ? 1 : 0);
    // aerial-perspective haze strength -- live A/B lever (1=on default, 0=off) to isolate/kill the
    // 'haze that melts into the land' on descent. The glancing-only graze weight in the shader
    // confines haze to the limb; this lever lets the witness compare with/without.
    // (aerialAmt setter DELETED 2026-06-11 dead-code sweep: the uniform left the shader earlier.)
    // FS-derivative normal lever. Default OFF: pure cross(dFdx,dFdy) of vWorld removes the tile
    // seam but exposes the coarse GRID mesh as facets -> moire (measured worse than the seam).
    // Kept as a lever; the seamless fix is the HYBRID in the FS (atlas detail + continuous base).
    // W5: fsNormal + pvNormal uniforms removed -- the shader no longer has them (THC Sobel is the sole
    // lit normal). The old per-vertex/dFdx normal levers are deleted.
    // per-vertex micro-displacement (coplanar-quads fix): unique sub-mesh-cell height per vertex
    // from world-continuous face-local fBm, so the surface isn't capped at the 21-texel atlas. default 1.
    // vtxDetail DEFAULT 0: the LOD-RELATIVE per-vertex micro-displacement popped between 1500/1200km
    // (amplitude scaled with tile size + faded in by altitude). The continuous broadShape (12 octaves,
    // absolute world wavelengths) now carries fine relief LOD-invariantly. Live re-enable via __vtxDetail.
    gl.uniform1f(U('vtxDetail'), (typeof window!=='undefined' && window.__vtxDetail!=null) ? +window.__vtxDetail : 1.0);
    // CLIFF / CANYON levers (live-tunable): canyon depth multiplier, cliff terrace strength (VS shape)
    // + strata band thickness and cliff-strata material strength (FS texturing). Defaults = the tuned
    // literals so the look is unchanged until the user dials a window global.
    const _g = (n,d)=> (typeof window!=='undefined' && window['__'+n]!=null) ? +window['__'+n] : d;
    gl.uniform1f(U('canyonDepthMul'), _g('canyonDepth', 1.0));
    gl.uniform1f(U('uVsCheap'),       (typeof window!=='undefined' && window.__vsCheap) ? 1.0 : 0.0);   // VS carve-cost profiling A/B
    gl.uniform1f(U('uBeachShelfM'),   _g('beachShelf', 0.0));   // land coastal shelf (geometry): h<S eased h*h/S = wide beach
    gl.uniform1f(U('uLandBias'),      _g('landBias', 0.0));        // +650m hypsometry bias = ~+30% land:sea (window.__landBias); MUST match the probe (setComposeHeightUniforms) for collision parity
    gl.uniform1f(U('uHiFreqCut'),     _g('hiFreqCut', 0.25));   // 0.5->0.25 (2026-06-10 'blotchy' -- see setComposeHeightUniforms)
    gl.uniform1f(U('uVertexAO'),      _g('vertexAO', 1.0));    // per-vertex shading/AO strength (DEFECT 2, 2026-06-06)
    gl.uniform1f(U('cliffAmt'),       _g('cliffAmt', 1.0));
    gl.uniform1f(U('uAoAmt'),         _g('aoAmt', 1.0));
    gl.uniform1f(U('uHpfInset'),      (typeof window!=='undefined' && window.__hpfInset === false) ? 0.0 : 1.0);   // SEAM FIX: inset sampler permanent default (matches bakeFace fu=x/(RES-1)); window.__hpfInset===false rolls back. PROBE+RENDER flip together.
    // uFloatLinearOK lets hpfSample collapse the manual 4-tap bilinear to a single hardware texture().
    gl.uniform1i(U('uFloatLinearOK'), _halfFloatLinearOK ? 1 : 0);
    gl.uniform1f(U('uWireframe'),     (typeof window!=='undefined' && window.__wireframe) ? 1.0 : 0.0);
    gl.uniform1f(U('uFsCheap'),        (typeof window!=='undefined' && window.__fsCheap) ? 1.0 : 0.0);  // GPU-timer VS-isolation frame (window.__gpuTimer)
    gl.uniform1f(U('uBiomeBandBias'), _g('biomeBandBias', 0.5));   // 1.0->0.5 (2026-06-10 'entire terrain white': elevCool h/4500 maxed the alpine temp-drop by 4.5km on the 11.6km terrain -> ice biome over all highland; halving via the uniform = effective h/9000, no shader-cache bust)
    // REAL-WORLD LOOK overhaul (live-tunable via window globals / DEFAULTS.look). Beer-Lambert ocean
    // extinction, biome saturation pull, intra-biome mottle, sky-fill relief, terminator sunset glow,
    // night floor + earthshine, exposure + post-ACES Look (sat/contrast). Defaults = the tuned look.
    gl.uniform1f(U('uBiomeSat'),       _g('biomeSat', 0.72));
    gl.uniform1f(U('uVariationAmt'),   _g('variationAmt', 0.04));   // 0.08->0.04 (2026-06-10 'blotchy': mottle patches across the 4x massifs)
    gl.uniform1f(U('uDetailOverlay'),  _g('detailOverlay', 6.0));   // perlin-everywhere albedo+elevation fbm (2026-06-10; user-tuned 6)
    gl.uniform1f(U('uHazeMul'),        _g('hazeMul', 0.65));        // aerial-perspective strength (2026-06-10 'pale hazy': 1.0 milked the midground)
    // uDiffWrap lives in ATMOSPHERE.glsl (atm_sunSkyIrradiance), not terrain.glsl -- the 2026-06-11
    // dead-code scan only covered terrain.glsl and wrongly deleted this setter (wrap silently -> 0,
    // restoring the very view-angle darkening b990add fixed). Scan BOTH shader files before
    // declaring a uniform dead.
    gl.uniform1f(U('uDiffWrap'),       _g('diffWrap', 0.5));   // diffuse wrap: 0.7 flattened ALL slope shading; 0.5 = grazing lift without killing the N.L relief keytion'); 0.5 = grazing lift without killing the N.L relief key
    // SURFACE PHOTO-TEXTURES (TEXTURE6/7): triplanar grass/rock/sand/snow splat. hasSurfTex stays 0
    // until the async loader uploads (procedural-only fallback, no flash -- the splat fades in).
    const hasSurf = !!_surfAlb;
    if (hasSurf) {
      gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D_ARRAY, _surfAlb); gl.uniform1i(U('uSurfAlb'), 6);
      gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D_ARRAY, _surfNrm); gl.uniform1i(U('uSurfNrm'), 7);
    }
    gl.uniform1f(U('uHasSurfTex'), hasSurf ? 1.0 : 0.0);
    gl.uniform1f(U('uTexTileM'),   _g('texTile', 2400.0));  // metres per repeat (user: 24m read as noise/rock -- 100x bigger)
    gl.uniform1f(U('uTexNrmK'),    _g('texNrmK', 1.5));     // 0.8 -> 3.0 (user 2026-06-11 'displacement normals must be THE texture normals'), 3.0->1.5 (2026-06-13 'texture displacement far too obvious vs elevation')
    gl.uniform1f(U('uTexMix'),     _g('texMix', 0.85));     // splat blend amount (0 = off)
    gl.uniform1f(U('uTexWarp'),    _g('texWarp', 0.23));    // anti-repetition warp amplitude (-30% from 0.325, grass warp too intense)
    gl.uniform1f(U('uTexPhoto'),   _g('texPhoto', 0.0));    // raw photo-color fraction (0 = patch matches the macro shade exactly)
    gl.uniform1f(U('uTexPhotoNear'), _g('texPhotoNear', 0.45));  // near-field material identity (photo hue at macro luminance; user 2026-06-12 'must be either grass or sand')
    gl.uniform4f(U('uSurfMeanL'), _surfMeanL[0], _surfMeanL[1], _surfMeanL[2], _surfMeanL[3]);   // per-layer mean linear luminance (shade-match divisor)
    // LIVE A/B ISOLATION TOGGLES (window.__rockBump / __chroma / __strata, default 1 = no change). Flip one
    // to 0 in the console to disable that detail layer and see which produces the close-up uv scramble.
    gl.uniform1f(U('uFlatNormal'),      _g('flatNormal', 0.0));   // 1 = smooth analytic normal (isolate the geometric-normal scramble)
    gl.uniform1f(U('uReliefShade'),    _g('reliefShade', 1.5));   // 2.4 -> 1.5 (user 'normals unusually darker': 2.4x tilt pushed ordinary 0.3-0.5 mountain slopes toward fully-shaded normals = broad darkening, not legible relief)
    gl.uniform1f(U('uSkyFill'),        _g('skyFill', 0.45));
    gl.uniform1f(U('uTerminatorGlow'), _g('terminatorGlow', 0.30));
    gl.uniform1f(U('uNightLights'),    _g('nightLights', 1.0));   // night/shadow FILL intensity (dim ambient lift so dark areas are not black); 0 = off
    gl.uniform1f(U('uNightFloor'),     _g('nightFloor', 0.16));   // night-longitude terminator floor RAISED 0.05->0.16 (no black night terrain)
    gl.uniform1f(U('uTermWidth'),      _g('termWidth', 0.25));
    gl.uniform1f(U('uExposure'),       _g('exposure', 1.0));
    gl.uniform1f(U('uLookSat'),        _g('lookSat', 1.15));
    gl.uniform1f(U('uLookContrast'),   _g('lookContrast', 1.08));
    { const o3=(n,d)=>{ const w=(typeof window!=='undefined'&&window['__'+n])||null; const v=(Array.isArray(w)&&w.length===3)?w:d; gl.uniform3f(U(n), v[0],v[1],v[2]); };
      o3('uOceanDeep',[0.008,0.025,0.06]); o3('uOceanShallow',[0.07,0.22,0.26]); o3('uOceanK',[0.016,0.007,0.0028]); }   // K halved (user 2026-06-14 'see the land under the water properly') = clearer water, bed visible through shallow/medium depth; deep basins still opaque
    // (the continuous broad-shape field is now always on - the single terrain shape source -
    // so its old on/off lever uniform was removed from terrain.glsl; nothing to set here.)
    // LIVE biome ramp (window.__gen.state.biome, else tuned defaults) -- full-adjustability.
    { const bm = (typeof window!=='undefined' && window.__gen && window.__gen.state && window.__gen.state.biome) || null;
      const C = (k,d)=> (bm && bm[k]) ? bm[k] : d;
      const c3 = (n,d)=>{ const v=C(n,d); gl.uniform3f(U(n), v[0],v[1],v[2]); };
      c3('bcDeepSea',[0.04,0.10,0.28]); c3('bcSea',[0.10,0.22,0.42]); c3('bcShore',[0.52,0.46,0.33]);
      c3('bcLowland',[0.24,0.42,0.18]); c3('bcGrass',[0.30,0.46,0.20]);
      // bcRock follows the ROCK PHOTO mean once loaded (user 2026-06-10 'replace the original rock
      // completely'): the far-field macro rock shade matches the near-field photo so the 15-20km
      // fade has no color pop. Falls back to the tuned grey-tan until the loader lands.
      c3('bcRock', (typeof window!=='undefined' && window.__surfRockMean) || [0.55,0.50,0.45]);
      c3('bcSnow',[0.92,0.94,0.97]);
      const e=C('bandEdgesLo',[150.0,1200.0]); gl.uniform2f(U('bandEdgesLo'), e[0],e[1]);
      const eh=C('bandEdgesHi',[3500.0,6500.0]); gl.uniform2f(U('bandEdgesHi'), eh[0],eh[1]);   // [1600,3200]->[3500,6500] (2026-06-10 'rockface everywhere': tuned pre-4x; with 11.6km peaks everything above 3200m was height-rock -- rescale the treeline)
      const sn=C('snowEdges',[6000.0,8500.0]); gl.uniform2f(U('snowEdges'), sn[0],sn[1]);   // 8000/10500->6000/8500 (user 2026-06-11 'all the snowy mountains have disappeared': only ~1% of land tops 8km (probe 3000-dir sweep, over7k 1.3%), so the whiteout-era snowline left virtually every massif bare; the whiteout's other sources (pre-rescale rock gates, alpine ice bias, tundra grey) are fixed independently, so 6km onset re-caps the real mountains without re-whitening the terrain)
      gl.uniform1f(U('seaDepthM'), C('seaDepthM',3000.0));
      const sr=C('slopeRock',[0.25,0.55]); gl.uniform2f(U('slopeRock'), sr[0],sr[1]); }   // [0.25,0.55] USER-SET 2026-06-12 (matches terrain-gen-controls persisted default)
    gl.uniform3f(U('sunDir'), sunDir[0],sunDir[1],sunDir[2]);
    gl.uniform1i(U('displayMode'), cam.displayMode||0);
    // ---- animated ocean uniforms. time advances the Gerstner waves; amp/choppy read
    // from the HUD ocean sliders (window.__cam) with sane defaults for v1.
    const oc = (typeof window !== 'undefined' && window.__cam) || {};
    gl.uniform1f(U('oceanTime'), time || 0.0);
    gl.uniform1f(U('oceanAmp'), (oc.oceanAmplitude != null) ? oc.oceanAmplitude : 1.0);
    gl.uniform1f(U('oceanChoppy'), (oc.oceanChoppiness != null) ? oc.oceanChoppiness : 0.5);
    gl.uniform1f(U('oceanFoam'), (oc.oceanFoam != null) ? oc.oceanFoam : 0.5);
    gl.uniform1f(U('uBeachTopM'), _g('beachTop', 80.0));   // beach ceiling: grass stops, sand to the waterline + under it. 30->90 (user 2026-06-14: 3x beach width)

    // SINGLE INSTANCED DRAW: the deform params that were per-quad uniforms (ox,oy,l,level + face)
    // are now PER-INSTANCE attributes. Build one interleaved instance buffer [ox,oy,l,level,face]
    // (5 floats/instance) from the visible leaf set and issue ONE gl.drawElementsInstanced -- no
    // per-quad uniform churn, no N draw calls. defViewProjRel is one uniform shared by all instances.
    gl.uniformMatrix4fv(U('defViewProjRel'), false, viewProjRel);
    gl.uniform1f(U('defRadius'), R);
    const n = quads.length;
    // Instance buffer: [ox,oy,l,level,face, iLayer] (6 floats). iLayer = the THC pool layer for this
    // tile (when __thc on); the VS samples the baked height there instead of composeHeight.
    const FLOATS = 6;   // [ox,oy,l,level,face, iLayer]
    if (n > 0) {
      // THC: when active, ensure every visible tile has a baked pool layer (bake on first sight). The
      // bakes clobber the FBO/program/viewport -> restore the canvas render state afterward.
      const _thc = thcActive();
      let _layers = null;
      if (_thc) {
        _tcFrame++; _tcBakesThisFrame = 0;
        _layers = new Float32Array(n);
        for (let i = 0; i < n; i++) { const q = quads[i].quad; _layers[i] = ensureTileLayer(quads[i].face, q.ox, q.oy, q.l, q.level); }
        gl.bindVertexArray(null);   // bakeTileToLayer left bakeVao bound; the main path uses the default VAO
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
        gl.enable(gl.DEPTH_TEST);   // bakeTileToLayer disabled depth
        gl.useProgram(_activeProg);
        if (typeof window !== 'undefined') window.__thcBakes = _tcBakesThisFrame;
      }
      // STATIC-FRAME SKIP: rebuild only when the quad set changed OR the toggle flipped (iLayer needs writing).
      const _dirty = (quads !== _instQuadsRef) || (_thc !== _lastThc);
      gl.bindBuffer(gl.ARRAY_BUFFER, instBuf);
      if (_dirty) {
        const inst = new Float32Array(n * FLOATS);
        for (let i = 0; i < n; i++) {
          const q = quads[i].quad;
          inst[i*FLOATS+0] = q.ox; inst[i*FLOATS+1] = q.oy; inst[i*FLOATS+2] = q.l; inst[i*FLOATS+3] = q.level;
          inst[i*FLOATS+4] = quads[i].face;
          inst[i*FLOATS+5] = _layers ? _layers[i] : 0.0;
        }
        gl.bufferData(gl.ARRAY_BUFFER, inst, gl.DYNAMIC_DRAW);
      }
      _lastThc = _thc;
      const STRIDE = FLOATS * 4;
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, STRIDE, 0);          gl.vertexAttribDivisor(1, 1);  // iOffset
      gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, STRIDE, 4 * 4);      gl.vertexAttribDivisor(2, 1);  // iFace
      gl.enableVertexAttribArray(3); gl.vertexAttribPointer(3, 1, gl.FLOAT, false, STRIDE, 5 * 4);      gl.vertexAttribDivisor(3, 1);  // iLayer (THC pool layer)
      gl.uniform1f(U('uThc'), _thc ? 1.0 : 0.0);
      if (_thc) { gl.activeTexture(gl.TEXTURE8); gl.bindTexture(gl.TEXTURE_2D_ARRAY, heightPool); gl.uniform1i(U('uHeightPool'), 8); gl.uniform1f(U('uPoolRes'), THC_BAKE_RES); gl.uniform1f(U('uPoolLinear'), _halfFloatLinearOK ? 1.0 : 0.0); }
      gl.uniform1f(U('uIsWater'), 0.0);
      // UNDERWATER DETECTION: camera below sea level enables underwater shading + water surface
      // rendering from below. Set before the terrain draw so the FS can apply underwater fog.
      const _uw = camDist < R - 2.0;
      gl.uniform1f(U('uUnderwater'), _uw ? 1.0 : 0.0);
      gl.drawElementsInstanced(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0, n);
      // SEPARATE WATER SURFACE (user 2026-06-11): second instanced draw with uIsWater=1 -- the VS
      // pins the mesh to sea level, the FS shades animated water and alpha-blends it over the
      // just-rendered seabed. Depth test keeps it behind land; depthMask off so the transparent
      // surface never occludes later passes. One program, one uniform flip.
      // OWN GEOMETRY, NOT THE TERRAIN TILES (user 2026-06-11 'the terrain tiles should not be
      // used for water'): the water sphere needs no terrain LOD -- deep leaves are wasted vertices
      // (each runs composeHeight) and re-tessellate with terrain detail the flat surface never
      // shows. Cap every visible leaf at level WCAP and DEDUP to its ancestor tile: a coarse,
      // LOD-churn-free cover of the same footprint, typically ~10-50x fewer water vertices.
      // __waterSurface=0 disables live.
      // The water surface draws in BOTH cases now (user 2026-06-14 'no water surface visible from
      // underneath'): it's geometrically ABOVE the camera, so underwater it is the up-view CEILING
      // (Snell's window, shaded in the uUnderwater branch) and the seabed below stays visible (the
      // down-view ray never hits the surface). With the fog 10x lighter it no longer washes the floor.
      if (typeof window === 'undefined' || window.__waterSurface !== false) {
        // WCAP 7 -> 9 (coast witness caught it: a level-7 tile's 16-cell mesh chord sags
        // A_cell^2/(8R) ~ 0.8m below the true sphere mid-cell -- BELOW the metres-deep shelf
        // seabed, so the depth test culled the water across entire shorelines. Level-9 cells
        // (~1.6km) sag ~5cm, far under any visible bathymetry, still ~16-64x fewer water verts
        // than the deep terrain leaves.
        // WCAP 9 -> 11 (user 2026-06-14 'water lines still jagged and square, doesnt meet land properly'):
        // the water-pass `if(vH>1.0) discard` keys off the water mesh's COARSE interpolated seabed height,
        // so the discarded waterline stepped at ~1.6km (level-9) cells = square/jagged edges that didn't
        // follow the fine seabed coastline. Level-11 cells (~400m) -> ~4x finer waterline. Water-vertex
        // cost rises (watch FPS); still far fewer verts than the full-LOD terrain leaves.
        const WCAP = 11;
        // OWN persistent buffer (instBufWater) + static-frame skip: on an unchanged quad set, reuse the
        // cached water instances (skip the dedup Set-loop + Float32Array + bufferData). Separate buffer
        // means the terrain pass never clobbers it (the prior water-as-terrain regression root).
        gl.bindBuffer(gl.ARRAY_BUFFER, instBufWater);
        if (_dirty || quads !== _instWaterRef) {
          const seen = new Set(); const wl = [];
          for (let i = 0; i < n; i++) {
            const q = quads[i].quad; let ox = q.ox, oy = q.oy, l = q.l, lv = q.level;
            if (lv > WCAP) { const A = l * Math.pow(2, lv - WCAP); ox = Math.floor(ox / A) * A; oy = Math.floor(oy / A) * A; l = A; lv = WCAP; }
            const key = quads[i].face + ':' + ox + ':' + oy + ':' + l;
            if (seen.has(key)) continue; seen.add(key);
            wl.push(ox, oy, l, lv, quads[i].face, 0);   // iLayer unused for water (VS pins sea level)
          }
          _instWaterN = wl.length / FLOATS;
          gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(wl), gl.DYNAMIC_DRAW);
          _instWaterRef = quads;
        }
        const wn = _instWaterN;
        gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 4, gl.FLOAT, false, STRIDE, 0);     gl.vertexAttribDivisor(1, 1);
        gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 1, gl.FLOAT, false, STRIDE, 4 * 4); gl.vertexAttribDivisor(2, 1);
        if (_uw) {
          gl.disable(gl.BLEND);
          gl.depthMask(true);
        } else {
          gl.enable(gl.BLEND);
          gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
          gl.depthMask(false);
        }
        gl.uniform1f(U('uIsWater'), 1.0);
        gl.drawElementsInstanced(gl.TRIANGLES, indices.length, gl.UNSIGNED_INT, 0, wn);
        gl.uniform1f(U('uIsWater'), 0.0);
        gl.depthMask(true);
        gl.disable(gl.BLEND);
        if (typeof window !== 'undefined') window.__lastWaterQuads = wn;
      }
      _instQuadsRef = quads;   // mark this quad set uploaded; next frame with the same array skips the rebuild
      if (typeof window !== 'undefined') window.__instUploads = (window.__instUploads | 0) + (_dirty ? 1 : 0);
    }
    if (typeof window !== 'undefined') window.__lastDrawCalls = (n > 0) ? 2 : 0;
    return 0;   // glError is checked via checkGlError() once per frame after quadtree (CPU/GPU pipelining)
  }

  function checkGlError() { return gl.getError(); }

  // ---- DEBUG PROBE: replicate the VS clip-space transform on the CPU for a quad's 4
  // corners (vertex.xy in {0,1}^2) so we can see which quads project off-screen.
  function probe(quads, cam) {
    const aspect = gl.drawingBufferWidth / gl.drawingBufferHeight;
    const near = (cam.near!=null)?cam.near:1.0, far=(cam.far!=null)?cam.far:R*8;
    const proj = M4.perspective(cam.fovy||0.785, aspect, near, far);
    const eye = cam.eye;
    const viewRel = M4.lookAt([0,0,0], [cam.center[0]-eye[0], cam.center[1]-eye[1], cam.center[2]-eye[2]], cam.up||[0,1,0]);
    const viewProjRel = M4.mul(M4.mul(proj, viewRel), M4.translate([-eye[0],-eye[1],-eye[2]]));
    const out = [];
    for (const q of quads) {
      const w3 = localToWorld3(q.face);
      const w4 = new Float32Array([ w3[0],w3[1],w3[2],0, w3[3],w3[4],w3[5],0, w3[6],w3[7],w3[8],0, 0,0,0,1 ]);
      const localToScreen = M4.mul(viewProjRel, w4);
      const {ox,oy,l} = q.quad;
      const cs = [[ox,oy],[ox+l,oy],[ox,oy+l],[ox+l,oy+l]];
      const v=[],L=[];
      for (let i=0;i<4;i++){ const px=cs[i][0],py=cs[i][1]; const len=Math.hypot(px,py,R); L.push(len); v.push([px/len,py/len,R/len]); }
      // C and N matrices (4x4) as in setQuadUniforms
      const dCorners = new Float32Array([ v[0][0]*R,v[0][1]*R,v[0][2]*R,1, v[1][0]*R,v[1][1]*R,v[1][2]*R,1, v[2][0]*R,v[2][1]*R,v[2][2]*R,1, v[3][0]*R,v[3][1]*R,v[3][2]*R,1 ]);
      const C = M4.mul(localToScreen, dCorners);
      // For each of the 4 mesh corners, alphaPrime picks out one column => clip = column i (h=0 baseline)
      const ndc = [];
      for (let i=0;i<4;i++){ const x=C[i*4],y=C[i*4+1],z=C[i*4+2],w=C[i*4+3];
        ndc.push({x:+(x/w).toFixed(3),y:+(y/w).toFixed(3),z:+(z/w).toFixed(3),w:+w.toFixed(1),
          off: (w<=0)||Math.abs(x/w)>1||Math.abs(y/w)>1||(z/w)<-1||(z/w)>1}); }
      out.push({face:q.face, level:q.quad.level, ox:+ox.toFixed(0), oy:+oy.toFixed(0), l:+l.toFixed(0), ndc});
    }
    return out;
  }
  function setHpf(tex, res, tex2) { _hpfTex = tex; _hpfRes = res|0; _hpfTex2 = tex2 || null; invalidatePool(); }   // tex2 = RG8(temp,humid) pack (W12); HPF change -> re-bake THC tiles
  return { get prog(){ return prog; }, render, checkGlError, probe, sampleGroundM, cullMatrix, recompile, setHpf, GRID, indexCount: indices.length, M4 };
}
