// tweak-panel.js -- LIVE control menus for every window.__ shader lever (no reload needed).
//
// Each row writes window.__<key>; gl-render reads window.__<key> EVERY FRAME via its g()/_g() helpers,
// so moving a slider changes the look instantly. Grouped + collapsible. A row does NOT force-set the
// global on build (so untouched levers keep gl-render's own default, and splitFactor stays on the
// altitude ramp until you actually touch it) -- the global is written only when you move the control.
//
// Imported (cache-busted) by planet.html so it reliably reaches a warm tab. Add a lever here the moment
// you wire a new window.__ uniform in gl-render -- this is the single place to expose tweakables.

// [key, label, min, max, step, default-shown]. key -> window.__<key>.
// DEFAULTS BAKED 2026-06-16 from the user's live-tuned session (read off their tab via /cmd, cw:1920).
// These ARE the look the user dialled in; applyBaked() below forces them as the live globals on load so
// a fresh page == the tuned look (no manual re-tweak). 'r' resets a slider to its baked value.
const GROUPS = [
  ['Canyon / carve', [
    ['canyonDepth',   'Canyon depth',          0,   8,    0.1,  0.0],
    ['carveWide',     'Carve climate width',   0,   1,    0.05, 0.0],
  ]],
  ['Cliffs / mesa', [
    ['cliffAmt',      'Cliff / mesa amount',   0,   3,    0.1,  3.0],
  ]],
  ['Terrain shape', [
    ['detailOverlay', 'Detail overlay relief', 0,   60,   0.5,  50.0],
    ['hiFreqCut',     'Hi-freq octave cut',    0,   1,    0.05, 0.95],
    ['landBias',      'Land bias (m)',        -1000, 1500, 50,  -800.0],
    ['mtnBandWide',   'Mountain belt width',   0,   1,    0.05, 0.1],
    ['climateRelief', 'Climate relief width',  0,   1,    0.05, 0.65],
    ['isleWide',      'Island zone width',     0,   1,    0.05, 1.0],
    ['nrmStepM',      'Normal smooth step m',  50,  1200, 25,   0.0],
    ['splitFactor',   'LOD density (split)',   0.05, 1.0, 0.01, 0.30],
  ]],
  ['Beach / coast', [
    ['beachTop',      'Beach ceiling (m)',     0,   1000, 10,   210.0],
    ['beachWidth',    'Beach crossover width', 0,   20,   0.5,  5.0],
    ['beachShelf',    'Coastal shelf (m)',     0,   1200, 50,   0.0],
    ['bandWarp',      'Biome-edge warp (m)',   0,   3000, 50,   1800.0],
  ]],
  ['Texture', [
    ['texMix',        'Splat blend',           0,   1,    0.05, 1.0],
    ['texNrmK',       'Detail-normal strength',0,   4,    0.1,  1.6],
    ['nrmLow',        'Low-octave normal',     0,   3,    0.1,  0.5],
    ['triSharp',      'Triplanar sharpness',   1,   10,   0.5,  10.0],
    ['texSat',        'Texture saturation',    0,   2,    0.05, 2.0],
    ['texBright',     'Texture brightness',    0.5, 1.5,  0.02, 0.8],
    ['biomeTint',     'Biome tint / texture',  0,   1,    0.02, 0.3],
    ['texWarp',       'Anti-repeat warp',      0,   0.6,  0.02, 0.22],
    ['xSoft',         'Crossover fade width',  0,   0.5,  0.02, 0.5],
    ['xFinger',       'Crossover fingering',   0,   5,    0.1,  2.5],
    ['ordPush',       'Overlay push (cover)',  0,   1.5,  0.05, 0.4],
    ['texPhoto',      'Photo color (far)',     0,   1,    0.05, 0.0],
    ['texPhotoNear',  'Photo color (near)',    0,   1,    0.05, 0.0],
    ['xFade0',        'Crossover fade start m',0,   40000,1000, 15000],
    ['xFade1',        'Crossover fade end m',  0,   80000,2000, 40000],
    ['texFar0',       'Splat fade start m',    0,   40000,1000, 0.0],
    ['texFar1',       'Splat fade end m',      0,   80000,2000, 12000],
    ['nrmFade0',      'Normal fade start m',   0,   120000,5000,45000],
    ['nrmFade1',      'Normal fade end m',     0,   200000,10000,100000],
  ]],
  ['Lighting / look', [
    ['exposure',      'Exposure',              0.2, 4,    0.05, 0.75],
    ['reliefShade',   'Relief shading',        0,   6,    0.1,  1.8],
    ['biomeSat',      'Biome saturation',      0,   2,    0.05, 1.05],
    ['biomeWarp',     'Biome distribution warp',0,  3,    0.1,  1.8],
    ['variationAmt',  'Color variation',       0,   0.3,  0.01, 0.04],
    ['hazeMul',       'Aerial haze',           0,   2,    0.05, 0.65],
    ['diffWrap',      'Diffuse wrap',          0,   1,    0.05, 0.5],
    ['skyFill',       'Sky fill',              0,   1,    0.05, 0.45],
    ['vertexAO',      'Vertex AO',             0,   2,    0.05, 1.0],
    ['aoAmt',         'Slope AO',              0,   3,    0.1,  1.0],
    ['biomeBandBias', 'Biome height bias',     0,   2,    0.05, 1.3],
    ['lookSat',       'Final saturation',      0,   2,    0.05, 1.15],
    ['lookContrast',  'Final contrast',        0.5, 2,    0.02, 1.12],
    ['terminatorGlow','Terminator glow',       0,   1,    0.05, 1.0],
    ['nightFloor',    'Night floor',           0,   0.5,  0.02, 0.4],
    ['nightLights',   'Night fill',            0,   3,    0.1,  1.2],
    ['termWidth',     'Terminator width',      0.05,1,    0.05, 0.05],
    ['flatNormal',    'Flat normal (diag)',    0,   1,    1,    0.0],
  ]],
  ['Performance', [
    // Render scale: drawing-buffer pixels per CSS pixel. Lower = faster (the deck is partly fill-bound:
    // half-res measured -2ms/frame on the APU), softer (browser upscales). The biggest detail-PRESERVING
    // FPS lever -- all geometry+material stays, just resolution drops. Live via window.__setRenderScale.
    ['renderScale',   'Render scale (fps<->sharp)', 0.4, 1.5, 0.05, 1.0],
  ]],
];

// The baked values applied as live globals on load. FORCE-set (not null-guarded) so they win over
// gen-controls' applyShaderGlobals for the shared look levers; runs after a double-rAF (post-init).
function applyBaked(){
  for (const [, levers] of GROUPS) for (const [key,,,,,def] of levers) {
    if (key === 'renderScale') continue;   // canvas init owns the load default (DPR); the slider drives it live, don't force a load-time resize
    window['__' + key] = def;
  }
  const o = window.__planetOrch; if (o && o.clearCache) o.clearCache();
}

function build(){
  if (document.getElementById('tweakPanel')) return;
  const btn = document.createElement('button');
  btn.id = 'tweakToggle'; btn.textContent = 'Tweaks';
  btn.style.cssText = 'position:fixed;left:92px;top:8px;z-index:31;font:11px monospace;padding:3px 7px';

  const panel = document.createElement('div');
  panel.id = 'tweakPanel';
  panel.style.cssText = 'position:fixed;left:8px;top:34px;z-index:31;display:none;max-height:90vh;overflow:auto;' +
    'background:rgba(12,16,20,0.94);color:#cfe;font:10px/1.4 monospace;padding:6px 8px;border:1px solid #2a3a44;width:330px';

  btn.onclick = () => {
    const show = panel.style.display === 'none';
    panel.style.display = show ? 'block' : 'none';
    const gp = document.getElementById('genPanel'); if (gp && show) gp.style.display = 'none';
  };

  for (const [groupName, levers] of GROUPS) {
    const det = document.createElement('details');
    det.open = groupName === 'Canyon / carve' || groupName === 'Cliffs / mesa';
    const sum = document.createElement('summary');
    sum.textContent = groupName;
    sum.style.cssText = 'cursor:pointer;color:#9fd;font-weight:bold;margin:5px 0 2px';
    det.appendChild(sum);

    for (const [key, label, min, max, step, def] of levers) {
      const g = '__' + key;
      const cur = (window[g] != null) ? +window[g] : def;   // SHOW current/default, do NOT write the global until touched
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;align-items:center;gap:4px;margin:1px 0';

      const lab = document.createElement('span');
      lab.textContent = label; lab.title = 'window.__' + key;
      lab.style.cssText = 'flex:0 0 120px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis';

      const rng = document.createElement('input');
      rng.type = 'range'; rng.min = min; rng.max = max; rng.step = step; rng.value = cur; rng.style.flex = '1';

      const num = document.createElement('input');
      num.type = 'number'; num.min = min; num.max = max; num.step = step; num.value = cur;
      num.style.cssText = 'flex:0 0 60px;background:#0a0e12;color:#cfe;border:1px solid #2a3a44;font:10px monospace';

      const apply = (v) => { v = +v; if (!isFinite(v)) return; window[g] = v; rng.value = v; num.value = v;
        if (key === 'renderScale' && window.__setRenderScale) { window.__setRenderScale(v); return; }   // live buffer resize, not a shader uniform
        const o = window.__planetOrch; if (o && o.clearCache) o.clearCache(); };
      rng.oninput = () => apply(rng.value);
      num.oninput = () => apply(num.value);

      const rst = document.createElement('button');
      rst.textContent = 'r'; rst.title = 'reset to ' + def;
      rst.style.cssText = 'flex:0 0 16px;font:10px monospace;padding:0;cursor:pointer';
      rst.onclick = () => apply(def);

      row.append(lab, rng, num, rst);
      det.appendChild(row);
    }
    panel.appendChild(det);
  }

  document.body.append(btn, panel);
  window.__tweakPanel = { rebuild: build };
}

function boot(){ build(); requestAnimationFrame(() => requestAnimationFrame(applyBaked)); }
if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
else boot();
