// tweak-panel.js -- LIVE control menus for every window.__ shader lever (no reload needed).
//
// Each row writes window.__<key>; gl-render reads window.__<key> EVERY FRAME via its g()/_g() helpers,
// so moving a slider changes the look instantly. Grouped + collapsible. A row does NOT force-set the
// global on build (so untouched levers keep gl-render's own default, and splitFactor stays on the
// altitude ramp until you actually touch it) -- the global is written only when you move the control.
//
// Imported (cache-busted) by planet.html so it reliably reaches a warm tab. Add a lever here the moment
// you wire a new window.__ uniform in gl-render -- this is the single place to expose tweakables.

// [key, label, min, max, step, default-shown]. key -> window.__<key>; default mirrors the gl-render g() default.
const GROUPS = [
  ['Canyon / carve', [
    ['canyonDepth',   'Canyon depth',          0,   8,    0.1,  2.0],
    ['carveWide',     'Carve climate width',   0,   1,    0.05, 0.0],
  ]],
  ['Cliffs / mesa', [
    ['cliffAmt',      'Cliff / mesa amount',   0,   3,    0.1,  1.0],
  ]],
  ['Terrain shape', [
    ['detailOverlay', 'Detail overlay relief', 0,   15,   0.5,  6.0],
    ['vtxDetail',     'Vertex micro-detail',   0,   2,    0.05, 1.0],
    ['hiFreqCut',     'Hi-freq octave cut',    0,   1,    0.05, 0.25],
    ['landBias',      'Land bias (m)',        -1000, 1500, 50,  0.0],
    ['mtnBandWide',   'Mountain belt width',   0,   1,    0.05, 0.0],
    ['climateRelief', 'Climate relief width',  0,   1,    0.05, 0.0],
    ['isleWide',      'Island zone width',     0,   1,    0.05, 0.0],
    ['nrmStepM',      'Normal smooth step m',  50,  1200, 25,   300.0],
    ['splitFactor',   'LOD density (split)',   0.05, 1.0, 0.01, 0.28],
  ]],
  ['Beach / coast', [
    ['beachTop',      'Beach ceiling (m)',     0,   1000, 10,   60.0],
    ['beachWidth',    'Beach crossover width', 0,   20,   0.5,  5.0],
    ['beachShelf',    'Coastal shelf (m)',     0,   1200, 50,   0.0],
    ['bandWarp',      'Biome-edge warp (m)',   0,   3000, 50,   1100.0],
  ]],
  ['Texture', [
    ['texMix',        'Splat blend',           0,   1,    0.05, 0.85],
    ['texNrmK',       'Detail-normal strength',0,   4,    0.1,  1.0],
    ['nrmLow',        'Low-octave normal',     0,   3,    0.1,  1.0],
    ['triSharp',      'Triplanar sharpness',   1,   10,   0.5,  4.0],
    ['texSat',        'Texture saturation',    0,   2,    0.05, 1.0],
    ['texBright',     'Texture brightness',    0.5, 1.5,  0.02, 0.92],
    ['biomeTint',     'Biome tint / texture',  0,   1,    0.02, 0.22],
    ['texWarp',       'Anti-repeat warp',      0,   0.6,  0.02, 0.23],
    ['xSoft',         'Crossover softness far',0,   0.5,  0.02, 0.14],
    ['texPhoto',      'Photo color (far)',     0,   1,    0.05, 0.0],
    ['texPhotoNear',  'Photo color (near)',    0,   1,    0.05, 0.45],
    ['xFade0',        'Crossover fade start m',0,   40000,1000, 8000],
    ['xFade1',        'Crossover fade end m',  0,   80000,2000, 20000],
    ['texFar0',       'Splat fade start m',    0,   40000,1000, 4000],
    ['texFar1',       'Splat fade end m',      0,   80000,2000, 26000],
    ['nrmFade0',      'Normal fade start m',   0,   120000,5000,40000],
    ['nrmFade1',      'Normal fade end m',     0,   200000,10000,80000],
  ]],
  ['Lighting / look', [
    ['exposure',      'Exposure',              0.2, 4,    0.05, 1.0],
    ['reliefShade',   'Relief shading',        0,   6,    0.1,  1.8],
    ['biomeSat',      'Biome saturation',      0,   2,    0.05, 0.72],
    ['variationAmt',  'Color variation',       0,   0.3,  0.01, 0.04],
    ['hazeMul',       'Aerial haze',           0,   2,    0.05, 0.65],
    ['diffWrap',      'Diffuse wrap',          0,   1,    0.05, 0.5],
    ['skyFill',       'Sky fill',              0,   1,    0.05, 0.45],
    ['vertexAO',      'Vertex AO',             0,   2,    0.05, 1.0],
    ['aoAmt',         'Slope AO',              0,   3,    0.1,  1.0],
    ['biomeBandBias', 'Biome height bias',     0,   2,    0.05, 0.5],
    ['lookSat',       'Final saturation',      0,   2,    0.05, 1.15],
    ['lookContrast',  'Final contrast',        0.5, 2,    0.02, 1.08],
    ['terminatorGlow','Terminator glow',       0,   1,    0.05, 0.30],
    ['nightFloor',    'Night floor',           0,   0.5,  0.02, 0.16],
    ['nightLights',   'Night fill',            0,   3,    0.1,  1.0],
    ['termWidth',     'Terminator width',      0.05,1,    0.05, 0.25],
    ['flatNormal',    'Flat normal (diag)',    0,   1,    1,    0.0],
  ]],
];

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

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', build);
else build();
