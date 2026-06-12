// quadtree.js -- the cube-face terrain quadtree LOD selection, ported from the deleted
// proland_terrain.cpp (TerrainNode/TerrainQuad subdivision + SphericalDeformation + the
// pixel-calibrated split distance). This is the ONE piece of the old C++/wasm the render
// needed; everything else (the elevation/normal/ortho atlas producer + cascade) is gone --
// terrain shape is the single GPU fractal (broadShapeM in terrain.glsl), evaluated per-vertex.
//
// Pure JS, no wasm, no allocation per frame beyond the leaf array. One Quadtree instance per
// run; call setConfig once, computeSplitDist when viewport/fov change, updateQuadtree per frame
// with the camera in LOCAL (undeformed cube-face) space -> returns leaf quads [{level,tx,ty,ox,oy,l}].

export class Quadtree {
  constructor() {
    this.size = 6360000.0;     // root half-extent / planet radius (m)
    this.maxLevel = 20;
    this.splitDist = 1.1;      // computed from splitFactor+viewport+fov
    this.distFactor = 2.0;     // altitude weighting
    this._cam = [0, 0, 0];     // camera in LOCAL (undeformed) space (aim-shifted LOD reference)
    this._camAlt = 0.0;        // true altitude above the sphere (|localCam|-R), same on every face
    this._nadir = [0, 0];      // TRUE camera nadir (face-local x,y) for the far-LOD foreground protect
    this._aim = null;          // AIM ground point (face-local x,y), 2nd foreground-protect center (or null)
    this._leaves = [];
  }

  setConfig(size, maxLevel, distFactor) {
    this.size = size; this.maxLevel = maxLevel; this.distFactor = distFactor;
  }

  // splitDist = splitFactor * viewportH/1024 * tan(40deg)/tan(fov/2), clamped >= 1.1.
  computeSplitDist(splitFactor, viewportH, fovRad) {
    let sd = splitFactor * viewportH / 1024.0 * Math.tan(40.0 / 180.0 * Math.PI) / Math.tan(fovRad / 2.0);
    if (!(sd >= 1.1) || !isFinite(sd)) sd = 1.1;
    this.splitDist = sd;
    return sd;
  }

  // TerrainNode::getCameraDist: local-space max-distance metric. The altitude term uses the TRUE
  // camera altitude above the sphere (same on every face -> adjacent faces subdivide by real
  // proximity, the fix for the turn-around-unpatched bug).
  _cameraDist(ox, oy, l) {
    const dz = Math.max(this._camAlt / this.distFactor, 0.0);
    const dx = Math.min(Math.abs(this._cam[0] - ox), Math.abs(this._cam[0] - (ox + l)));
    const dy = Math.min(Math.abs(this._cam[1] - oy), Math.abs(this._cam[1] - (oy + l)));
    return Math.max(dz, Math.max(dx, dy));
  }

  // TerrainQuad subdivision: subdivide while dist < l*splitDist AND level < maxLevel; emit leaves.
  // DISTANCE FALLOFF (user 2026-06-01j: 'when close to terrain, far-away quads should LOD better').
  // The plain dist<l*splitDist gives constant screen-space leaf size (leaf ~ dist/splitDist), so when
  // the camera is LOW the horizon-distant terrain still subdivides fine = wasted quads. Tighten the
  // effective splitDist with the quad's LATERAL distance from the camera: far quads need to be much
  // bigger to keep splitting, so distant terrain stays coarse while the near/under-camera stays fine.
  // Falloff scaled by altitude so high views (where 'far' is the whole visible disc) are unaffected.
  _recurse(level, tx, ty, ox, oy, l) {
    const dist = this._cameraDist(ox, oy, l);
    // DISTANCE FALLOFF: coarsen far quads when low. Use the lateral distance to the quad CENTER
    // (NOT its nearest edge -- a big near quad has a far edge, which must NOT penalize it, or the
    // root never splits). Only quads whose CENTRE sits well beyond the near footprint coarsen.
    const cxv = ox + l * 0.5, cyv = oy + l * 0.5;
    // latC = lateral distance from the TRUE camera NADIR (this._nadir), NOT the aim-shifted LOD ref.
    // This decouples foreground-protection from the aim-shift: the foreground (near the true nadir)
    // always has latC~0 -> fall=1.0 -> full LOD, no matter how low the floor. So a LOW floor can
    // reduce the far horizon hard WITHOUT collapsing nearby terrain (user: 'nearby reduced all the
    // way when going to land' was the aim-shifted ref penalizing the foreground).
    // latC = lateral distance from the nadir to the quad's NEAREST POINT (clamp nadir into the quad's
    // [ox,ox+l]x[oy,oy+l] extent), NOT its CENTER. THE CENTER WAS THE BUG (user 2026-06-02: 'dense LOD
    // off to the side, not under the camera; moving toward a face edge it wraps'): a LARGE quad that
    // CONTAINS the camera has its center offset by up to l/2, so a center-distance falloff penalized
    // the camera's own containing quads and they never subdivided -- the only fine quads landed where a
    // quad CENTER happened to align with nadir (off toward the face centre). Nearest-point distance is
    // ~0 for any quad containing the nadir, so the footprint refines UNDER the camera at any face pos.
    const nx0 = Math.max(ox, Math.min(this._nadir[0], ox + l));
    const ny0 = Math.max(oy, Math.min(this._nadir[1], oy + l));
    let latC = Math.max(Math.abs(this._nadir[0] - nx0), Math.abs(this._nadir[1] - ny0));
    // Protect the foreground around the AIM ground point too (the pitched-down bottom-of-screen band):
    // nearest-point distance to the aim box as well; take the NEARER so a quad close to EITHER is fine.
    if (this._aim !== null) {
      const ax0 = Math.max(ox, Math.min(this._aim[0], ox + l));
      const ay0 = Math.max(oy, Math.min(this._aim[1], oy + l));
      const latA = Math.max(Math.abs(this._aim[0] - ax0), Math.abs(this._aim[1] - ay0));
      if (latA < latC) latC = latA;
    }
    // penalty-free near radius. At very low alt the ON-SCREEN foreground extends to the HORIZON
    // (~sqrt(2*R*alt): 1km->113km), so a fixed 20km near coarsened the visible field on sea-level
    // landing (user 2026-06-02: 'at sea level LOD reduces when landing'). Scale near to the horizon
    // distance so the whole visible field out to ~0.6*horizon stays full-LOD at any landing altitude;
    // camAlt*6 dominates at higher alt where the horizon is far. size == R.
    // near = penalty-free radius around the foreground. It must cover the WHOLE on-screen field so
    // detail keeps INCREASING all the way to the ground (user 2026-06-02: 'LOD not increasing under
    // 500km'). The visible field reaches the HORIZON (~sqrt(2*R*alt)); protect the FULL horizon (was
    // 0.6) so nothing on-screen is coarsened by the far-LOD falloff -- the falloff then only trims
    // terrain BEYOND the horizon (off-screen waste), never visible detail.
    const horizon = Math.sqrt(2.0 * this.size * Math.max(this._camAlt, 0.0));
    // W2 SINGLE-VERSION near-radius tighten (mob-w2, unconditional): max(camAlt*6, horizon*0.9,
    // 20km). Shrinking the penalty-free near radius lets the far-LOD falloff coarsen MORE of the
    // off-screen / past-horizon field, cutting peak visible leaves toward ~600-900 (precondition
    // for the 512 layer cap). These are THE only values -- no device tier.
    const near = Math.max(this._camAlt * 6.0, horizon * 0.9, 20000.0);
    // floor 0.5 (was 0.18): past the horizon the split may HALVE, not crush to ~1/5 (which was
    // collapsing near-field detail on descent). Foreground (latC<near) stays fall=1 -> full LOD.
    const fall = 1.0 / (1.0 + Math.max(0.0, latC - near) / (near * 2.0));
    // ALTITUDE-DETAIL-GRADIENT-SWAP (user: 'above fps height, swap more near detail for far detail so the
    // higher we get the less of a detail gradient there is to the land center'). The far-coarsening floor
    // is normally 0.5 (far quads split at half the near rate = a radial detail gradient around the
    // camera). Above the fps height (~5km AGL = first-person ground play) raise that floor toward 1.0 as
    // altitude climbs, so far and near refine at the same rate = the gradient FLATTENS the higher we get
    // (detail budget spreads from the near foreground out to the far field). Below fps height the gradient
    // stays sharp (floor 0.5) so the deck keeps its near-field detail. fpsAltM live via this.fpsAltM.
    const fpsAltM = this.fpsAltM || 5000.0;
    const aboveFps = Math.min(1.0, Math.max(0.0, (this._camAlt - fpsAltM) / (fpsAltM * 8.0)));  // 0 at fps height -> 1 by ~45km
    const floor = 0.5 + 0.5 * (aboveFps * aboveFps * (3 - 2 * aboveFps));   // 0.5 (fps deck) -> 1.0 (high alt, flat gradient)
    const effSplit = this.splitDist * Math.max(floor, fall);
    if (dist < l * effSplit && level < this.maxLevel) {
      const hl = l / 2.0;
      this._recurse(level + 1, 2 * tx,     2 * ty,     ox,      oy,      hl);
      this._recurse(level + 1, 2 * tx + 1, 2 * ty,     ox + hl, oy,      hl);
      this._recurse(level + 1, 2 * tx,     2 * ty + 1, ox,      oy + hl, hl);
      this._recurse(level + 1, 2 * tx + 1, 2 * ty + 1, ox + hl, oy + hl, hl);
    } else {
      this._leaves.push({ level, tx, ty, ox, oy, l });
    }
  }

  // Run the quadtree for the current camera (LOCAL cube-face coords); returns the leaf-quad array.
  // The root quad spans [-size, size] in x and y (one cube face).
  // camX/Y/Z = the (aim-shifted) LOD reference; nadirX/nadirY (optional) = the TRUE camera nadir in
  // face-local coords, used by the far-LOD falloff to protect the real foreground (decoupled from the
  // aim-shift). Defaults to the LOD ref's x,y when not supplied (back-compat). aimX/aimY (optional) =
  // the look ray's ground-hit in face-local coords; a SECOND foreground-protect center so the
  // pitched-down bottom-of-screen band stays full-LOD in every azimuth (null/omitted -> nadir only).
  updateQuadtree(camX, camY, camZ, nadirX, nadirY, aimX, aimY, camAlt) {
    this._cam[0] = camX; this._cam[1] = camY; this._cam[2] = camZ;
    // TRUE ALTITUDE (fix off-center LOD stall, user 2026-06-03 'LOD dense only at the start point').
    // camX,camY are the atan-WARPED face-local lateral coords (from worldToFaceLocal); off the face
    // centre they are large (up to ~R near the edge), so sqrt(camX^2+camY^2+camZ^2) hugely
    // OVERESTIMATES altitude -> the altitude term dominates the split metric -> LOD stalls everywhere
    // but the face centre (where camX=camY=0 makes the hypot correct). Use the caller's TRUE altitude
    // (|camWorld|-R) when supplied; fall back to the old hypot only for back-compat.
    this._camAlt = (camAlt !== undefined && camAlt !== null)
      ? camAlt
      : Math.sqrt(camX * camX + camY * camY + camZ * camZ) - this.size;
    this._nadir[0] = (nadirX !== undefined) ? nadirX : camX;
    this._nadir[1] = (nadirY !== undefined) ? nadirY : camY;
    this._aim = (aimX !== undefined && aimY !== undefined) ? [aimX, aimY] : null;
    this._leaves.length = 0;
    this._recurse(0, 0, 0, -this.size, -this.size, 2.0 * this.size);
    return this._leaves;
  }
}

// SphericalDeformation::localToDeformed with the tangent-adjusted (equal-area-ish) cube->sphere
// remap: warp the normalized face coord s=x/R by tan(s*pi/4) before the radial projection so cell
// area is near-uniform (seam-safe: identity at s=+-1). MUST match terrain.glsl faceWarp + the VS.
// q = (R+z) * normalize(R*tan(x/R*pi/4), R*tan(y/R*pi/4), R).
export function localToDeformed(x, y, z, R) {
  const k = Math.PI / 4.0;
  const wx = R * Math.tan((x / R) * k);
  const wy = R * Math.tan((y / R) * k);
  const inv = (z + R) / Math.sqrt(wx * wx + wy * wy + R * R);
  return [wx * inv, wy * inv, R * inv];
}
