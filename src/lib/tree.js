import * as THREE from 'three';
import RNG from './rng';
import { Branch } from './branch';
import { Billboard, TreeType } from './enums';
import TreeOptions from './options';
import { loadPreset } from './presets/index';
import { getBarkTexture, getLeafTexture } from './textures';
import { Trellis } from './trellis';

// Single shared unit-quad geometry for all leaves across all trees. The
// per-instance origin/orientation/scale is encoded into instanceMatrix —
// the geometry itself is just a 4-vertex / 6-index quad.
let _unitLeafGeo = null;
function _getUnitLeafGeometry() {
  if (_unitLeafGeo) return _unitLeafGeo;
  const g = new THREE.BufferGeometry();
  // Match the original generateLeaf corner layout (W=L=1):
  //   v[0]=(-W/2, L, 0)  uv (0,1)
  //   v[1]=(-W/2, 0, 0)  uv (0,0)
  //   v[2]=( W/2, 0, 0)  uv (1,0)
  //   v[3]=( W/2, L, 0)  uv (1,1)
  const positions = new Float32Array([
    -0.5, 1, 0,
    -0.5, 0, 0,
    0.5, 0, 0,
    0.5, 1, 0,
  ]);
  const uvs = new Float32Array([0, 1, 0, 0, 1, 0, 1, 1]);
  const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);
  g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
  g.setIndex(new THREE.BufferAttribute(indices, 1));
  // Front-facing normal (z+); shader runs with side: DoubleSide so back faces
  // get flipped lighting via the standard pipeline.
  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
  g.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  g.computeBoundingSphere();
  // Boost bounding sphere so frustum culling doesn't drop tilted instances.
  g.boundingSphere.radius = 1.5;
  _unitLeafGeo = g;
  return _unitLeafGeo;
}

// Compose the per-instance leaf transform: rotate by ryRot around local Y,
// rotate by parent orientation Euler, translate by origin, uniform scale.
const _tmpEuler = new THREE.Euler();
const _tmpQ = new THREE.Quaternion();
const _tmpQy = new THREE.Quaternion();
const _tmpQfinal = new THREE.Quaternion();
const _tmpV = new THREE.Vector3();
const _tmpScale = new THREE.Vector3();
const _yAxis = new THREE.Vector3(0, 1, 0);
function _composeLeafMatrix(out, inst) {
  _tmpEuler.set(inst.ex, inst.ey, inst.ez);
  _tmpQ.setFromEuler(_tmpEuler);
  _tmpQy.setFromAxisAngle(_yAxis, inst.ryRot);
  _tmpQfinal.multiplyQuaternions(_tmpQ, _tmpQy);
  _tmpV.set(inst.ox, inst.oy, inst.oz);
  _tmpScale.set(inst.size, inst.size, inst.size);
  out.compose(_tmpV, _tmpQfinal, _tmpScale);
}

// Module-level material caches. Sharing materials across Tree instances cuts
// allocation cost from O(trees) to O(distinct_visible_configs) and lets the
// per-frame uTime write target one shader handle per material instead of one
// per tree. The cache key is the *visible* identity: same key => identical
// pixel output, so sharing is visually equivalent.
const _leafMatCache = new Map();
const _branchMatCache = new Map();
// Active leaf shader handles (populated on first material compile by Three).
// Updated en masse via Tree.updateAllShaders(t) — O(distinct_materials).
const _leafShaders = new Set();

function _leafKey(o) {
  return [o.type, o.tint, o.alphaTest, !!o.textured].join('|');
}
function _branchKey(o) {
  return [o.type, o.tint, !!o.flatShading, !!o.textured, o.textureScale].join('|');
}

export class Tree extends THREE.Group {
  /** Single per-frame call updates all shared leaf shaders. */
  static updateAllShaders(elapsedTime) {
    for (const sh of _leafShaders) {
      if (sh && sh.uniforms && sh.uniforms.uTime) sh.uniforms.uTime.value = elapsedTime;
    }
  }
  static get sharedLeafMaterialCount() { return _leafMatCache.size; }
  static get sharedLeafShaderCount() { return _leafShaders.size; }
  static get sharedBranchMaterialCount() { return _branchMatCache.size; }

  /**
   * Consolidate all leaf InstancedMeshes inside `rootGroup` into one big
   * InstancedMesh per leaf material. Cuts per-tree draw call count from
   * O(trees) to O(distinct_leaf_materials). Tree branches stay per-tree
   * (they're unique geometry) but leaves merge across the whole forest.
   * After this call, each tree's leavesMesh is removed from the scene.
   */
  static consolidateLeaves(rootGroup) {
    // Collect (worldMatrix, leavesMesh, instances) for every tree in the group.
    const buckets = new Map(); // material -> Matrix4[] in rootGroup-local space
    rootGroup.updateMatrixWorld(true);
    const rootInverse = new THREE.Matrix4().copy(rootGroup.matrixWorld).invert();
    const treesToClean = [];
    rootGroup.traverse((o) => {
      if (o instanceof Tree) treesToClean.push(o);
    });
    const tmpM = new THREE.Matrix4();
    for (const tree of treesToClean) {
      const leafMesh = tree.leavesMesh;
      if (!leafMesh || !leafMesh.isInstancedMesh) continue;
      const mat = leafMesh.material;
      // Tree-local instance matrix -> world -> rootGroup-local
      tree.updateMatrixWorld(true);
      const treeWorld = tree.matrixWorld;
      const localToRoot = new THREE.Matrix4().multiplyMatrices(rootInverse, treeWorld);
      const list = buckets.get(mat) || [];
      const inst = leafMesh.instanceMatrix;
      const count = leafMesh.count;
      for (let i = 0; i < count; i++) {
        leafMesh.getMatrixAt(i, tmpM);
        const world = new THREE.Matrix4().multiplyMatrices(localToRoot, tmpM);
        list.push(world);
      }
      buckets.set(mat, list);
      // Remove the per-tree leaves entirely.
      tree.remove(leafMesh);
      tree.leavesMesh = null;
    }
    const geom = _getUnitLeafGeometry();
    const consolidated = [];
    for (const [mat, mats] of buckets.entries()) {
      const merged = new THREE.InstancedMesh(geom, mat, mats.length);
      merged.name = 'leaves-merged';
      merged.castShadow = true;
      merged.receiveShadow = true;
      merged.frustumCulled = false;
      for (let i = 0; i < mats.length; i++) merged.setMatrixAt(i, mats[i]);
      merged.instanceMatrix.needsUpdate = true;
      rootGroup.add(merged);
      consolidated.push(merged);
    }
    return { meshes: consolidated, totalInstances: Array.from(buckets.values()).reduce((s,a)=>s+a.length,0) };
  }

  /**
   * Merge per-tree branch meshes (regular non-instanced THREE.Mesh objects)
   * into a small set of merged BufferGeometries grouped by material. Cuts
   * forest branch draw calls from O(trees) to O(distinct_branch_materials).
   * Each tree's branchesMesh is removed from the scene; its geometry data is
   * baked into world space so the merged mesh sits at the rootGroup origin.
   */
  static consolidateBranches(rootGroup) {
    rootGroup.updateMatrixWorld(true);
    const rootInverse = new THREE.Matrix4().copy(rootGroup.matrixWorld).invert();
    const trees = [];
    rootGroup.traverse((o) => { if (o instanceof Tree) trees.push(o); });
    // material -> { positions:[], normals:[], uvs:[], windFactors:[], indices:[] }
    const buckets = new Map();
    let nextIndexBase = new Map(); // material -> running base
    for (const tree of trees) {
      const bm = tree.branchesMesh;
      if (!bm || !bm.geometry || !bm.geometry.attributes || !bm.geometry.attributes.position) continue;
      const mat = bm.material;
      tree.updateMatrixWorld(true);
      const local = new THREE.Matrix4().multiplyMatrices(rootInverse, tree.matrixWorld);
      const normalLocal = new THREE.Matrix3().getNormalMatrix(local);
      let bucket = buckets.get(mat);
      if (!bucket) { bucket = { pos:[], nrm:[], uvs:[], wind:[], idx:[], base:0 }; buckets.set(mat, bucket); }
      const g = bm.geometry;
      const posAttr = g.attributes.position;
      const nrmAttr = g.attributes.normal;
      const uvAttr  = g.attributes.uv;
      const windAttr = g.attributes.windFactor;
      const idxAttr = g.index;
      const v = new THREE.Vector3();
      const n = new THREE.Vector3();
      const base = bucket.base;
      for (let i = 0; i < posAttr.count; i++) {
        v.fromBufferAttribute(posAttr, i).applyMatrix4(local);
        bucket.pos.push(v.x, v.y, v.z);
        if (nrmAttr) {
          n.fromBufferAttribute(nrmAttr, i).applyMatrix3(normalLocal).normalize();
          bucket.nrm.push(n.x, n.y, n.z);
        }
        if (uvAttr) bucket.uvs.push(uvAttr.getX(i), uvAttr.getY(i));
        if (windAttr) bucket.wind.push(windAttr.getX(i));
      }
      if (idxAttr) {
        for (let i = 0; i < idxAttr.count; i++) bucket.idx.push(idxAttr.getX(i) + base);
      } else {
        for (let i = 0; i < posAttr.count; i++) bucket.idx.push(i + base);
      }
      bucket.base = base + posAttr.count;
      // Detach per-tree branch mesh
      tree.remove(bm);
      bm.geometry.dispose();
      tree.branchesMesh = null;
    }
    const merged = [];
    for (const [mat, b] of buckets.entries()) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(b.pos, 3));
      if (b.nrm.length) geom.setAttribute('normal', new THREE.Float32BufferAttribute(b.nrm, 3));
      if (b.uvs.length) geom.setAttribute('uv', new THREE.Float32BufferAttribute(b.uvs, 2));
      if (b.wind.length) geom.setAttribute('windFactor', new THREE.Float32BufferAttribute(b.wind, 1));
      geom.setIndex(b.idx.length > 65535 ? new THREE.Uint32BufferAttribute(b.idx, 1) : new THREE.Uint16BufferAttribute(b.idx, 1));
      geom.computeBoundingSphere();
      const m = new THREE.Mesh(geom, mat);
      m.name = 'branches-merged';
      m.castShadow = true;
      m.receiveShadow = true;
      rootGroup.add(m);
      merged.push(m);
    }
    return { meshes: merged, materials: buckets.size };
  }

  /**
   * @type {RNG}
   */
  rng;

  /**
   * @type {TreeOptions}
   */
  options;

  /**
   * @type {Branch[]}
   */
  branchQueue = [];

  /**
   * @param {TreeOptions} params
   */
  constructor(options = new TreeOptions()) {
    super();
    this.name = 'Tree';
    this.branchesMesh = new THREE.Mesh();
    // leavesMesh is rebuilt as an InstancedMesh on every generate() —
    // start as a placeholder so .remove()/.add() still work uniformly.
    this.leavesMesh = new THREE.Object3D();
    this.trellisMesh = null;
    this.add(this.branchesMesh);
    this.add(this.leavesMesh);
    this.options = options;
  }

  update(elapsedTime) {
    // Materials are shared across trees; uTime is written once per material
    // via Tree.updateAllShaders. Keep this as a no-op for API compatibility.
  }

  /**
   * Loads a preset tree from JSON 
   * @param {string} preset 
   */
  loadPreset(name) {
    const json = loadPreset(name);
    this.loadFromJson(json);
  }

  /**
   * Loads a tree from JSON
   * @param {TreeOptions} json 
   */
  loadFromJson(json) {
    this.options.copy(json);
    this.generate();
  }

  /**
   * Generate a new tree
   */
  generate() {
    // Clean up old geometry
    this.branches = {
      verts: [],
      normals: [],
      indices: [],
      uvs: [],
      windFactor: []
    };

    // Instance descriptors: each entry produces one InstancedMesh slot.
    // Two-sided billboard => emit two entries per leaf with rotation offset.
    this.leaves = { instances: [] };

    this.rng = new RNG(this.options.seed);

    // Create the trunk of the tree first
    this.branchQueue.push(
      new Branch(
        new THREE.Vector3(),
        new THREE.Euler(),
        this.options.branch.length[0],
        this.options.branch.radius[0],
        0,
        this.options.branch.sections[0],
        this.options.branch.segments[0],
      ),
    );

    while (this.branchQueue.length > 0) {
      const branch = this.branchQueue.shift();
      this.generateBranch(branch);
    }

    this.createBranchesGeometry();
    this.createLeavesGeometry();
    this.createTrellis();
  }

  /**
   * Generates a new branch
   * @param {Branch} branch
   * @returns
   */
  generateBranch(branch) {
    // Used later for geometry index generation
    const indexOffset = this.branches.verts.length / 3;

    let sectionOrientation = branch.orientation.clone();
    let sectionOrigin = branch.origin.clone();
    let sectionLength =
      branch.length /
      branch.sectionCount /
      (this.options.type === 'Deciduous' ? this.options.branch.levels - 1 : 1);

    // This information is used for generating child branches after the branch
    // geometry has been constructed
    let sections = [];

    for (let i = 0; i <= branch.sectionCount; i++) {
      let sectionRadius = branch.radius;

      // If final section of final level, set radius to effecively zero
      if (
        i === branch.sectionCount &&
        branch.level === this.options.branch.levels
      ) {
        sectionRadius = 0.001;
      } else if (this.options.type === TreeType.Deciduous) {
        sectionRadius *=
          1 - this.options.branch.taper[branch.level] * (i / branch.sectionCount);
      } else if (this.options.type === TreeType.Evergreen) {
        // Evergreens do not have a terminal branch so they have a taper of 1
        sectionRadius *= 1 - (i / branch.sectionCount);
      }

      // Create the segments that make up this section.
      let first;
      for (let j = 0; j < branch.segmentCount; j++) {
        let angle = (2.0 * Math.PI * j) / branch.segmentCount;

        // Create the segment vertex
        const vertex = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle))
          .multiplyScalar(sectionRadius)
          .applyEuler(sectionOrientation)
          .add(sectionOrigin);

        const normal = new THREE.Vector3(Math.cos(angle), 0, Math.sin(angle))
          .applyEuler(sectionOrientation)
          .normalize();

        const uv = new THREE.Vector2(
          j / branch.segmentCount,
          (i % 2 === 0) ? 0 : 1,
        );

        this.branches.verts.push(...Object.values(vertex));
        this.branches.normals.push(...Object.values(normal));
        this.branches.uvs.push(...Object.values(uv));

        if (j === 0) {
          first = { vertex, normal, uv };
        }
      }

      // Duplicate the first vertex so there is continuity in the UV mapping
      this.branches.verts.push(...Object.values(first.vertex));
      this.branches.normals.push(...Object.values(first.normal));
      this.branches.uvs.push(1, first.uv.y);

      // Use this information later on when generating child branches
      sections.push({
        origin: sectionOrigin.clone(),
        orientation: sectionOrientation.clone(),
        radius: sectionRadius,
      });

      sectionOrigin.add(
        new THREE.Vector3(0, sectionLength, 0).applyEuler(sectionOrientation),
      );

      // Perturb the orientation of the next section randomly. The higher the
      // gnarliness, the larger potential perturbation
      const gnarliness =
        Math.max(1, 1 / Math.sqrt(sectionRadius)) *
        this.options.branch.gnarliness[branch.level];

      sectionOrientation.x += this.rng.random(gnarliness, -gnarliness);
      sectionOrientation.z += this.rng.random(gnarliness, -gnarliness);

      // Apply growth force to the branch
      const qSection = new THREE.Quaternion().setFromEuler(sectionOrientation);

      const qTwist = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        this.options.branch.twist[branch.level],
      );

      const qForce = new THREE.Quaternion().setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3().copy(this.options.branch.force.direction),
      );

      qSection.multiply(qTwist);
      qSection.rotateTowards(
        qForce,
        this.options.branch.force.strength / sectionRadius,
      );

      // Apply trellis force if enabled
      if (this.options.trellis.enabled) {
        const trellisResult = this.calculateTrellisForce(sectionOrigin, sectionRadius);
        if (trellisResult) {
          const qTrellis = new THREE.Quaternion().setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            trellisResult.direction,
          );
          qSection.rotateTowards(qTrellis, trellisResult.strength);
        }
      }

      sectionOrientation.setFromQuaternion(qSection);
    }

    this.generateBranchIndices(indexOffset, branch);

    // Deciduous trees have a terminal branch that grows out of the
    // end of the parent branch
    if (this.options.type === 'deciduous') {
      const lastSection = sections[sections.length - 1];

      if (branch.level < this.options.branch.levels) {
        this.branchQueue.push(
          new Branch(
            lastSection.origin,
            lastSection.orientation,
            this.options.branch.length[branch.level + 1],
            lastSection.radius,
            branch.level + 1,
            // Section count and segment count must be same as parent branch
            // since the child branch is growing from the end of the parent branch
            branch.sectionCount,
            branch.segmentCount,
          ),
        );
      } else {
        this.generateLeaf(lastSection.origin, lastSection.orientation);
      }
    }

    // If we are on the last branch level, generate leaves
    if (branch.level === this.options.branch.levels) {
      this.generateLeaves(sections);
    } else if (branch.level < this.options.branch.levels) {
      this.generateChildBranches(
        this.options.branch.children[branch.level],
        branch.level + 1,
        sections);
    }
  }

  /**
   * Generate branches from a parent branch
   * @param {number} count The number of child branches to generate
   * @param {number} level The level of the child branches
   * @param {{
   *  origin: THREE.Vector3,
   *  orientation: THREE.Euler,
   *  radius: number
   * }[]} sections The parent branch's sections
   * @returns
   */
  generateChildBranches(count, level, sections) {
    const radialOffset = this.rng.random();

    for (let i = 0; i < count; i++) {
      // Determine how far along the length of the parent branch the child
      // branch should originate from (0 to 1)
      let childBranchStart = this.rng.random(1.0, this.options.branch.start[level]);

      // Find which sections are on either side of the child branch origin point
      // so we can determine the origin, orientation and radius of the branch
      const sectionIndex = Math.floor(childBranchStart * (sections.length - 1));
      let sectionA, sectionB;
      sectionA = sections[sectionIndex];
      if (sectionIndex === sections.length - 1) {
        sectionB = sectionA;
      } else {
        sectionB = sections[sectionIndex + 1];
      }

      // Find normalized distance from section A to section B (0 to 1)
      const alpha =
        (childBranchStart - sectionIndex / (sections.length - 1)) /
        (1 / (sections.length - 1));

      // Linearly interpolate origin from section A to section B
      const childBranchOrigin = new THREE.Vector3().lerpVectors(
        sectionA.origin,
        sectionB.origin,
        alpha,
      );

      // Linearly interpolate radius
      const childBranchRadius =
        this.options.branch.radius[level] *
        ((1 - alpha) * sectionA.radius + alpha * sectionB.radius);

      // Linearlly interpolate the orientation
      const qA = new THREE.Quaternion().setFromEuler(sectionA.orientation);
      const qB = new THREE.Quaternion().setFromEuler(sectionB.orientation);
      const parentOrientation = new THREE.Euler().setFromQuaternion(
        qB.slerp(qA, alpha),
      );

      // Calculate the angle offset from the parent branch and the radial angle
      const radialAngle = 2.0 * Math.PI * (radialOffset + i / count);
      const q1 = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(1, 0, 0),
        this.options.branch.angle[level] / (180 / Math.PI),
      );
      const q2 = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        radialAngle,
      );
      const q3 = new THREE.Quaternion().setFromEuler(parentOrientation);

      const childBranchOrientation = new THREE.Euler().setFromQuaternion(
        q3.multiply(q2.multiply(q1)),
      );

      let childBranchLength =
        this.options.branch.length[level] *
        (this.options.type === TreeType.Evergreen
          ? 1.0 - childBranchStart
          : 1.0);

      this.branchQueue.push(
        new Branch(
          childBranchOrigin,
          childBranchOrientation,
          childBranchLength,
          childBranchRadius,
          level,
          this.options.branch.sections[level],
          this.options.branch.segments[level],
        ),
      );
    }
  }

  /**
   * Logic for spawning child branches from a parent branch's section
   * @param {{
  *  origin: THREE.Vector3,
  *  orientation: THREE.Euler,
  *  radius: number
  * }[]} sections The parent branch's sections
  * @returns
  */
  generateLeaves(sections) {
    const radialOffset = this.rng.random();

    for (let i = 0; i < this.options.leaves.count; i++) {
      // Determine how far along the length of the parent
      // branch the leaf should originate from (0 to 1)
      let leafStart = this.rng.random(1.0, this.options.leaves.start);

      // Find which sections are on either side of the child branch origin point
      // so we can determine the origin, orientation and radius of the branch
      const sectionIndex = Math.floor(leafStart * (sections.length - 1));
      let sectionA, sectionB;
      sectionA = sections[sectionIndex];
      if (sectionIndex === sections.length - 1) {
        sectionB = sectionA;
      } else {
        sectionB = sections[sectionIndex + 1];
      }

      // Find normalized distance from section A to section B (0 to 1)
      const alpha =
        (leafStart - sectionIndex / (sections.length - 1)) /
        (1 / (sections.length - 1));

      // Linearly interpolate origin from section A to section B
      const leafOrigin = new THREE.Vector3().lerpVectors(
        sectionA.origin,
        sectionB.origin,
        alpha,
      );

      // Linearlly interpolate the orientation
      const qA = new THREE.Quaternion().setFromEuler(sectionA.orientation);
      const qB = new THREE.Quaternion().setFromEuler(sectionB.orientation);
      const parentOrientation = new THREE.Euler().setFromQuaternion(
        qB.slerp(qA, alpha),
      );

      // Calculate the angle offset from the parent branch and the radial angle
      const radialAngle = 2.0 * Math.PI * (radialOffset + i / this.options.leaves.count);
      const q1 = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(1, 0, 0),
        this.options.leaves.angle / (180 / Math.PI),
      );
      const q2 = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0),
        radialAngle,
      );
      const q3 = new THREE.Quaternion().setFromEuler(parentOrientation);

      const leafOrientation = new THREE.Euler().setFromQuaternion(
        q3.multiply(q2.multiply(q1)),
      );

      this.generateLeaf(leafOrigin, leafOrientation);
    }
  }

  /**
  * Generates a leaves
  * @param {THREE.Vector3} origin The starting point of the branch
  * @param {THREE.Euler} orientation The starting orientation of the branch
  */
  generateLeaf(origin, orientation) {
    const leafSize =
      this.options.leaves.size *
      (1 +
        this.rng.random(
          this.options.leaves.sizeVariance,
          -this.options.leaves.sizeVariance,
        ));

    // Each instance carries origin, orientation Euler, an extra Y rotation
    // (for the two-sided crossed-quad billboard), and uniform scale.
    this.leaves.instances.push({
      ox: origin.x, oy: origin.y, oz: origin.z,
      ex: orientation.x, ey: orientation.y, ez: orientation.z,
      ryRot: 0,
      size: leafSize,
    });
    if (this.options.leaves.billboard === Billboard.Double) {
      this.leaves.instances.push({
        ox: origin.x, oy: origin.y, oz: origin.z,
        ex: orientation.x, ey: orientation.y, ez: orientation.z,
        ryRot: Math.PI / 2,
        size: leafSize,
      });
    }
  }

  /**
   * Generates the indices for branch geometry
   * @param {Branch} branch
   */
  generateBranchIndices(indexOffset, branch) {
    // Build geometry each section of the branch (cylinder without end caps)
    let v1, v2, v3, v4;
    const N = branch.segmentCount + 1;
    for (let i = 0; i < branch.sectionCount; i++) {
      // Build the quad for each segment of the section
      for (let j = 0; j < branch.segmentCount; j++) {
        v1 = indexOffset + i * N + j;
        // The last segment wraps around back to the starting segment, so omit j + 1 term
        v2 = indexOffset + i * N + (j + 1);
        v3 = v1 + N;
        v4 = v2 + N;
        this.branches.indices.push(v1, v3, v2, v2, v3, v4);
      }
    }
  }

  /**
   * Generates the geometry for the branches
   */
  createBranchesGeometry() {
    const g = new THREE.BufferGeometry();
    g.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(this.branches.verts), 3),
    );
    g.setAttribute(
      'normal',
      new THREE.BufferAttribute(new Float32Array(this.branches.normals), 3),
    );
    g.setAttribute(
      'uv',
      new THREE.BufferAttribute(new Float32Array(this.branches.uvs), 2),
    );
    g.setIndex(
      new THREE.BufferAttribute(new Uint16Array(this.branches.indices), 1),
    );
    g.computeBoundingSphere();

    const bk = _branchKey(this.options.bark);
    let mat = _branchMatCache.get(bk);
    if (!mat) {
      mat = new THREE.MeshPhongMaterial({
        name: 'branches',
        flatShading: this.options.bark.flatShading,
        color: new THREE.Color(this.options.bark.tint),
      });
      if (this.options.bark.textured) {
        mat.aoMap = getBarkTexture(this.options.bark.type, 'ao', this.options.bark.textureScale);
        mat.map = getBarkTexture(this.options.bark.type, 'color', this.options.bark.textureScale);
        mat.normalMap = getBarkTexture(this.options.bark.type, 'normal', this.options.bark.textureScale);
        mat.roughnessMap = getBarkTexture(this.options.bark.type, 'roughness', this.options.bark.textureScale);
      }
      _branchMatCache.set(bk, mat);
    }

    this.branchesMesh.geometry.dispose();
    this.branchesMesh.geometry = g;
    // Do NOT dispose cached material — it's shared across trees.
    this.branchesMesh.material = mat;
    this.branchesMesh.castShadow = true;
    this.branchesMesh.receiveShadow = true;
  }

  /**
   * Generates the InstancedMesh for the leaves. One unit-quad geometry,
   * per-instance origin/orientation/scale carried in instanceMatrix.
   */
  createLeavesGeometry() {
    // Single quad in local space — corners match the original generateLeaf
    // layout: bottom-left/right at y=0, top-left/right at y=1, x in [-0.5, 0.5].
    // Per-instance scale lifts these to the leaf's actual world size.
    const g = _getUnitLeafGeometry();

    const lk = _leafKey(this.options.leaves);
    let mat = _leafMatCache.get(lk);
    if (!mat) {
      mat = this._buildLeafMaterial();
      _leafMatCache.set(lk, mat);
    }
    // Replace the existing leavesMesh with a fresh InstancedMesh sized to
    // current instance count. The old mesh (if any) is removed from this Tree.
    if (this.leavesMesh) {
      this.remove(this.leavesMesh);
      // Don't dispose the unit geometry (shared) and don't dispose the cached
      // material — both are deliberately re-used across all trees.
    }
    const count = this.leaves.instances.length;
    const inst = new THREE.InstancedMesh(g, mat, count);
    inst.name = 'leaves';
    const m = new THREE.Matrix4();
    for (let i = 0; i < count; i++) {
      _composeLeafMatrix(m, this.leaves.instances[i]);
      inst.setMatrixAt(i, m);
    }
    inst.instanceMatrix.needsUpdate = true;
    inst.castShadow = true;
    inst.receiveShadow = true;
    inst.frustumCulled = false; // Wind sway can push verts outside the static bound
    this.leavesMesh = inst;
    this.add(inst);
  }

  /** Build a fresh leaf material with the wind-sway onBeforeCompile hook. */
  _buildLeafMaterial() {
    const mat = new THREE.MeshPhongMaterial({
      name: 'leaves',
      map: getLeafTexture(this.options.leaves.type),
      color: new THREE.Color(this.options.leaves.tint),
      emissive: new THREE.Color(0x0a0e08),
      emissiveIntensity: 0.10,
      side: THREE.DoubleSide,
      alphaTest: this.options.leaves.alphaTest,
      dithering: true
    });

    // Add custom shader code for branch swaying
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uTime = { value: 0 };
      shader.uniforms.uWindStrength = { value: new THREE.Vector3(0.5, 0, 0.5) };
      shader.uniforms.uWindFrequency = { value: 0.5 };
      shader.uniforms.uWindScale = { value: 70 };

      // Wrap-shading hemisphere term + subsurface transmission so leaf quads
      // catch light from any angle AND glow softly when sun is behind them
      // (light passing through the leaf, like a real translucent leaf in sun).
      const phongPars = THREE.ShaderChunk.lights_phong_pars_fragment.replace(
        'float dotNL = saturate( dot( geometryNormal, directLight.direction ) );',
        `float dotNLraw = dot(geometryNormal, directLight.direction);
         float dotNL = pow(saturate(dotNLraw*0.5+0.5), 1.6);
         // Translucent transmission: when sun is BEHIND the leaf (dotNLraw<0),
         // add diffuse-tinted contribution proportional to how much sun is
         // hitting the back side. Falls off with view angle so it reads as
         // soft glow, not flat fill.
         float transmit = max(-dotNLraw, 0.0);
         transmit = pow(transmit, 1.5);                                  // softer ramp`
      );
      // Inject transmission into the irradiance accumulation. The Phong RE_Direct
      // multiplies BRDF_Lambert(diffuseColor) by lightColor*dotNL — we add a
      // separate transmission term that adds tinted light passing through.
      const phongFragChunk = THREE.ShaderChunk.lights_phong_pars_fragment.includes('RE_Direct_BlinnPhong')
        ? phongPars.replace(
            'reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );',
            `reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
             // Transmitted light: leaf-color tinted, attenuated, additive
             reflectedLight.directDiffuse += directLight.color * material.diffuseColor * transmit * 0.42;`
          )
        : phongPars;
      shader.fragmentShader = shader.fragmentShader.replace(
        '#include <lights_phong_pars_fragment>',
        phongFragChunk
      );

      shader.vertexShader = `
        uniform float uTime;
        uniform vec3 uWindStrength;
        uniform float uWindFrequency;
        uniform float uWindScale;
        ` + shader.vertexShader;

      // Add code for simplex noise
      shader.vertexShader = shader.vertexShader.replace(
        `void main() {`,
        `
        // GLSL Simplex Noise 3D
        // Source: https://github.com/ashima/webgl-noise

        vec3 mod289(vec3 x) {
            return x - floor(x * (1.0 / 289.0)) * 289.0;
        }

        vec4 mod289(vec4 x) {
            return x - floor(x * (1.0 / 289.0)) * 289.0;
        }

        vec4 permute(vec4 x) {
            return mod289(((x*34.0)+1.0)*x);
        }

        vec4 taylorInvSqrt(vec4 r) {
            return 1.79284291400159 - 0.85373472095314 * r;
        }

        vec3 fade(vec3 t) {
            return t*t*t*(t*(t*6.0-15.0)+10.0);
        }

        // Classic Simplex Noise 3D
        float simplex3(vec3 v) {
            const vec2  C = vec2(1.0/6.0, 1.0/3.0);
            const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

            // First corner
            vec3 i  = floor(v + dot(v, C.yyy) );
            vec3 x0 = v - i + dot(i, C.xxx);

            // Other corners
            vec3 g = step(x0.yzx, x0.xyz);
            vec3 l = 1.0 - g;
            vec3 i1 = min( g.xyz, l.zxy );
            vec3 i2 = max( g.xyz, l.zxy );

            //  x0 = x0 - 0. + 0.0 * C 
            vec3 x1 = x0 - i1 + C.xxx;
            vec3 x2 = x0 - i2 + C.yyy; // 2.0 * C.x = 1/3 = C.y
            vec3 x3 = x0 - D.yyy;      // -1.0 + 3.0 * C.x = -0.5

            // Permutations
            i = mod289(i);
            vec4 p = permute( permute( permute( 
                        i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
                      + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
                      + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

            // Gradients: 7x7 points over a square, mapped onto an octahedron.
            // The ring size 17*17 = 289 is close to the mapping's singularity.
            float n_ = 0.142857142857; // 1.0/7.0
            vec3  ns = n_ * D.wyz - D.xzx;

            vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,7*7)

            vec4 x_ = floor(j * ns.z);
            vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

            vec4 x = x_ *ns.x + ns.yyyy;
            vec4 y = y_ *ns.x + ns.yyyy;
            vec4 h = 1.0 - abs(x) - abs(y);

            vec4 b0 = vec4( x.xy, y.xy );
            vec4 b1 = vec4( x.zw, y.zw );

            vec4 s0 = floor(b0)*2.0 + 1.0;
            vec4 s1 = floor(b1)*2.0 + 1.0;
            vec4 sh = -step(h, vec4(0.0));

            vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
            vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

            vec3 g0 = vec3(a0.xy,h.x);
            vec3 g1 = vec3(a0.zw,h.y);
            vec3 g2 = vec3(a1.xy,h.z);
            vec3 g3 = vec3(a1.zw,h.w);

            // Normalise gradients
            vec4 norm = taylorInvSqrt(vec4(dot(g0,g0), dot(g1,g1), dot(g2,g2), dot(g3,g3)));
            g0 *= norm.x;
            g1 *= norm.y;
            g2 *= norm.z;
            g3 *= norm.w;

            // Mix contributions from the four corners
            vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
            m = m * m;
            return 42.0 * dot( m*m, vec4( dot(g0,x0), dot(g1,x1), 
                                          dot(g2,x2), dot(g3,x3) ) );
        }
          
        void main() {`,
      );

      shader.vertexShader = shader.vertexShader.replace(
        `#include <project_vertex>`,
        `
        vec4 mvPosition = vec4(transformed, 1.0);
        #ifdef USE_INSTANCING
          mvPosition = instanceMatrix * mvPosition;
        #endif

        // Sample wind noise after instance placement so leaves at different
        // origins get different sway phase — matches the world-space layout.
        float windOffset = 2.0 * 3.14 * simplex3(mvPosition.xyz / uWindScale);
        vec3 windSway = uv.y * uWindStrength * (
          0.5 * sin(uTime * uWindFrequency + windOffset) +
          0.3 * sin(2.0 * uTime * uWindFrequency + 1.3 * windOffset) +
          0.2 * sin(5.0 * uTime * uWindFrequency + 1.5 * windOffset)
        );
        mvPosition.xyz += windSway;

        mvPosition = modelViewMatrix * mvPosition;
        gl_Position = projectionMatrix * mvPosition;
        `
      );

      mat.userData.shader = shader;
      _leafShaders.add(shader);
    };

    return mat;
  }

  /**
   * Create or update the trellis geometry
   */
  createTrellis() {
    // Remove old trellis if exists
    if (this.trellisMesh) {
      this.remove(this.trellisMesh);
      this.trellisMesh.dispose();
      this.trellisMesh = null;
    }

    // Create new trellis if enabled and visible
    if (this.options.trellis.enabled && this.options.trellis.visible) {
      this.trellisMesh = new Trellis(this.options.trellis);
      this.trellisMesh.generate();
      this.add(this.trellisMesh);
    }
  }

  /**
   * Find the nearest point on the trellis grid to a given position
   * @param {THREE.Vector3} position
   * @returns {THREE.Vector3}
   */
  getNearestTrellisPoint(position) {
    const t = this.options.trellis;
    const trellisX = t.position.x;
    const trellisY = t.position.y;
    const trellisZ = t.position.z;

    // Trellis bounds
    const minX = trellisX - t.width / 2;
    const maxX = trellisX + t.width / 2;
    const minY = trellisY;
    const maxY = trellisY + t.height;

    // Clamp position to trellis bounds for projection
    const clampedX = Math.max(minX, Math.min(maxX, position.x));
    const clampedY = Math.max(minY, Math.min(maxY, position.y));

    // Find nearest horizontal line (Y = constant)
    const nearestHLineY = Math.round((clampedY - minY) / t.spacing) * t.spacing + minY;
    const finalHLineY = Math.max(minY, Math.min(maxY, nearestHLineY));

    // Find nearest vertical line (X = constant)
    const nearestVLineX = Math.round((clampedX - minX) / t.spacing) * t.spacing + minX;
    const finalVLineX = Math.max(minX, Math.min(maxX, nearestVLineX));

    // Point on nearest horizontal line (X can vary along the line)
    const pointOnHLine = new THREE.Vector3(clampedX, finalHLineY, trellisZ);

    // Point on nearest vertical line (Y can vary along the line)
    const pointOnVLine = new THREE.Vector3(finalVLineX, clampedY, trellisZ);

    // Return whichever is closer
    const distH = position.distanceTo(pointOnHLine);
    const distV = position.distanceTo(pointOnVLine);

    return distH < distV ? pointOnHLine : pointOnVLine;
  }

  /**
   * Calculate the force vector toward the nearest trellis point
   * @param {THREE.Vector3} position Current section position
   * @param {number} radius Current section radius
   * @returns {{ direction: THREE.Vector3, strength: number } | null}
   */
  calculateTrellisForce(position, radius) {
    const trellis = this.options.trellis;
    const nearestPoint = this.getNearestTrellisPoint(position);

    const distance = position.distanceTo(nearestPoint);

    // Only apply force within max distance
    if (distance > trellis.force.maxDistance) return null;
    if (distance < 0.001) return null; // Avoid division by zero

    // Calculate direction toward trellis
    const direction = new THREE.Vector3()
      .subVectors(nearestPoint, position)
      .normalize();

    // Calculate strength with distance falloff
    // Closer = stronger force, scaled by inverse radius (like existing force)
    const distanceFactor = 1 - Math.pow(
      distance / trellis.force.maxDistance,
      trellis.force.falloff,
    );
    const strength = trellis.force.strength * distanceFactor / radius;

    return { direction, strength };
  }

  get vertexCount() {
    // Leaves are instanced: 4 verts per instance.
    return this.branches.verts.length / 3 + this.leaves.instances.length * 4;
  }

  get triangleCount() {
    // 2 triangles per leaf instance.
    return this.branches.indices.length / 3 + this.leaves.instances.length * 2;
  }
}
