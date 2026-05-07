import * as THREE from 'three';
import { Grass } from './grass.js';
import { Rocks } from './rocks.js';
import { Clouds } from './clouds.js';
import { TerrainSystem } from './terrain.js';

export class Environment extends THREE.Object3D {
  constructor(renderer) {
    super();

    this.terrainSystem = new TerrainSystem(renderer);
    this.add(this.terrainSystem.skyMesh);
    this.add(this.terrainSystem.terrainMesh);
    this.add(this.terrainSystem.waterMesh);

    // Sun + ambient lights for trees/grass MeshPhongMaterials. Position is
    // derived from terrain's uTimeOfDay so sky and lighting stay in sync.
    this.sun = new THREE.DirectionalLight(0xffe5b0, 5);
    this.sun.userData.dynamic = true;       // sun moves per frame — exclude from freezeStatics
    this.sun.castShadow = true;
    this.sun.shadow.camera.left = -100;
    this.sun.shadow.camera.right = 100;
    this.sun.shadow.camera.top = 100;
    this.sun.shadow.camera.bottom = -100;
    this.sun.shadow.mapSize = new THREE.Vector2(512, 512);
    this.sun.shadow.bias = -0.001;
    this.sun.shadow.normalBias = 0.2;
    this.add(this.sun);
    this.ambient = new THREE.AmbientLight(0xffffff, 0.4);
    this.add(this.ambient);

    const tu = this.terrainSystem.uniforms;
    this.skybox = {
      get sunAzimuth() { return tu.uTimeOfDay.value * 360; },
      set sunAzimuth(deg) { tu.uTimeOfDay.value = ((deg % 360) + 360) % 360 / 360; },
    };

    this.grass = new Grass(undefined, this.terrainSystem);
    this.add(this.grass);

    this.rocks = new Rocks();
    this.add(this.rocks);

    this.clouds = new Clouds();
    this.clouds.position.set(0, 200, 0);
    this.clouds.rotation.x = Math.PI / 2;
    this.clouds.visible = false;
    this.add(this.clouds);
  }

  update(elapsedTime, camera, centerXZ) {
    this.grass.update(elapsedTime, camera, centerXZ);
    this.rocks.update(elapsedTime, camera);
    this.clouds.update(elapsedTime);
    if (camera) this.terrainSystem.update(camera, elapsedTime);
    // Sun direction follows terrain's uTimeOfDay (0..1 → full diurnal cycle).
    const tod = this.terrainSystem.uniforms.uTimeOfDay.value;
    const el = Math.sin((tod - 0.25) * 2 * Math.PI);
    const az = tod * 2 * Math.PI;
    const horiz = Math.sqrt(Math.max(1 - el * el, 0));
    const dist = 200;
    this.sun.position.set(Math.cos(az) * horiz * dist, Math.max(0.05, el) * dist, Math.sin(az) * horiz * dist);
    // Dim ambient at night so the scene actually goes dark; bright at day.
    const dayAmt = Math.max(0, Math.min(1, (el + 0.05) / 0.35));
    this.sun.intensity = 8 * dayAmt + 0.5;
    this.ambient.intensity = 0.7 * dayAmt + 0.2;
  }

  renderSky(renderer) {
    this.terrainSystem.renderSky(renderer);
  }
}