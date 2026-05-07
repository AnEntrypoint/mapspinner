import * as THREE from 'three';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { SMAAPass } from 'three/examples/jsm/postprocessing/SMAAPass.js';
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Tree } from 'mapspinner';
import { setupUI } from './ui.js';

async function __mapspinnerEditorBoot() {
  const container = document.getElementById('app');

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setClearColor(0x1a1a22);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(devicePixelRatio);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a22);

  const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 2000);
  camera.position.set(40, 30, 40);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 15, 0);
  controls.update();

  // Lights tuned for editor preview.
  const sun = new THREE.DirectionalLight(0xffe5b0, 4);
  sun.position.set(60, 80, 40);
  sun.castShadow = true;
  sun.shadow.camera.left = -50;
  sun.shadow.camera.right = 50;
  sun.shadow.camera.top = 50;
  sun.shadow.camera.bottom = -50;
  sun.shadow.mapSize.set(1024, 1024);
  scene.add(sun);
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));

  // Ground plane so trees cast shadows on something.
  const groundMat = new THREE.MeshPhongMaterial({ color: 0x556633 });
  const groundMesh = new THREE.Mesh(new THREE.PlaneGeometry(200, 200), groundMat);
  groundMesh.rotation.x = -Math.PI / 2;
  groundMesh.receiveShadow = true;
  scene.add(groundMesh);

  // Central editable tree.
  const tree = new Tree();
  const __preset =
    new URLSearchParams(window.location.search).get('preset') || 'Ash Medium';
  tree.loadPreset(__preset);
  tree.generate();
  tree.castShadow = true;
  tree.receiveShadow = true;
  scene.add(tree);

  // Minimal environment shim so setupUI's environment.* calls don't crash.
  const environment = {
    skybox: { sunAzimuth: 180 },
    update() {},
  };

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const smaaPass = new SMAAPass(
    container.clientWidth * renderer.getPixelRatio(),
    container.clientHeight * renderer.getPixelRatio());
  composer.addPass(smaaPass);
  composer.addPass(new OutputPass());

  const clock = new THREE.Clock();
  function animate() {
    const t = clock.getElapsedTime();
    Tree.updateAllShaders(t);
    controls.update();
    composer.render();
    requestAnimationFrame(animate);
  }

  function resize() {
    renderer.setSize(container.clientWidth, container.clientHeight);
    smaaPass.setSize(container.clientWidth, container.clientHeight);
    composer.setSize(container.clientWidth, container.clientHeight);
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);

  setupUI(tree, environment, renderer, scene, camera, controls, __preset);
  animate();
  resize();
}

__mapspinnerEditorBoot();
