import * as THREE from 'three';

const birchAo = new URL('./assets/bark/birch_ao_1k.jpg', import.meta.url).href;
const birchColor = new URL('./assets/bark/birch_color_1k.jpg', import.meta.url).href;
const birchNormal = new URL('./assets/bark/birch_normal_1k.jpg', import.meta.url).href;
const birchRoughness = new URL('./assets/bark/birch_roughness_1k.jpg', import.meta.url).href;

const oakAo = new URL('./assets/bark/oak_ao_1k.jpg', import.meta.url).href;
const oakColor = new URL('./assets/bark/oak_color_1k.jpg', import.meta.url).href;
const oakNormal = new URL('./assets/bark/oak_normal_1k.jpg', import.meta.url).href;
const oakRoughness = new URL('./assets/bark/oak_roughness_1k.jpg', import.meta.url).href;

const pineAo = new URL('./assets/bark/pine_ao_1k.jpg', import.meta.url).href;
const pineColor = new URL('./assets/bark/pine_color_1k.jpg', import.meta.url).href;
const pineNormal = new URL('./assets/bark/pine_normal_1k.jpg', import.meta.url).href;
const pineRoughness = new URL('./assets/bark/pine_roughness_1k.jpg', import.meta.url).href;

const willowAo = new URL('./assets/bark/willow_ao_1k.jpg', import.meta.url).href;
const willowColor = new URL('./assets/bark/willow_color_1k.jpg', import.meta.url).href;
const willowNormal = new URL('./assets/bark/willow_normal_1k.jpg', import.meta.url).href;
const willowRoughness = new URL('./assets/bark/willow_roughness_1k.jpg', import.meta.url).href;

const ashLeaves = new URL('./assets/leaves/ash_color.png', import.meta.url).href;
const aspenLeaves = new URL('./assets/leaves/aspen_color.png', import.meta.url).href;
const oakLeaves = new URL('./assets/leaves/oak_color.png', import.meta.url).href;
const pineLeaves = new URL('./assets/leaves/pine_color.png', import.meta.url).href;

const textureLoader = new THREE.TextureLoader();

/**
 * Gets a bark texture for the specified bark type
 * @param {string} barkType 
 * @param {'ao' | 'color' | 'normal' | 'roughness'} fileType 
 * @param {THREE.Vector2} scale 
 * @returns 
 */
export function getBarkTexture(barkType, fileType, scale = { x: 1, y: 1 }) {
  const texture = textures.bark[barkType][fileType];
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.x = scale.x;
  texture.repeat.y = 1 / scale.y;
  return texture;
}

/**
 * Gets the leaf texture for the specified leaf type
 * @param {string} leafType 
 * @returns 
 */
export function getLeafTexture(leafType) {
  return textures.leaves[leafType];
}

/**
 * 
 * @param {string} url Path to texture
 * @param {THREE.Vector2} scale Scale of the texture repeat
 * @param {boolean} srgb Set to true to set texture color space to SRGB
 * @returns {THREE.Texture}
 */
const loadTexture = (url, srgb = true) => {
  const texture = textureLoader.load(url);
  texture.premultiplyAlpha = true;
  if (srgb) {
    texture.colorSpace = THREE.SRGBColorSpace;
  }

  return texture;
};

const textures = {
  "bark": {
    "birch": {
      "ao": loadTexture(birchAo, false),
      "color": loadTexture(birchColor),
      "normal": loadTexture(birchNormal, false),
      "roughness": loadTexture(birchRoughness, false),
    },
    "oak": {
      "ao": loadTexture(oakAo, false),
      "color": loadTexture(oakColor),
      "normal": loadTexture(oakNormal, false),
      "roughness": loadTexture(oakRoughness, false),
    },
    "pine": {
      "ao": loadTexture(pineAo, false),
      "color": loadTexture(pineColor),
      "normal": loadTexture(pineNormal, false),
      "roughness": loadTexture(pineRoughness, false),
    },
    "willow": {
      "ao": loadTexture(willowAo, false),
      "color": loadTexture(willowColor),
      "normal": loadTexture(willowNormal, false),
      "roughness": loadTexture(willowRoughness, false),
    }
  },
  "leaves": {
    "ash": loadTexture(ashLeaves),
    "aspen": loadTexture(aspenLeaves),
    "oak": loadTexture(oakLeaves),
    "pine": loadTexture(pineLeaves)
  }
};