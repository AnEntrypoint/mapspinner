// mapspinner SDK entry point
// Primary exports for external consumers (e.g., spoint integration)

export { initMapspinnerPlanet, initMapspinnerRender } from './planet-orchestrator.js';
export { Quadtree } from './quadtree.js';
export { createAnchorField } from './anchor-field.js';

// Version marker
export const VERSION = '0.1.0';

/**
 * Initialize mapspinner renderer on a WebGL2 context.
 * 
 * @param {WebGL2RenderingContext} gl - WebGL2 context (must support required extensions)
 * @param {Object} config - Configuration object
 * @param {number} config.radius - Planet radius in meters (default 6360000)
 * @param {number} config.gridMeshSize - Mesh subdivision level (default 16)
 * @param {HTMLCanvasElement} config.canvas - Target canvas element
 * @returns {Promise<Object>} - Renderer instance with methods: render(cam), dispose()
 * 
 * Example:
 *   const renderer = await mapspinner.createRenderer(gl, { radius: 6360000 });
 *   // In animation loop:
 *   renderer.render(camera);
 */
export async function createRenderer(gl, config = {}) {
  const { initMapspinnerRender } = await import('./gl-render.js');
  return initMapspinnerRender(gl, config);
}

/**
 * Initialize a full planet+camera+LOD system.
 * 
 * @param {WebGL2RenderingContext} gl - WebGL2 context
 * @param {Object} config - Configuration object
 * @returns {Promise<Object>} - Planet instance with properties: cam, quadtree, render(gl)
 */
export async function createPlanet(gl, config = {}) {
  const { initMapspinnerPlanet } = await import('./planet-orchestrator.js');
  return initMapspinnerPlanet(gl, config);
}
