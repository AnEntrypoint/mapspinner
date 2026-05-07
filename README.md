# MapSpinner

![GitHub Repo stars](https://img.shields.io/github/stars/AnEntrypoint/mapspinner)

Procedural tree generator + buildless Three.js demo world.
Live: https://anentrypoint.github.io/mapspinner/

MapSpinner is derived from the original EZ-Tree library by Daniel Greenheck.

# No build step

MapSpinner ships **raw ESM source**. There is no bundler. There is no `npm run build`. The `index.html` at the repo root carries an `<script type="importmap">` that resolves `three`, `three/addons/`, `three/examples/jsm/`, and `mapspinner` to vendored ESM under `vendor/three/` and `src/lib/`.

To run locally:

```bash
python -m http.server 8000
# open http://127.0.0.1:8000/
```

Any static server works (Caddy, nginx, `npx serve`, etc.). Edit a `.js` file and refresh — that is the whole dev loop.

# Installing as an npm dependency

```bash
npm install mapspinner three
```

`three` is a peer dependency. Consumers must provide it themselves (and the matching importmap entries when running buildless in the browser).

```js
import { Tree, TreePreset, BarkType, LeafType } from 'mapspinner';

const tree = new Tree();
tree.options.seed = 12345;
tree.options.trunk.length = 20;
tree.options.branch.levels = 3;
tree.generate();
scene.add(tree);
```

Any time the tree parameters are changed, call `generate()` to regenerate the geometry.

# Repo layout

```
index.html, editor.html      buildless entry points (importmap inside)
*.glb, *.png, *.jpg, ...     static assets served at page base URL
src/lib/                     npm-published library source (raw ESM)
src/app/                     demo-world source (scene, terrain, grass, UI, shaders)
vendor/three/                pinned three.js vendor (build/ + examples/jsm/)
```

# Publishing

Pushes to `main` auto-bump patch version and publish to npm via
`.github/workflows/publish.yml` (requires `NPM_TOKEN` repo secret).
Add `[skip publish]` to a commit message to skip the auto-bump.

# Tree Parameters

The `TreeOptions` class defines an options object that controls various
parameters of a procedurally generated tree. Each property allows for
customization of the tree's appearance — bark, branches, leaves.

## General

- **`seed`**: Initial value for random generation.
- **`type`**: One of the `TreeType` enumeration values (e.g., `TreeType.Deciduous`).

## Bark

- **`type`** (`BarkType.Oak`, etc.)
- **`tint`** — hex color
- **`flatShading`** — boolean
- **`textured`** — boolean
- **`textureScale`** — `{ x, y }`

## Branch

- **`levels`** — recursive branch levels
- **`angle`** — degrees
- **`children`** — count per level
- **`force`** — `{ direction: {x,y,z}, strength }`
- **`gnarliness`**, **`length`**, **`radius`**, **`sections`**, **`segments`**, **`start`**, **`taper`**, **`twist`**

## Leaves

- **`type`** (`LeafType.Oak`, etc.)
- **`billboard`** (`Billboard.Single` / `Billboard.Double`)
- **`angle`**, **`count`**, **`start`**, **`size`**, **`sizeVariance`**, **`tint`**, **`alphaTest`**
