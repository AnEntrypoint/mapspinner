// Config file for running the demo locally
import path from 'path';

/**
 * @type {import('vite').UserConfig}
 */
export default {
  build: {
    emptyOutDir: true,
    outDir: '../../dist',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: 'src/app/index.html',
        editor: 'src/app/editor.html',
      },
    },
  },
  root: './src/app',
  resolve: {
    alias: {
      'mapspinner': path.resolve(
        __dirname,
        'build/mapspinner.es.js',
      ),
    },
  },
  server: {
    hmr: true,
  },
  assetsInclude: ['**/*.frag', '**/*.vert'],
};
