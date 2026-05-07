import dts from "vite-plugin-dts";

/**
 * @type {import('vite').UserConfig}
 */
export default {
  build: {
    outDir: './build',
    lib: {
      entry: './src/lib/index.js',
      name: 'mapspinner',
      fileName: (format) => `mapspinner.${format}.js`,
    },
    rollupOptions: {
      external: ['three'],
      output: {
        globals: {
          three: 'THREE',
        },
      },
    },
    sourcemap: true,
  },
  plugins: [
    dts({
      outDir: './build',
      insertTypesEntry: true,
      rollupTypes: true,
      tsconfigPath: "./tsconfig.json"
    }),
  ],
};