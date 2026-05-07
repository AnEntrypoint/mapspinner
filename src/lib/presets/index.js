const [ashSmall, ashMedium, ashLarge, aspenSmall, aspenMedium, aspenLarge, bush1, bush2, bush3, oakSmall, oakMedium, oakLarge, pineSmall, pineMedium, pineLarge, trellis] = await Promise.all([
  fetch(new URL('./ash_small.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./ash_medium.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./ash_large.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./aspen_small.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./aspen_medium.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./aspen_large.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./bush_1.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./bush_2.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./bush_3.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./oak_small.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./oak_medium.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./oak_large.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./pine_small.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./pine_medium.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./pine_large.json', import.meta.url)).then(r => r.json()),
  fetch(new URL('./trellis.json', import.meta.url)).then(r => r.json())
]);
import TreeOptions from '../options.js';

export const TreePreset = {
  'Ash Small': ashSmall,
  'Ash Medium': ashMedium,
  'Ash Large': ashLarge,
  'Aspen Small': aspenSmall,
  'Aspen Medium': aspenMedium,
  'Aspen Large': aspenLarge,
  'Bush 1': bush1,
  'Bush 2': bush2,
  'Bush 3': bush3,
  'Oak Small': oakSmall,
  'Oak Medium': oakMedium,
  'Oak Large': oakLarge,
  'Pine Small': pineSmall,
  'Pine Medium': pineMedium,
  'Pine Large': pineLarge,
  'Trellis': trellis,
};

/**
 * @param {string} name The name of the preset to load
 * @returns {TreeOptions}
 */
export function loadPreset(name) {
  const preset = TreePreset[name];
  return preset ? structuredClone(preset) : new TreeOptions();
}