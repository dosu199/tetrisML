/**
 * models.js - downloads the trained models for the website. Works the same
 * from a page and from a Web Worker, because the files are found relative to
 * this module, not to the page.
 */

import { loadDqn } from '../../../src/tetris-dqn.js';
import { PUBLISHED_WEIGHTS } from '../../../src/tetris-ai.js';

const MODELS = new URL('../../../models/', import.meta.url);

/** Fetches a JSON file from models/ (e.g. 'snapshots/game-0.json'); a readable error if it fails. */
export async function fetchModel(path) {
  const res = await fetch(new URL(path, MODELS));
  if (!res.ok) throw new Error(`models/${path}: HTTP ${res.status}`);
  return res.json();
}

/**
 * The network (required: rejects if it cannot be loaded) and the search
 * bot's weights (optional: the published weights if weights.json is missing).
 * @returns {Promise<{ net: import('../../../src/tetris-dqn.js').MLP, weights: readonly number[] }>}
 */
export async function loadModels() {
  const [model, weights] = await Promise.all([
    fetchModel('dqn-model.json'),
    fetchModel('weights.json').then(
      (w) => w.weights ?? PUBLISHED_WEIGHTS,
      () => PUBLISHED_WEIGHTS
    ),
  ]);
  return { net: loadDqn(model), weights };
}
