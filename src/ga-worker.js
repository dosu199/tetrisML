/**
 * ga-worker.js - worker thread body. Receives one candidate, plays its games,
 * sends back the average lines cleared. Nothing clever; the parallelism is
 * worth it only because fitness evaluation is embarrassingly parallel - each
 * candidate's games are completely independent of every other candidate's.
 */

import { parentPort } from 'node:worker_threads';
import { playGame } from './tetris-headless.js';

parentPort.on('message', ({ weights, seeds, maxPieces }) => {
  let lines = 0;
  let deaths = 0;
  for (const seed of seeds) {
    const r = playGame(weights, seed, { maxPieces, lookahead: false });
    lines += r.lines;
    if (r.toppedOut) deaths++;
  }
  parentPort.postMessage({
    fitness: lines / seeds.length,
    survival: 1 - deaths / seeds.length,
  });
});
