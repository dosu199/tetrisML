/**
 * ga-worker.js - worker thread body: scores one candidate weight vector at a
 * time. Fitness evaluation parallelises perfectly, because every candidate's
 * games are independent of every other candidate's.
 */

import { parentPort } from 'node:worker_threads';
import { fitnessOf } from './ga.js';

parentPort.on('message', ({ weights, seeds, maxPieces }) => {
  parentPort.postMessage(fitnessOf(weights, seeds, maxPieces));
});
