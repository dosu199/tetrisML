/**
 * train.js - run the genetic algorithm and save the weights it finds.
 *
 *   node train.js                            # defaults: 50 x 40 generations
 *   node train.js --generations 60 --workers 4
 *   node train.js --population 30 --generations 15 --out quick.json
 *
 * Writes weights.json (the learned weights) and training-log.csv (one row per
 * generation, so you can plot the learning curve - which is a nice first use
 * of the matplotlib/pandas material from your course).
 */

import { writeFileSync } from 'node:fs';
import os from 'node:os';
import { runGA, DEFAULTS } from './src/ga.js';
import { FEATURE_NAMES, PUBLISHED_WEIGHTS } from './src/tetris-ai.js';
import { fitnessOf } from './src/ga.js';

/* ------------------------------------------------------------------- flags */

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};

const config = {
  populationSize: Number(flag('population', DEFAULTS.populationSize)),
  generations: Number(flag('generations', DEFAULTS.generations)),
  maxPieces: Number(flag('cap', DEFAULTS.maxPieces)),
  maxPiecesCeiling: Number(flag('ceiling', DEFAULTS.maxPiecesCeiling)),
  randomSeed: Number(flag('seed', DEFAULTS.randomSeed)),
  workers: Number(flag('workers', Math.max(1, os.cpus().length - 1))),
};
const outFile = flag('out', 'weights.json');

// Seeds 1..8 train; the high seeds in compare.js are never seen during
// training. Same discipline as a train/test split - weights that only work on
// the games they were tuned against have not learned anything general.
const TRAIN_SEEDS = DEFAULTS.seeds;

console.log('\ntraining');
console.log('--------');
console.log(`population   ${config.populationSize}`);
console.log(`generations  ${config.generations}`);
console.log(`train seeds  ${TRAIN_SEEDS.join(', ')}`);
console.log(`piece cap    ${config.maxPieces} (doubles when the population saturates)`);
console.log(`workers      ${config.workers}`);
console.log('');
console.log('  gen    best   median     mean    cap   elapsed');
console.log('  ---  ------   ------   ------   ----   -------');

/* -------------------------------------------------------------------- run */

const started = Date.now();
const log = [];

const result = await runGA(config, (row) => {
  log.push(row);
  const elapsed = ((Date.now() - started) / 1000).toFixed(0) + 's';
  console.log(
    `  ${String(row.generation).padStart(3)}  ` +
      `${row.best.toFixed(1).padStart(6)}   ` +
      `${row.median.toFixed(1).padStart(6)}   ` +
      `${row.mean.toFixed(1).padStart(6)}   ` +
      `${String(row.maxPieces).padStart(4)}   ` +
      `${elapsed.padStart(7)}` +
      (row.raised ? '   <- cap raised' : '')
  );
});

const elapsed = (Date.now() - started) / 1000;
const learned = result.best.weights;

/* ---------------------------------------------------------------- results */

console.log(`\ndone in ${elapsed.toFixed(1)}s\n`);
console.log('learned weights');
console.log('---------------');
console.log('  feature            learned    published');
FEATURE_NAMES.forEach((name, i) => {
  console.log(
    `  ${name.padEnd(16)} ${learned[i].toFixed(5).padStart(9)}    ` +
      `${PUBLISHED_WEIGHTS[i].toFixed(5).padStart(9)}`
  );
});

// A like-for-like check at the cap training ended on.
const cap = result.maxPieces;
const mine = fitnessOf(learned, TRAIN_SEEDS, cap);
const theirs = fitnessOf(PUBLISHED_WEIGHTS, TRAIN_SEEDS, cap);
console.log(`\non the training seeds at a ${cap}-piece cap:`);
console.log(`  learned    ${mine.fitness.toFixed(1)} lines`);
console.log(`  published  ${theirs.fitness.toFixed(1)} lines`);
console.log('\nrun `node compare.js` for the honest number, on seeds never trained on.');

writeFileSync(
  outFile,
  JSON.stringify(
    {
      weights: learned,
      featureNames: FEATURE_NAMES,
      trainFitness: result.best.fitness,
      finalMaxPieces: cap,
      generations: config.generations,
      populationSize: config.populationSize,
      trainSeeds: TRAIN_SEEDS,
      elapsedSeconds: elapsed,
      trainedAt: new Date().toISOString(),
    },
    null,
    2
  )
);

writeFileSync(
  'training-log.csv',
  'generation,best,median,mean,maxPieces\n' +
    log.map((r) => `${r.generation},${r.best},${r.median},${r.mean},${r.maxPieces}`).join('\n') +
    '\n'
);

console.log(`\nwrote ${outFile} and training-log.csv\n`);
