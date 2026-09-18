/**
 * compare.js - learned weights against the published ones, head to head.
 *
 *   node compare.js                      # 30 held-out games each
 *   node compare.js --games 50 --cap 2000
 *   node compare.js --weights quick.json
 *
 * The seeds here start at 10,000 and were never used during training. This is
 * the same idea as the train/test split in your course: a model scored on the
 * data it was fitted to will always flatter itself. The number that counts is
 * the one from games neither set of weights has ever seen.
 */

import { readFileSync } from 'node:fs';
import { playGame } from './src/tetris-headless.js';
import { PUBLISHED_WEIGHTS, FEATURE_NAMES } from './src/tetris-ai.js';

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};

const games = Number(flag('games', 30));
const cap = Number(flag('cap', 2000));
const lookahead = argv.includes('--lookahead');
const weightsFile = flag('weights', 'weights.json');

let learned;
try {
  learned = JSON.parse(readFileSync(weightsFile, 'utf8')).weights;
} catch {
  console.error(`\nCould not read ${weightsFile}. Run "node train.js" first.\n`);
  process.exit(1);
}

const HELD_OUT = Array.from({ length: games }, (_, i) => 10000 + i);

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

function run(weights) {
  const lines = [];
  let deaths = 0;
  const t0 = performance.now();
  for (const seed of HELD_OUT) {
    const r = playGame(weights, seed, { maxPieces: cap, lookahead });
    lines.push(r.lines);
    if (r.toppedOut) deaths++;
  }
  return {
    median: median(lines),
    mean: lines.reduce((a, b) => a + b, 0) / lines.length,
    worst: Math.min(...lines),
    best: Math.max(...lines),
    deaths,
    seconds: (performance.now() - t0) / 1000,
  };
}

console.log(`\nhead to head`);
console.log(`------------`);
console.log(`${games} games on seeds 10000-${10000 + games - 1} (never trained on)`);
console.log(`${cap}-piece cap, ${lookahead ? '2-ply lookahead' : '1-ply'}\n`);

const a = run(PUBLISHED_WEIGHTS);
const b = run(learned);

const row = (label, r) =>
  `  ${label.padEnd(11)} ${String(r.median).padStart(7)} ${r.mean.toFixed(1).padStart(8)} ` +
  `${String(r.worst).padStart(7)} ${String(r.best).padStart(7)} ${String(r.deaths).padStart(7)}`;

console.log('               median     mean   worst    best  deaths');
console.log('              -------  -------  ------  ------  ------');
console.log(row('published', a));
console.log(row('learned', b));

const pct = (from, to) => {
  const d = ((to - from) / Math.max(Math.abs(from), 1)) * 100;
  return `${d >= 0 ? '+' : ''}${d.toFixed(1)}%`;
};

console.log('\nlearned vs published');
console.log(`  median     ${pct(a.median, b.median)}`);
console.log(`  mean       ${pct(a.mean, b.mean)}`);
console.log(`  worst game ${pct(a.worst, b.worst)}`);
console.log(`  topped out ${b.deaths} times vs ${a.deaths}`);

if (a.median === b.median || (a.deaths === 0 && b.deaths === 0)) {
  console.log(
    '\nBoth sets of weights are reaching the piece cap, so the median has\n' +
      'saturated and is no longer telling them apart. Mean, worst game and\n' +
      'deaths are the numbers that still discriminate - raise --cap until\n' +
      'games start ending on their own.'
  );
}

console.log('\nweights');
console.log('-------');
console.log('  feature            learned    published');
FEATURE_NAMES.forEach((name, i) => {
  console.log(
    `  ${name.padEnd(16)} ${learned[i].toFixed(5).padStart(9)}    ${PUBLISHED_WEIGHTS[i]
      .toFixed(5)
      .padStart(9)}`
  );
});
console.log('');
