/**
 * compare.js - learned weights against the published ones, head to head.
 *
 *   node scripts/legacy-ga/compare.js                      # 30 held-out games each
 *   node scripts/legacy-ga/compare.js --games 50 --cap 2000
 *   node scripts/legacy-ga/compare.js --weights quick.json
 *
 * The seeds here start at 10,000 and were never used during training: a model
 * scored on the games it was fitted to will always flatter itself. The number
 * that counts is the one from games neither set of weights has ever seen.
 */

import { readFileSync } from 'node:fs';
import { playBotGame } from './ga.js';
import { PUBLISHED_WEIGHTS, FEATURE_NAMES } from '../../src/tetris-ai.js';
import { parseFlags, fail } from '../lib/cli.js';
import { SEED_BASE } from '../lib/seeds.js';
import { mean, median } from '../lib/stats.js';

const { games, cap, lookahead, weights: weightsFile } = parseFlags({
  games: 30,
  cap: 2000,
  lookahead: false,
  weights: 'models/weights.json',
});

let learned;
try {
  learned = JSON.parse(readFileSync(weightsFile, 'utf8')).weights;
} catch {
  fail(`Could not read ${weightsFile}. Run "node scripts/legacy-ga/train.js" first.`);
}

const HELD_OUT = Array.from({ length: games }, (_, i) => SEED_BASE.gaHeldOut + i);

function run(weights) {
  const lines = [];
  let deaths = 0;
  const t0 = performance.now();
  for (const seed of HELD_OUT) {
    const r = playBotGame(weights, seed, { maxPieces: cap, lookahead });
    lines.push(r.lines);
    if (r.toppedOut) deaths++;
  }
  return {
    median: median(lines),
    mean: mean(lines),
    worst: Math.min(...lines),
    best: Math.max(...lines),
    deaths,
    seconds: (performance.now() - t0) / 1000,
  };
}

console.log(`\nhead to head`);
console.log(`------------`);
console.log(`${games} games on seeds ${HELD_OUT[0]}-${HELD_OUT.at(-1)} (never trained on)`);
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
