/**
 * train_many.js - train several networks with different seeds, keep the best.
 *
 *   node train_many.js                          # seeds 1-4, 2 at a time
 *   node train_many.js --seeds 1,2,3,4,5,6 --parallel 3
 *   node train_many.js --seeds 1,2 --pieces mix  # extra flags go to train_dqn.js
 *
 * WHY
 * ---
 * The same training code with a different random seed can give a network that
 * loses 16 of 50 games or one that loses 33. That spread is bigger than most
 * of the improvements we tried. So one training run proves little. Train
 * several and pick the best, the way you would pick a model on a validation
 * set.
 *
 * THREE SETS OF GAMES, NEVER MIXED
 * --------------------------------
 *   500000+  the check inside training (picks each run's best checkpoint)
 *    70000+  this script's selection test (picks the best run)
 *    90000+  the final benchmark in hardmode.js (reports how good it is)
 *
 * If we picked the winner on the 90000+ games and then reported its score on
 * those same games, the score would be flattering: we would have chosen the
 * run that happened to suit them. A separate selection set avoids that. It is
 * the train / validation / test split from the course.
 *
 * Each run is saved as runs/seed-N.json. The current --out file (default
 * dqn-model.json) takes part in the selection test too, and it is replaced
 * only if a new run beats it. So running this can never make your model worse.
 */

import { spawn } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { playEpisode } from './src/tetris-core.js';
import { loadDqn, chooseDqnMove } from './src/tetris-dqn.js';

const argv = process.argv.slice(2);
const own = new Set(['seeds', 'parallel', 'out', 'select-games', 'select-cap']);
const flag = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i === -1 ? d : argv[i + 1];
};
// everything that is not ours is passed through to train_dqn.js
const passThrough = [];
for (let i = 0; i < argv.length; i++) {
  const name = argv[i].replace(/^--/, '');
  if (own.has(name)) i++;
  else passThrough.push(argv[i]);
}

const SEEDS = flag('seeds', '1,2,3,4').split(',').map(Number);
const PARALLEL = Number(flag('parallel', 2));
const OUT = flag('out', 'dqn-model.json');
const SELECT_GAMES = Number(flag('select-games', 50));
const SELECT_CAP = Number(flag('select-cap', 2000));
const PIECES = flag('pieces', 'random');
const RANDOMIZER = PIECES === 'bag' ? 'bag' : 'memoryless';

mkdirSync('runs', { recursive: true });

function train(seed) {
  return new Promise((resolve, reject) => {
    const file = `runs/seed-${seed}.json`;
    const args = ['train_dqn.js', '--seed', String(seed), '--out', file, ...passThrough];
    const child = spawn(process.execPath, args, { stdio: ['ignore', 'pipe', 'inherit'] });
    let last = '';
    child.stdout.on('data', (d) => {
      for (const line of d.toString().split('\n')) {
        if (line.includes('<- best') || line.startsWith('best network')) last = line.trim();
        if (line.startsWith('best network')) console.log(`  seed ${seed}: ${line.trim()}`);
      }
    });
    child.on('close', (code) => (code === 0 ? resolve(file) : reject(new Error(`seed ${seed} failed (${last})`))));
  });
}

function select(file) {
  const net = loadDqn(JSON.parse(readFileSync(file, 'utf8')));
  let lost = 0;
  let pieces = 0;
  for (let g = 0; g < SELECT_GAMES; g++) {
    const r = playEpisode((b, c) => chooseDqnMove(b, c, net), 70000 + g, {
      cap: SELECT_CAP,
      randomizer: RANDOMIZER,
      preview: false,
    });
    if (r.died) lost++;
    pieces += r.pieces;
  }
  return { file, lost, mean: pieces / SELECT_GAMES };
}

const t0 = Date.now();
console.log(`\ntraining ${SEEDS.length} networks (seeds ${SEEDS.join(', ')}), ${PARALLEL} at a time`);
console.log(`train_dqn.js flags: ${passThrough.join(' ') || '(defaults)'}\n`);

// a tiny worker pool: PARALLEL trainings at once
const queue = [...SEEDS];
const files = [];
await Promise.all(
  Array.from({ length: Math.min(PARALLEL, SEEDS.length) }, async () => {
    while (queue.length) files.push(await train(queue.shift()));
  })
);

console.log(`\nselection test: ${SELECT_GAMES} games on seeds 70000+, ${RANDOMIZER} pieces, cap ${SELECT_CAP}\n`);
// the current model competes too, so it is only replaced by something better
const contenders = existsSync(OUT) ? [OUT, ...files.sort()] : files.sort();
const results = contenders.map(select);
results.sort((a, b) => a.lost - b.lost || b.mean - a.mean);
for (const [i, r] of results.entries()) {
  const label = r.file === OUT ? `${OUT} (current)` : r.file;
  console.log(`  ${label.padEnd(30)} lost ${String(r.lost).padStart(2)}/${SELECT_GAMES}   mean ${r.mean.toFixed(0).padStart(5)} pieces${i === 0 ? '   <- best' : ''}`);
}
const minutes = ((Date.now() - t0) / 60000).toFixed(1);
if (results[0].file === OUT) {
  console.log(`\nno new run beat the current ${OUT}; it is unchanged   (${minutes} min)`);
} else {
  copyFileSync(results[0].file, OUT);
  console.log(`\n${results[0].file} -> ${OUT}   (${minutes} min)`);
}
console.log(`final check: node hardmode.js --model ${OUT} --games 50 --seed 90000 --mode memoryless-no-preview\n`);
