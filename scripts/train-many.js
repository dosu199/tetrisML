/**
 * train-many.js - train several networks with different seeds, keep the best.
 *
 *   node scripts/train-many.js                          # seeds 1-4, 2 at a time
 *   node scripts/train-many.js --seeds 1,2,3,4,5,6 --parallel 3
 *   node scripts/train-many.js --seeds 1,2 --pieces mix  # extra flags go to train-dqn.js
 *
 * WHY
 * ---
 * The same training code with a different random seed can give a network that
 * loses 16 of 50 games or one that loses 33. That spread is bigger than most
 * of the improvements we tried. So one training run proves little. Train
 * several and pick the best, the way you would pick a model on a validation
 * set.
 *
 * THREE SETS OF GAMES, NEVER MIXED (scripts/lib/seeds.js)
 * --------------------------------
 *   500000+  the check inside training (picks each run's best checkpoint)
 *    70000+  this script's selection test (picks the best run)
 *    90000+  the final benchmark in hardmode.js (reports how good it is)
 *
 * If we picked the winner on the 90000+ games and then reported its score on
 * those same games, the score would be flattering: we would have chosen the
 * run that happened to suit them. A separate selection set avoids that. It is
 * the train / validation / test split of machine learning.
 *
 * Each run is saved as runs/seed-N.json. The current --out file (default
 * models/dqn-model.json) takes part in the selection test too, and it is
 * replaced only if a new run beats it. So running this can never make the
 * model worse.
 *
 * --pieces and --hold are used here (for the selection test) and also passed
 * on to train-dqn.js. --snapshots cannot be used: the parallel runs would all
 * write the same snapshot files.
 */

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { playEpisode } from '../src/tetris-core.js';
import { loadDqn, chooseDqnMove, chooseDqnAction } from '../src/tetris-dqn.js';
import { parseFlags, oneOf, numberList, fail } from './lib/cli.js';
import { SEED_BASE } from './lib/seeds.js';

const opts = parseFlags(
  {
    seeds: '1,2,3,4',
    parallel: 2,
    out: 'models/dqn-model.json',
    'select-games': 50,
    'select-cap': 2000,
    pieces: 'random', // shared with train-dqn.js
    hold: false, // shared with train-dqn.js
  },
  { passThrough: true }
);
const SEEDS = numberList(opts.seeds, 'seeds');
const OUT = opts.out;
const PIECES = oneOf(opts.pieces, 'pieces', ['bag', 'random', 'mix']);
const RANDOMIZER = PIECES === 'bag' ? 'bag' : 'memoryless';
// With --hold, ordinary games almost never end, so the selection test uses
// hold on fast garbage (a row every 3 pieces): every game ends, and the
// network that survives longest on average wins.
const SELECT_HOLD = opts.hold;
if (opts.rest.some((a) => a.startsWith('--snapshot'))) fail('--snapshots cannot be used with train-many.js (the runs would overwrite each other\'s snapshots)');
if (opts.rest.some((a) => a === '--seed' || a === '--out')) fail('train-many.js sets --seed and --out of each run itself (use --seeds)');
// what train-dqn.js gets: everything that is not ours, plus the shared flags
const trainFlags = [...opts.rest, '--pieces', PIECES, ...(opts.hold ? ['--hold'] : [])];

const TRAIN_SCRIPT = fileURLToPath(new URL('./train-dqn.js', import.meta.url));
const runFile = (seed) => `runs/seed-${seed}.json`;
const running = new Set(); // child processes, stopped if any run fails

function train(seed) {
  return new Promise((resolve, reject) => {
    const args = [TRAIN_SCRIPT, '--seed', String(seed), '--out', runFile(seed), ...trainFlags];
    const child = spawn(process.execPath, args, { stdio: ['ignore', 'pipe', 'inherit'] });
    running.add(child);
    let last = '';
    child.stdout.on('data', (d) => {
      for (const line of d.toString().split('\n')) {
        if (line.includes('<- best') || line.startsWith('best network')) last = line.trim();
        if (line.startsWith('best network')) console.log(`  seed ${seed}: ${line.trim()}`);
      }
    });
    child.on('close', (code) => {
      running.delete(child);
      if (code === 0) resolve(runFile(seed));
      else reject(new Error(`seed ${seed} failed (${last || `exit code ${code}`})`));
    });
  });
}

/** Trains every seed, `parallel` at a time. */
async function trainAll() {
  const queue = [...SEEDS];
  try {
    await Promise.all(
      Array.from({ length: Math.min(opts.parallel, SEEDS.length) }, async () => {
        while (queue.length) await train(queue.shift());
      })
    );
  } catch (err) {
    queue.length = 0;
    for (const child of running) child.kill();
    fail(err.message);
  }
}

/** How a model does on the selection games: games lost and mean pieces survived. */
function select(file) {
  const net = loadDqn(JSON.parse(readFileSync(file, 'utf8')));
  let lost = 0;
  let pieces = 0;
  for (let g = 0; g < opts.selectGames; g++) {
    const seed = SEED_BASE.selection + g;
    const r = SELECT_HOLD
      ? playEpisode((b, c, n, h) => chooseDqnAction(b, c, n, net, h), seed, {
          cap: opts.selectCap,
          randomizer: RANDOMIZER,
          garbageEvery: 3,
          hold: true,
        })
      : playEpisode((b, c) => chooseDqnMove(b, c, null, net), seed, {
          cap: opts.selectCap,
          randomizer: RANDOMIZER,
          preview: false,
        });
    if (r.died) lost++;
    pieces += r.pieces;
  }
  return { file, lost, mean: pieces / opts.selectGames };
}

async function main() {
  const t0 = Date.now();
  mkdirSync('runs', { recursive: true });
  console.log(`\ntraining ${SEEDS.length} networks (seeds ${SEEDS.join(', ')}), ${opts.parallel} at a time`);
  console.log(`train-dqn.js flags: ${trainFlags.join(' ')}\n`);
  await trainAll();

  console.log(
    `\nselection test: ${opts.selectGames} games on seeds ${SEED_BASE.selection}+, ${RANDOMIZER} pieces, cap ${opts.selectCap}` +
      `${SELECT_HOLD ? ', hold on, garbage every 3 pieces' : ''}\n`
  );
  // the current model competes too, so it is only replaced by something better
  const files = SEEDS.map(runFile);
  const contenders = existsSync(OUT) ? [OUT, ...files] : files;
  const results = contenders.map(select);
  results.sort((a, b) => a.lost - b.lost || b.mean - a.mean); // stable: on a tie the current model stays
  for (const [i, r] of results.entries()) {
    const label = r.file === OUT ? `${OUT} (current)` : r.file;
    console.log(
      `  ${label.padEnd(30)} lost ${String(r.lost).padStart(2)}/${opts.selectGames}   mean ${r.mean.toFixed(0).padStart(5)} pieces` +
        (i === 0 ? '   <- best' : '')
    );
  }
  const minutes = ((Date.now() - t0) / 60000).toFixed(1);
  if (results[0].file === OUT) {
    console.log(`\nno new run beat the current ${OUT}; it is unchanged   (${minutes} min)`);
  } else {
    copyFileSync(results[0].file, OUT);
    console.log(`\n${results[0].file} -> ${OUT}   (${minutes} min)`);
  }
  console.log(`final check: node scripts/hardmode.js --model ${OUT} --games 50 --seed ${SEED_BASE.benchmark} --mode memoryless-no-preview\n`);
}

await main();
