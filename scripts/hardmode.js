/**
 * hardmode.js - the search bot vs the DQN network, on the same games.
 *
 *   node scripts/hardmode.js                                   # the plain modes, 10 games
 *   node scripts/hardmode.js --games 50 --seed 90000           # a decisive test
 *   node scripts/hardmode.js --mode normal,garbage3 --model models/dqn-model.json,other.json
 *
 * Modes (rules), from easiest to hardest:
 *
 *   normal                 7-bag pieces, the next piece is visible
 *   no-preview             7-bag, the next piece is NOT visible
 *   memoryless             independent random pieces, next piece visible
 *   memoryless-no-preview  independent random pieces, no preview
 *   garbage5               7-bag, preview, a garbage row rises every 5 pieces
 *   garbage4, garbage3     the same, a garbage row every 4 / every 3 pieces
 *
 * All but garbage4 and garbage3 run by default; --mode picks modes by name.
 *
 * The bot uses the next piece when it can see it (2-ply search). The network
 * decides on the current piece alone (1-ply) unless you add:
 *
 *   --lookahead   also play each network with 2-ply search (sees the next piece)
 *   --deep        also play each network with 3-ply search (beam search)
 *   --no-bot      leave the search bot out
 *   --hold        everyone may use the hold box
 *   --workers N   games run in parallel on N CPU cores (default: all cores).
 *                 The results are identical to --workers 1, only faster.
 *   --weights F   the bot's weights (default models/weights.json)
 *
 * Per player: pieces survived (median and mean), lines, tetrises per 100
 * pieces, and how many games ended in a top-out before the cap. Every player
 * plays the same seeds, so the same pieces and the same garbage.
 */

import { readFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { playEpisode } from '../src/tetris-core.js';
import { chooseMoveAction, PUBLISHED_WEIGHTS } from '../src/tetris-ai.js';
import { loadDqn, chooseDqnAction } from '../src/tetris-dqn.js';
import { parseFlags, oneOf, fail, STRING } from './lib/cli.js';
import { SEED_BASE } from './lib/seeds.js';
import { median, sum } from './lib/stats.js';

// Workers re-run this file; they parse the same command line as the main thread.
const argv = isMainThread ? process.argv.slice(2) : workerData.argv;
const opts = parseFlags(
  {
    games: 10,
    cap: 2000,
    seed: SEED_BASE.quickBenchmark,
    mode: STRING,
    model: 'models/dqn-model.json',
    weights: 'models/weights.json',
    workers: availableParallelism(),
    hold: false,
    lookahead: false,
    deep: false,
    'no-bot': false,
  },
  { argv }
);

const MODES = {
  normal: { randomizer: 'bag', preview: true, garbageEvery: 0 },
  'no-preview': { randomizer: 'bag', preview: false, garbageEvery: 0 },
  memoryless: { randomizer: 'memoryless', preview: true, garbageEvery: 0 },
  'memoryless-no-preview': { randomizer: 'memoryless', preview: false, garbageEvery: 0 },
  garbage5: { randomizer: 'bag', preview: true, garbageEvery: 5 },
  garbage4: { randomizer: 'bag', preview: true, garbageEvery: 4, onlyByName: true },
  garbage3: { randomizer: 'bag', preview: true, garbageEvery: 3, onlyByName: true },
};
const modes = opts.mode
  ? opts.mode.split(',').map((m) => oneOf(m, 'mode', Object.keys(MODES)))
  : Object.keys(MODES).filter((m) => !MODES[m].onlyByName);
const modelFiles = opts.model.split(',').filter(Boolean);

/* ---------------------------------------------------------------- players */

function loadBotWeights(file) {
  try {
    return JSON.parse(readFileSync(file, 'utf8')).weights;
  } catch (err) {
    if (err.code === 'ENOENT') return PUBLISHED_WEIGHTS; // no trained weights: the published ones
    throw err;
  }
}

/** A short, unique name per model file: its file name, or its path if two share one. */
function modelNames(files) {
  const short = files.map((f) => f.replace(/\.json$/, '').replace(/^.*\//, ''));
  return short.map((name, i) => (short.indexOf(name) !== short.lastIndexOf(name) ? files[i].replace(/\.json$/, '') : name));
}

function makePlayers() {
  const players = {};
  if (!opts.noBot) {
    const weights = loadBotWeights(opts.weights);
    players.bot = (b, c, n, h) => chooseMoveAction(b, c, n, weights, h, { lookahead: n != null });
  }
  const names = modelNames(modelFiles);
  modelFiles.forEach((file, i) => {
    const net = loadDqn(JSON.parse(readFileSync(file, 'utf8')));
    const name = names[i];
    // 1-ply: decides on the current piece; with hold it may still use the
    // preview to judge what holding would bring in
    players[name] = (b, c, n, h) => chooseDqnAction(b, c, h ? n : null, net, h);
    if (opts.lookahead) players[`${name}+2ply`] = (b, c, n, h) => chooseDqnAction(b, c, n, net, h, { lookahead: true });
    if (opts.deep) players[`${name}+3ply`] = (b, c, n, h) => chooseDqnAction(b, c, n, net, h, { deep: true });
  });
  return players;
}

const players = makePlayers();

function runGames(mode, name, seeds) {
  const { onlyByName, ...rules } = MODES[mode];
  return seeds.map((seed) => playEpisode(players[name], seed, { cap: opts.cap, ...rules, hold: opts.hold }));
}

/* ----------------------------------------------- worker: run what it's sent */

if (!isMainThread) {
  parentPort.on('message', ({ id, mode, name, seeds }) => {
    parentPort.postMessage({ id, results: runGames(mode, name, seeds) });
  });
}

/* ------------------------------------------------------------- main thread */

/** A pool of `size` workers (none for size 1: games then run on this thread). */
function createPool(size) {
  const workers = [];
  const waiting = new Map();
  let nextId = 0;
  if (size > 1) {
    for (let w = 0; w < size; w++) {
      const worker = new Worker(new URL(import.meta.url), { workerData: { argv } });
      worker.on('message', ({ id, results }) => {
        waiting.get(id)(results);
        waiting.delete(id);
      });
      worker.on('error', (e) => fail(e.stack));
      workers.push(worker);
    }
  }
  const run = (worker, task) =>
    new Promise((resolve) => {
      const id = nextId++;
      waiting.set(id, resolve);
      worker.postMessage({ id, ...task });
    });
  return { workers, run, close: () => workers.forEach((w) => w.terminate()) };
}

/**
 * Every seed of one (mode, player) row, spread over the workers. Each worker
 * gets every N-th seed, so slow and fast games mix evenly; the results come
 * back in seed order, identical to running them on one thread.
 */
async function runRow(pool, mode, name) {
  const seeds = Array.from({ length: opts.games }, (_, g) => opts.seed + g);
  const n = pool.workers.length;
  if (n === 0) return runGames(mode, name, seeds);
  const parts = await Promise.all(
    pool.workers.map((worker, w) => pool.run(worker, { mode, name, seeds: seeds.filter((_, g) => g % n === w) }))
  );
  const out = new Array(opts.games);
  parts.forEach((part, w) => part.forEach((r, k) => (out[w + k * n] = r)));
  return out;
}

async function main() {
  const { games, cap } = opts;
  const pool = createPool(Math.max(1, Math.min(opts.workers, games)));
  const nameWidth = Math.max(16, ...Object.keys(players).map((n) => n.length));
  const cores = pool.workers.length || 1;

  console.log(`\nbenchmark: ${opts.noBot ? '' : 'bot vs '}${modelFiles.join(', ')}`);
  console.log(
    `${games} games per player per mode, seeds ${opts.seed}+, cap ${cap} pieces, hold ${opts.hold ? 'ON' : 'off'}, ` +
      `${cores} core${cores > 1 ? 's' : ''}\n`
  );
  console.log(
    `  mode                    ${'player'.padEnd(nameWidth)}    pieces med   pieces mean   lines mean   tetris/100   topped out   time` +
      (opts.hold ? '   holds/100' : '')
  );
  console.log(
    `  ----------------------  ${'-'.repeat(nameWidth)}    ----------   -----------   ----------   ----------   ----------   ----` +
      (opts.hold ? '   ---------' : '')
  );

  for (const mode of modes) {
    for (const name of Object.keys(players)) {
      const t0 = Date.now();
      const r = await runRow(pool, mode, name);
      const pieces = r.map((x) => x.pieces);
      const totalPieces = Math.max(1, sum(pieces));
      const per100 = (key) => (100 * sum(r.map((x) => x[key]))) / totalPieces;
      console.log(
        `  ${mode.padEnd(22)}  ${name.padEnd(nameWidth)}    ${String(median(pieces)).padStart(10)}   ` +
          `${(sum(pieces) / games).toFixed(1).padStart(11)}   ` +
          `${(sum(r.map((x) => x.lines)) / games).toFixed(1).padStart(10)}   ` +
          `${per100('tetrises').toFixed(2).padStart(10)}   ` +
          `${`${r.filter((x) => x.died).length}/${games}`.padStart(10)}   ${((Date.now() - t0) / 1000).toFixed(0).padStart(3)}s` +
          (opts.hold ? `   ${per100('holds').toFixed(1).padStart(9)}` : '')
      );
    }
  }
  console.log('');
  pool.close();
}

if (isMainThread) await main();
