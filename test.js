/**
 * hardmode.js - the search bot vs the DQN network, on the same games.
 *
 *   node hardmode.js                                   # all modes, 10 games
 *   node hardmode.js --games 50 --seed 90000           # a decisive test
 *   node hardmode.js --mode normal --model dqn-model.json,other.json
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
 * The plain modes run by default; name a garbage4/garbage3 mode with --mode.
 *
 * The bot uses the next piece when it can see it (2-ply search). The DQN, like
 * the reference implementation, decides on the current piece alone unless you
 * add --lookahead, which also runs it with 2-ply search.
 *
 *   --lookahead   also play each network with 2-ply search (sees the next piece)
 *   --no-bot      leave the search bot out
 *   --hold        everyone may use the hold box
 *   --workers N   games run in parallel on N CPU cores (default: all cores).
 *                 The results are identical to --workers 1, only faster.
 *
 * Per player: pieces survived (median and mean), lines, tetrises per 100
 * pieces, and how many games ended in a top-out before the cap. Every player
 * plays the same seeds, so the same pieces and the same garbage.
 */

import { readFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { playEpisode } from './src/tetris-core.js';
import { chooseMoveAction, PUBLISHED_WEIGHTS } from './src/tetris-ai.js';
import { loadDqn, chooseDqnAction } from './src/tetris-dqn.js';

const argv = isMainThread ? process.argv.slice(2) : workerData.argv;
const flag = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i === -1 ? d : argv[i + 1];
};

const games = Number(flag('games', 10));
const cap = Number(flag('cap', 2000));
const seedBase = Number(flag('seed', 20000));
const only = flag('mode', null);
// --hold: every player may use the hold box (modern Tetris rule)
const HOLD_ON = argv.includes('--hold');
const WORKERS = Math.max(1, Number(flag('workers', availableParallelism())));
const modelFiles = flag('model', 'dqn-model.json').split(',').filter(Boolean);

const MODES = {
  normal: { randomizer: 'bag', preview: true, garbageEvery: 0 },
  'no-preview': { randomizer: 'bag', preview: false, garbageEvery: 0 },
  memoryless: { randomizer: 'memoryless', preview: true, garbageEvery: 0 },
  'memoryless-no-preview': { randomizer: 'memoryless', preview: false, garbageEvery: 0 },
  garbage5: { randomizer: 'bag', preview: true, garbageEvery: 5 },
  garbage4: { randomizer: 'bag', preview: true, garbageEvery: 4, onlyByName: true },
  garbage3: { randomizer: 'bag', preview: true, garbageEvery: 3, onlyByName: true },
};

/* ---------------------------------------------------------------- players */

function makePlayers() {
  let weights = PUBLISHED_WEIGHTS;
  try {
    weights = JSON.parse(readFileSync('weights.json', 'utf8')).weights;
  } catch {}

  const players = {};
  // --no-bot skips the search bot (handy when only comparing networks)
  if (!argv.includes('--no-bot')) {
    players.bot = (b, c, n, h) => {
      const m = chooseMoveAction(b, c, n, weights, h, { lookahead: n != null });
      return m && m.score !== -Infinity ? m : null;
    };
  }
  for (const file of modelFiles) {
    const net = loadDqn(JSON.parse(readFileSync(file, 'utf8')));
    const name = file.replace(/\.json$/, '').replace(/^.*\//, '');
    // 1-ply: decides on the current piece; with hold it may still use the
    // preview to judge what holding would bring in
    players[name] = (b, c, n, h) => chooseDqnAction(b, c, net, h ? n : null, h);
    // --lookahead: also run each network with 2-ply search (uses the preview)
    if (argv.includes('--lookahead')) {
      players[`${name}+2ply`] = (b, c, n, h) => chooseDqnAction(b, c, net, n, h, { lookahead: true });
    }
  }
  return players;
}

const players = makePlayers();

function runGames(mode, name, seeds) {
  const { onlyByName, ...opts } = MODES[mode];
  return seeds.map((seed) => playEpisode(players[name], seed, { cap, ...opts, hold: HOLD_ON }));
}

/* ----------------------------------------------- worker: run what it's sent */

if (!isMainThread) {
  parentPort.on('message', ({ id, mode, name, seeds }) => {
    parentPort.postMessage({ id, results: runGames(mode, name, seeds) });
  });
}

/* ------------------------------------------------------------- main thread */

if (isMainThread) {
  const pool = [];
  const waiting = new Map();
  let nextId = 0;
  if (WORKERS > 1) {
    for (let w = 0; w < WORKERS; w++) {
      const worker = new Worker(new URL(import.meta.url), { workerData: { argv } });
      worker.on('message', ({ id, results }) => {
        waiting.get(id)(results);
        waiting.delete(id);
      });
      worker.on('error', (e) => {
        console.error(e);
        process.exit(1);
      });
      pool.push(worker);
    }
  }

  // Every seed of one (mode, player) row, spread over the workers. Each
  // worker gets every WORKERS-th seed, so slow and fast games mix evenly.
  async function runRow(mode, name) {
    const seeds = Array.from({ length: games }, (_, g) => seedBase + g);
    if (pool.length === 0) return runGames(mode, name, seeds);
    const parts = await Promise.all(
      pool.map(
        (worker, w) =>
          new Promise((resolve) => {
            const id = nextId++;
            waiting.set(id, resolve);
            worker.postMessage({ id, mode, name, seeds: seeds.filter((_, g) => g % pool.length === w) });
          })
      )
    );
    // put the results back in seed order
    const out = new Array(games);
    parts.forEach((part, w) => part.forEach((r, k) => (out[w + k * pool.length] = r)));
    return out;
  }

  const median = (xs) => {
    const s = [...xs].sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  const sum = (xs) => xs.reduce((a, b) => a + b, 0);

  console.log(`\nbenchmark: bot vs ${modelFiles.join(', ')}`);
  console.log(
    `${games} games per player per mode, seeds ${seedBase}+, cap ${cap} pieces, hold ${HOLD_ON ? 'ON' : 'off'}, ` +
      `${pool.length || 1} core${pool.length > 1 ? 's' : ''}\n`
  );
  console.log(
    '  mode                    player              pieces med   pieces mean   lines mean   tetris/100   topped out   time' +
      (HOLD_ON ? '   holds/100' : '')
  );
  console.log(
    '  ----------------------  ----------------    ----------   -----------   ----------   ----------   ----------   ----' +
      (HOLD_ON ? '   ---------' : '')
  );

  for (const [mode, opts] of Object.entries(MODES)) {
    if (only ? mode !== only : opts.onlyByName) continue;
    for (const name of Object.keys(players)) {
      const t0 = Date.now();
      const r = await runRow(mode, name);
      const pcs = r.map((x) => x.pieces);
      const totalPieces = Math.max(1, sum(pcs));
      console.log(
        `  ${mode.padEnd(22)}  ${name.padEnd(16)}    ${String(median(pcs)).padStart(10)}   ` +
          `${(sum(pcs) / games).toFixed(1).padStart(11)}   ` +
          `${(sum(r.map((x) => x.lines)) / games).toFixed(1).padStart(10)}   ` +
          `${((100 * sum(r.map((x) => x.tetrises))) / totalPieces).toFixed(2).padStart(10)}   ` +
          `${`${r.filter((x) => x.died).length}/${games}`.padStart(10)}   ${((Date.now() - t0) / 1000).toFixed(0).padStart(3)}s` +
          (HOLD_ON ? `   ${((100 * sum(r.map((x) => x.holds))) / totalPieces).toFixed(1).padStart(9)}` : '')
      );
    }
  }
  console.log('');
  for (const worker of pool) worker.terminate();
}