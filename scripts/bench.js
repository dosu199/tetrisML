/**
 * bench.js - how fast do the players run?   npm run bench
 *
 * Speed decides what is practical: training plays thousands of games, and
 * the website must answer within one frame (16 ms) for the hint to feel live.
 * Prints games and pieces per second for the search bot and the network at
 * each search depth, on the same seeds.
 */

import { readFileSync } from 'node:fs';
import { playEpisode } from '../src/tetris-core.js';
import { chooseMove, PUBLISHED_WEIGHTS } from '../src/tetris-ai.js';
import { loadDqn, chooseDqnAction } from '../src/tetris-dqn.js';
import { parseFlags } from './lib/cli.js';
import { mean, median } from './lib/stats.js';

const opts = parseFlags({ games: 20, cap: 500 });
const net = loadDqn(JSON.parse(readFileSync(new URL('../models/dqn-model.json', import.meta.url), 'utf8')));

const PLAYERS = [
  { label: 'bot 1-ply', games: opts.games * 5, chooser: (b, c, n) => chooseMove(b, c, n, PUBLISHED_WEIGHTS, { lookahead: false }) },
  { label: 'bot 2-ply', games: opts.games, chooser: (b, c, n) => chooseMove(b, c, n, PUBLISHED_WEIGHTS, { lookahead: true }) },
  { label: 'network 1-ply', games: opts.games * 5, chooser: (b, c, n) => chooseDqnAction(b, c, n, net) },
  { label: 'network 2-ply', games: opts.games, chooser: (b, c, n) => chooseDqnAction(b, c, n, net, null, { lookahead: true }) },
  { label: 'network 3-ply', games: Math.max(1, opts.games >> 2), chooser: (b, c, n) => chooseDqnAction(b, c, n, net, null, { deep: true }) },
];

console.log(`\nbenchmark: 7-bag games capped at ${opts.cap} pieces, seeds 1+\n`);
console.log('  player           games   lines med   lines mean   topped out    games/s    pieces/s   ms/piece');
console.log('  -------------    -----   ---------   ----------   ----------    -------    --------   --------');
for (const { label, games, chooser } of PLAYERS) {
  const t0 = performance.now();
  const results = Array.from({ length: games }, (_, g) => playEpisode(chooser, 1 + g, { cap: opts.cap }));
  const seconds = (performance.now() - t0) / 1000;
  const lines = results.map((r) => r.lines);
  const pieces = results.reduce((a, r) => a + r.pieces, 0);
  console.log(
    `  ${label.padEnd(13)}    ${String(games).padStart(5)}   ${String(median(lines)).padStart(9)}   ${mean(lines).toFixed(1).padStart(10)}   ` +
      `${String(results.filter((r) => r.died).length).padStart(10)}    ${(games / seconds).toFixed(1).padStart(7)}    ` +
      `${Math.round(pieces / seconds).toString().padStart(8)}   ${((1000 * seconds) / pieces).toFixed(2).padStart(8)}`
  );
}
console.log('');
