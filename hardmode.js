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
 *
 * The bot uses the next piece when it can see it (2-ply search). The DQN, like
 * the reference implementation, decides on the current piece alone unless you
 * add --lookahead, which also runs it with 2-ply search.
 *
 *   --lookahead   also play each network with 2-ply search (sees the next piece)
 *   --no-bot      leave the search bot out
 *
 * Per player: pieces survived (median and mean), lines, and how many games
 * ended in a top-out before the cap. Every player plays the same seeds, so the
 * same pieces and the same garbage.
 */

import { readFileSync } from 'node:fs';
import { playEpisode } from './src/tetris-core.js';
import { chooseMove, PUBLISHED_WEIGHTS } from './src/tetris-ai.js';
import { loadDqn, chooseDqnMove } from './src/tetris-dqn.js';

const argv = process.argv.slice(2);
const flag = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i === -1 ? d : argv[i + 1];
};

const games = Number(flag('games', 10));
const cap = Number(flag('cap', 2000));
const seedBase = Number(flag('seed', 20000));
const only = flag('mode', null);
// --no-bot skips the search bot (handy when only comparing networks)
const modelFiles = flag('model', 'dqn-model.json').split(',').filter(Boolean);

const MODES = {
  normal: { randomizer: 'bag', preview: true, garbageEvery: 0 },
  'no-preview': { randomizer: 'bag', preview: false, garbageEvery: 0 },
  memoryless: { randomizer: 'memoryless', preview: true, garbageEvery: 0 },
  'memoryless-no-preview': { randomizer: 'memoryless', preview: false, garbageEvery: 0 },
  garbage5: { randomizer: 'bag', preview: true, garbageEvery: 5 },
};

let weights = PUBLISHED_WEIGHTS;
try {
  weights = JSON.parse(readFileSync('weights.json', 'utf8')).weights;
} catch {}

const players = {};
if (!argv.includes('--no-bot')) {
  players.bot = (b, c, n) => {
    const m = chooseMove(b, c, n, weights, { lookahead: n != null });
    return m && m.score !== -Infinity ? m : null;
  };
}
for (const file of modelFiles) {
  const net = loadDqn(JSON.parse(readFileSync(file, 'utf8')));
  const name = file.replace(/\.json$/, '').replace(/^.*\//, '');
  players[name] = (b, c) => chooseDqnMove(b, c, net);
  // --lookahead: also run each network with 2-ply search (uses the preview)
  if (argv.includes('--lookahead')) players[`${name}+2ply`] = (b, c, n) => chooseDqnMove(b, c, net, n, { lookahead: true });
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

console.log(`\nbenchmark: bot vs ${modelFiles.join(', ')}`);
console.log(`${games} games per player per mode, seeds ${seedBase}+, cap ${cap} pieces\n`);
console.log('  mode                    player              pieces med   pieces mean   lines mean   topped out   time');
console.log('  ----------------------  ----------------    ----------   -----------   ----------   ----------   ----');

for (const [mode, opts] of Object.entries(MODES)) {
  if (only && mode !== only) continue;
  for (const [name, chooser] of Object.entries(players)) {
    const t0 = Date.now();
    const r = [];
    for (let g = 0; g < games; g++) r.push(playEpisode(chooser, seedBase + g, { cap, ...opts }));
    const pcs = r.map((x) => x.pieces);
    const lines = r.map((x) => x.lines);
    console.log(
      `  ${mode.padEnd(22)}  ${name.padEnd(16)}    ${String(median(pcs)).padStart(10)}   ` +
        `${(pcs.reduce((a, b) => a + b, 0) / games).toFixed(1).padStart(11)}   ` +
        `${(lines.reduce((a, b) => a + b, 0) / games).toFixed(1).padStart(10)}   ` +
        `${`${r.filter((x) => x.died).length}/${games}`.padStart(10)}   ${((Date.now() - t0) / 1000).toFixed(0)}s`
    );
  }
}
console.log('');
