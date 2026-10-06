/**
 * hardmode.js - search bot vs a network, under harder rules.
 *
 *   node hardmode.js --value cnn-model.json --games 10 --cap 2000
 *   node hardmode.js --mode no-preview --games 20
 *
 * On normal rules (7-bag + one piece of preview) both the bot and the CNN
 * survive every game we have tried, up to 5,000 pieces - so normal rules can
 * no longer tell them apart. These modes take away some of the help:
 *
 *   normal                 7-bag, sees the next piece        (the baseline)
 *   no-preview             7-bag, does NOT see the next piece
 *   memoryless             independent random pieces, sees the next piece
 *   memoryless-no-preview  independent random pieces, no preview
 *   garbage5               7-bag, preview, a garbage row every 5 pieces
 *
 * "No preview" also takes away 2-ply search for both players, since 2-ply is
 * exactly "look at the next piece". Both then decide on the current piece
 * alone.
 *
 * Reported per player: pieces survived (the real measure once games end),
 * lines, and how many games ended in a top-out before the cap.
 */

import { readFileSync } from 'node:fs';
import { playEpisode } from './src/tetris-core.js';
import { chooseMove, PUBLISHED_WEIGHTS } from './src/tetris-ai.js';
import { loadValueNet, chooseValueMove } from './src/tetris-policy.js';

const argv = process.argv.slice(2);
const flag = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i === -1 ? d : argv[i + 1];
};

const games = Number(flag('games', 10));
const cap = Number(flag('cap', 2000));
// One or more networks, comma-separated: --value cnn-hard.json,cnn-big.json
const valueFiles = flag('value', 'cnn-model.json').split(',').filter(Boolean);
const only = flag('mode', null);
// Use fresh seeds for a decisive test, so no earlier result can bias the
// choice of games. 20000+ is the default; collection used 30000-85000+.
const seedBase = Number(flag('seed', 20000));

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

// next === null when there is no preview; both choosers then search 1-ply.
const players = {
  bot: (b, c, n) => {
    const m = chooseMove(b, c, n, weights, { lookahead: n != null });
    return m && m.score !== -Infinity ? m : null;
  },
};
for (const file of valueFiles) {
  const net = loadValueNet(JSON.parse(readFileSync(file, 'utf8')));
  const name = file.replace(/\.json$/, '');
  players[name] = (b, c, n) => chooseValueMove(b, c, net, { next: n, lookahead: n != null });
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

console.log(`\nhard-mode benchmark: bot vs ${valueFiles.join(', ')}`);
console.log(`${games} games per player per mode, seeds ${seedBase}+, cap ${cap} pieces\n`);
console.log('  mode                    player        pieces med   pieces mean   lines mean   topped out   time');
console.log('  ----------------------  ----------    ----------   -----------   ----------   ----------   ----');

for (const [mode, opts] of Object.entries(MODES)) {
  if (only && mode !== only) continue;
  for (const [name, chooser] of Object.entries(players)) {
    const t0 = Date.now();
    const r = [];
    for (let g = 0; g < games; g++) r.push(playEpisode(chooser, seedBase + g, { cap, ...opts }));
    const pcs = r.map((x) => x.pieces);
    const lines = r.map((x) => x.lines);
    console.log(
      `  ${mode.padEnd(22)}  ${name.padEnd(10)}    ${String(median(pcs)).padStart(10)}   ` +
        `${(pcs.reduce((a, b) => a + b, 0) / games).toFixed(1).padStart(11)}   ` +
        `${(lines.reduce((a, b) => a + b, 0) / games).toFixed(1).padStart(10)}   ` +
        `${`${r.filter((x) => x.died).length}/${games}`.padStart(10)}   ${((Date.now() - t0) / 1000).toFixed(0)}s`
    );
  }
}
console.log('');
