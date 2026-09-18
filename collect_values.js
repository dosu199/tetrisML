/**
 * collect_values.js - dataset for the value-network variant of Phase 2.
 *
 *   node collect_values.js --games 40 --cap 300 --samples 6
 *
 * Where collect.js asks "which move did the expert pick?", this asks "what did
 * the expert think of this board?" Each row is one *afterstate* - a board as it
 * would look after some legal placement - and the label is the expert's own
 * evaluation score for it.
 *
 * We sample several candidate placements per decision, not just the chosen
 * one, because the network needs to see bad boards to learn they are bad. A
 * dataset of only chosen moves contains almost exclusively good positions, and
 * a model trained on it has no idea what it is avoiding.
 */

import { writeFileSync, mkdirSync, readFileSync, existsSync } from 'node:fs';
import { dirname } from 'node:path';
import {
  createBoard,
  Bag,
  PIECES,
  placePiece,
  clearLines,
  copyBoard,
  placements,
  mulberry32,
} from './src/tetris-core.js';
import { chooseMove, evaluate, PUBLISHED_WEIGHTS } from './src/tetris-ai.js';
import { buildValueInput, valueColumns } from './src/tetris-policy.js';

const argv = process.argv.slice(2);
const flag = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i === -1 ? d : argv[i + 1];
};

const games = Number(flag('games', 40));
const cap = Number(flag('cap', 300));
const samples = Number(flag('samples', 6));
const outFile = flag('out', 'data/values.csv');

let weights = PUBLISHED_WEIGHTS;
if (existsSync('weights.json')) weights = JSON.parse(readFileSync('weights.json', 'utf8')).weights;

const rand = mulberry32(99);
const rows = [];
const started = Date.now();

for (let g = 0; g < games; g++) {
  const board = createBoard();
  const bag = new Bag(30000 + g);
  let current = bag.next();
  let next = bag.next();

  for (let p = 0; p < cap; p++) {
    const legal = placements(board, current);
    if (legal.length === 0) break;

    const chosen = chooseMove(board, current, next, weights, { lookahead: false });
    if (chosen === null) break;

    // The chosen placement, plus a random handful of the alternatives.
    const picks = [chosen];
    for (let s = 0; s < samples - 1 && legal.length > 0; s++) {
      picks.push(legal[Math.floor(rand() * legal.length)]);
    }

    for (const m of picks) {
      const after = copyBoard(board);
      placePiece(after, PIECES[current].rotations[m.rotation], m.row, m.col);
      const lines = clearLines(after);
      const target = evaluate(after, lines, weights);
      const x = buildValueInput(after, lines);
      rows.push(`${Array.from(x, (v) => (Number.isInteger(v) ? v : v.toFixed(0))).join(',')},${target.toFixed(6)}`);
    }

    placePiece(board, PIECES[current].rotations[chosen.rotation], chosen.row, chosen.col);
    clearLines(board);
    current = next;
    next = bag.next();
  }

  if ((g + 1) % 10 === 0) {
    process.stdout.write(`\r  ${g + 1}/${games} games, ${rows.length} rows, ${((Date.now() - started) / 1000).toFixed(0)}s   `);
  }
}

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, `${valueColumns().join(',')},target\n${rows.join('\n')}\n`);

console.log(`\n\nrows    ${rows.length.toLocaleString()}`);
console.log(`file    ${outFile} (${(Buffer.byteLength(rows.join('\n')) / 1e6).toFixed(1)} MB)`);
console.log(`\nnext: python3 train_value.py --data ${outFile}\n`);
