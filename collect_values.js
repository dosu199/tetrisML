/**
 * collect_values.js - dataset for the value network.
 *
 *   node collect_values.js --games 60 --cap 400 --samples 6
 *   node collect_values.js --policy rank-model.json --beta 0.5 --out data/dagger1.csv
 *
 * Each row is one *afterstate* - a board as it would look after some legal
 * placement - and the label is the expert's own evaluation score for it.
 *
 * We sample several candidate placements per decision, not just the chosen
 * one, because the network needs to see bad boards to learn they are bad. A
 * dataset of only chosen moves contains almost exclusively good positions, and
 * a model trained on it has no idea what it is avoiding.
 *
 * DAGGER MODE (--policy)
 * ----------------------
 * Without --policy the expert drives, so every board in the file is a board
 * the expert would reach: tidy, flat, low. The network trained on that agrees
 * with the expert 93% of the time on those boards and under 50% on the boards
 * it reaches when it plays by itself. That gap is covariate shift.
 *
 * With --policy, the NETWORK drives some of the time and takes the game into
 * the messy positions it actually gets itself into - but every candidate board
 * is still labelled by the EXPERT. The labels stay correct; only the coverage
 * changes. Merge the result with your original data (merge_csv.js) and
 * retrain: that is one round of Dataset Aggregation.
 *
 * --beta is the chance the expert drives on any given piece. Pure network
 * driving (beta 0) tops out after a few dozen pieces and yields hopeless
 * boards; around 0.5 keeps games long while still visiting the network's own
 * mistakes. Lower it on later rounds, as the network gets better.
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
  MemorylessGen,
} from './src/tetris-core.js';
import { chooseMove, evaluate, PUBLISHED_WEIGHTS } from './src/tetris-ai.js';
import {
  buildValueInput,
  valueColumns,
  loadValueNet,
  chooseValueMove,
} from './src/tetris-policy.js';

const argv = process.argv.slice(2);
const flag = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i === -1 ? d : argv[i + 1];
};

const games = Number(flag('games', 40));
const cap = Number(flag('cap', 300));
const samples = Number(flag('samples', 6));
const outFile = flag('out', 'data/values.csv');
const seedBase = Number(flag('seed', 30000));
const policyFile = flag('policy', null);
const beta = Number(flag('beta', 0.5));
const netLookahead = argv.includes('--lookahead');
// Rule changes for hard-mode data. --memoryless swaps the 7-bag for
// independent random pieces; --no-preview hides the next piece from the
// network driver, so it plays 1-ply exactly as it would in that mode.
const memoryless = argv.includes('--memoryless');
const noPreview = argv.includes('--no-preview');

let weights = PUBLISHED_WEIGHTS;
if (existsSync('weights.json')) weights = JSON.parse(readFileSync('weights.json', 'utf8')).weights;

let net = null;
if (policyFile) {
  net = loadValueNet(JSON.parse(readFileSync(policyFile, 'utf8')));
  console.log(
    `DAgger: ${policyFile} drives ${((1 - beta) * 100).toFixed(0)}% of pieces` +
      `${netLookahead ? ' (2-ply)' : ''}, the expert labels every board`
  );
} else {
  console.log('expert drives every piece');
}
if (memoryless || noPreview) {
  console.log(`rules: ${memoryless ? 'memoryless pieces' : '7-bag'}, ${noPreview ? 'no preview' : 'preview'}`);
}

const rand = mulberry32(seedBase * 31 + 99);
const rows = [];
const started = Date.now();
let networkMoves = 0;
let expertMoves = 0;
let lost = 0;

// Every candidate board that belonged to the SAME decision shares this id.
// The regression trainer ignores it; the ranking trainer needs it, because a
// pair is only meaningful between two boards the bot was actually choosing
// between at one moment. merge_csv.js keeps ids unique across files.
let decisionId = 0;

for (let g = 0; g < games; g++) {
  const board = createBoard();
  const bag = memoryless ? new MemorylessGen(seedBase + g) : new Bag(seedBase + g);
  let current = bag.next();
  let next = bag.next();

  for (let p = 0; p < cap; p++) {
    const legal = placements(board, current);
    if (legal.length === 0) {
      lost++;
      break;
    }

    const expert = chooseMove(board, current, next, weights, { lookahead: false });
    if (expert === null) {
      lost++;
      break;
    }

    // Candidate boards to label: the expert's choice, plus random alternatives.
    const picks = [expert];
    for (let s = 0; s < samples - 1; s++) picks.push(legal[Math.floor(rand() * legal.length)]);

    for (const m of picks) {
      const after = copyBoard(board);
      placePiece(after, PIECES[current].rotations[m.rotation], m.row, m.col);
      const lines = clearLines(after);
      const target = evaluate(after, lines, weights);
      const x = buildValueInput(after, lines);
      rows.push(`${Array.from(x, (v) => v | 0).join(',')},${target.toFixed(6)},${decisionId}`);
    }
    decisionId++;

    // Who actually moves decides which boards we see next - nothing else.
    let played = expert;
    if (net && rand() >= beta) {
      played =
        chooseValueMove(board, current, net, {
          next: noPreview ? null : next,
          lookahead: netLookahead && !noPreview,
        }) ?? expert;
      networkMoves++;
    } else {
      expertMoves++;
    }

    placePiece(board, PIECES[current].rotations[played.rotation], played.row, played.col);
    clearLines(board);
    current = next;
    next = bag.next();
  }

  if ((g + 1) % 10 === 0 || g === games - 1) {
    process.stdout.write(
      `\r  ${g + 1}/${games} games, ${rows.length.toLocaleString()} rows, ` +
        `${((Date.now() - started) / 1000).toFixed(0)}s   `
    );
  }
}

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, `${valueColumns().join(',')},target,decision_id\n${rows.join('\n')}\n`);

console.log(`\n\nrows       ${rows.length.toLocaleString()}`);
console.log(`decisions  ${decisionId.toLocaleString()}`);
if (net) {
  console.log(`moves      network ${networkMoves.toLocaleString()}, expert ${expertMoves.toLocaleString()}`);
  console.log(`games lost ${lost} of ${games}  (expected: the network's mistakes are the point)`);
}
console.log(`file       ${outFile} (${(Buffer.byteLength(rows.join('\n')) / 1e6).toFixed(1)} MB)`);
console.log(
  net
    ? `\nnext: node merge_csv.js <your current training csv> ${outFile} --out data/combined-new.csv\n`
    : `\nnext: python3 train_rank.py --data ${outFile}\n`
);
