/**
 * collect.js - turns the Phase 1 bot into a labelled dataset.
 *
 *   node collect.js                                  # 120 games -> data/moves.csv
 *   node collect.js --games 200 --out data/big.csv
 *   node collect.js --policy model.json --out data/dagger.csv   # see DAgger below
 *
 * Every row is one decision: the board as the bot saw it, which piece it was
 * holding, which piece was coming next, and where it decided to put the piece.
 * That is a supervised learning problem in the most ordinary sense - 254
 * inputs, one label out of 40 - and everything downstream is the material from
 * your course applied to data you generated yourself.
 *
 * The expert uses 2-ply lookahead here even though training runs 1-ply. The
 * next piece is one of the inputs, so the teacher had better actually be using
 * it; otherwise the model would be learning to read a column that carries no
 * signal.
 *
 * DAGGER MODE (--policy)
 * ----------------------
 * Behaviour cloning has a famous failure: the model only ever sees boards the
 * *expert* reached. The expert keeps the stack flat, so the training set is
 * full of flat boards. The moment the model makes one mistake it is looking at
 * a messy board it has no experience of, makes a worse choice, and compounds.
 * This is covariate shift, and it is why a policy with high move accuracy can
 * still play badly.
 *
 * The fix is Dataset Aggregation: let the *model* drive, but label every state
 * with what the *expert* would have done. Now the training set contains the
 * messy boards the model actually gets itself into, along with the right
 * answer for each. Pass --policy to do that, append the result to your
 * original CSV, and retrain.
 */

import { writeFileSync, mkdirSync, readFileSync, existsSync } from 'node:fs';
import { dirname } from 'node:path';
import { createBoard, Bag, PIECES, placePiece, clearLines, mulberry32 } from './src/tetris-core.js';
import { chooseMove, PUBLISHED_WEIGHTS } from './src/tetris-ai.js';
import {
  buildInput,
  inputColumns,
  moveClass,
  loadPolicy,
  choosePolicyMove,
} from './src/tetris-policy.js';

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};

const games = Number(flag('games', 120));
const cap = Number(flag('cap', 500));
const outFile = flag('out', 'data/moves.csv');
const policyFile = flag('policy', null);
const seedBase = Number(flag('seed', 1));

/**
 * Probability that the EXPERT drives on any given piece, when a policy is
 * supplied. This is the beta of DAgger.
 *
 * It matters because a weak policy left fully in charge tops out after a
 * handful of pieces, so a whole round of collection yields a few hundred rows,
 * all of them from hopeless boards. Letting the expert drive most of the time
 * keeps games long and the boards realistic, while the policy's occasional
 * turn introduces exactly the kind of small deviation it will need to recover
 * from. Lower beta each round as the policy improves.
 */
const beta = Number(flag('beta', 0.8));

// The expert is the Phase 1 bot, driven by whatever train.js learned.
let weights = PUBLISHED_WEIGHTS;
if (existsSync('weights.json')) {
  weights = JSON.parse(readFileSync('weights.json', 'utf8')).weights;
  console.log('expert: learned weights from weights.json');
} else {
  console.log('expert: published weights (no weights.json found - run train.js first)');
}

// In DAgger mode the policy chooses the moves that are actually played, while
// the expert still supplies every label.
let policy = null;
if (policyFile) {
  policy = loadPolicy(JSON.parse(readFileSync(policyFile, 'utf8')));
  console.log(`driver: ${policyFile} (DAgger, beta=${beta} - expert drives ${(beta * 100).toFixed(0)}% of pieces, labels everything)`);
} else {
  console.log('driver: the expert itself');
}

console.log(`collecting ${games} games, ${cap}-piece cap -> ${outFile}\n`);

const columns = inputColumns();
const rand = mulberry32(seedBase * 7919 + 13);
const rows = [];
const started = Date.now();
let totalLines = 0;
let deaths = 0;

for (let g = 0; g < games; g++) {
  const board = createBoard();
  const bag = new Bag(seedBase + g);
  let current = bag.next();
  let next = bag.next();

  for (let piece = 0; piece < cap; piece++) {
    // What the expert would do here. This is the label, always.
    const expert = chooseMove(board, current, next, weights, { lookahead: true });
    if (expert === null || expert.score === -Infinity) {
      deaths++;
      break;
    }

    const x = buildInput(board, current, next);
    const label = moveClass(expert.rotation, expert.col);
    rows.push(`${Array.from(x, (v) => (v ? 1 : 0)).join(',')},${current},${next},${label}`);

    // Who actually moves. In plain collection that is the expert; in DAgger
    // mode the model sometimes drives us into states the expert would never
    // visit, and we record the expert's answer for those states too.
    const played =
      policy && rand() >= beta ? choosePolicyMove(board, current, next, policy) : expert;
    if (played === null) {
      deaths++;
      break;
    }

    placePiece(board, PIECES[current].rotations[played.rotation], played.row, played.col);
    totalLines += clearLines(board);

    current = next;
    next = bag.next();
  }

  if ((g + 1) % 10 === 0 || g === games - 1) {
    const secs = ((Date.now() - started) / 1000).toFixed(0);
    process.stdout.write(`\r  ${g + 1}/${games} games, ${rows.length} rows, ${secs}s   `);
  }
}

console.log('\n');

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, `${columns.join(',')},piece_id,next_id,move_class\n${rows.join('\n')}\n`);

const mb = (Buffer.byteLength(rows.join('\n')) / 1e6).toFixed(1);
console.log(`rows          ${rows.length.toLocaleString()}`);
console.log(`columns       ${columns.length + 3} (${columns.length} inputs + 2 ids + 1 label)`);
console.log(`lines cleared ${totalLines.toLocaleString()} across ${games} games`);
console.log(`games lost    ${deaths}`);
console.log(`file          ${outFile} (${mb} MB)`);
console.log(`\nnext: python3 train_policy.py --data ${outFile}\n`);
