/**
 * play_policy.js - let the cloned network actually play, and compare it with
 * the bot it was trained to imitate.
 *
 *   node play_policy.js
 *   node play_policy.js --games 20 --cap 2000 --model model.json
 *
 * This is the number that matters, and it is almost always much worse than the
 * accuracy figure from train_policy.py. The gap is the whole lesson of the
 * phase, so it is worth being precise about why it exists.
 *
 * Accuracy is measured one position at a time, on positions the expert
 * reached. Play is a chain of hundreds of decisions where each one determines
 * what you see next. A model that is right 90% of the time is wrong roughly
 * once every ten pieces, and each mistake moves it onto a board slightly
 * unlike anything in its training data - where it is now less than 90%
 * accurate, so it errs sooner, and the boards get stranger. Errors compound
 * instead of averaging out.
 *
 * That is covariate shift. `node collect.js --policy model.json` is the fix.
 */

import { readFileSync } from 'node:fs';
import { createBoard, Bag, PIECES, placePiece, clearLines } from './src/tetris-core.js';
import { chooseMove, PUBLISHED_WEIGHTS } from './src/tetris-ai.js';
import {
  loadPolicy,
  choosePolicyMove,
  moveClass,
  loadValueNet,
  chooseValueMove,
} from './src/tetris-policy.js';

const argv = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? fallback : argv[i + 1];
};

const games = Number(flag('games', 20));
const cap = Number(flag('cap', 500));
const valueFile = flag('value', null);
const modelFile = flag('model', 'model.json');

// Two ways to use a network here. --model is the behaviour-cloned policy that
// names a move directly. --value is the afterstate scorer: we enumerate the
// legal moves ourselves and ask the network only to judge the results.
const model = JSON.parse(readFileSync(valueFile ?? modelFile, 'utf8'));
const policy = valueFile ? loadValueNet(model) : loadPolicy(model);
const chooseCloned = valueFile
  ? (b, c) => chooseValueMove(b, c, policy)
  : (b, c, n) => choosePolicyMove(b, c, n, policy);

let weights = PUBLISHED_WEIGHTS;
try {
  weights = JSON.parse(readFileSync('weights.json', 'utf8')).weights;
} catch {}

// Seeds 20000+ : not used to train the weights, not used to collect the data.
const SEEDS = Array.from({ length: games }, (_, i) => 20000 + i);

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** chooser(board, current, next) -> {rotation, col, row} | null */
function playWith(chooser) {
  const lines = [];
  let deaths = 0;
  let pieces = 0;

  for (const seed of SEEDS) {
    const board = createBoard();
    const bag = new Bag(seed);
    let current = bag.next();
    let next = bag.next();
    let cleared = 0;

    for (let i = 0; i < cap; i++) {
      const move = chooser(board, current, next);
      if (move === null) {
        deaths++;
        break;
      }
      placePiece(board, PIECES[current].rotations[move.rotation], move.row, move.col);
      cleared += clearLines(board);
      pieces++;
      current = next;
      next = bag.next();
    }
    lines.push(cleared);
  }

  return { lines, median: median(lines), mean: lines.reduce((a, b) => a + b, 0) / games, deaths, pieces };
}

console.log(`\nthe cloned policy, playing for itself`);
console.log(`-------------------------------------`);
console.log(`model      ${valueFile ?? modelFile}  (${valueFile ? 'value network, scores afterstates' : 'cloned policy, names the move'})`);
console.log(`           ${model.hidden?.join(' x ') ?? '?'} hidden units, trained on ${model.trainRows?.toLocaleString()} rows`);
console.log(
  valueFile
    ? `reported   test R2 ${model.testR2.toFixed(3)}`
    : `reported   ${(model.testAccuracy * 100).toFixed(1)}% test accuracy, ${(model.top3Accuracy * 100).toFixed(1)}% top-3`
);
console.log(`games      ${games} on seeds 20000+, ${cap}-piece cap\n`);

const expert = playWith((b, c, n) => {
  const m = chooseMove(b, c, n, weights, { lookahead: true });
  return m && m.score !== -Infinity ? m : null;
});

const cloned = playWith(chooseCloned);

const row = (label, r) =>
  `  ${label.padEnd(18)} ${String(r.median).padStart(7)} ${r.mean.toFixed(1).padStart(8)} ` +
  `${String(Math.min(...r.lines)).padStart(7)} ${String(Math.max(...r.lines)).padStart(7)} ${String(r.deaths).padStart(7)}`;

console.log('                      median     mean   worst    best  died');
console.log('                     -------  -------  ------  ------  ------');
console.log(row('expert (search)', expert));
console.log(row('cloned (network)', cloned));

const ratio = cloned.mean / Math.max(expert.mean, 1);
console.log(`\nthe network plays at ${(ratio * 100).toFixed(1)}% of its teacher's level`);

// Agreement measured on boards the POLICY reaches, not boards the expert
// reaches. This is the diagnostic that reveals covariate shift: if it is much
// lower than the test accuracy from training, the model is being asked about
// positions its training set never contained.
let agree = 0;
let total = 0;
for (const seed of SEEDS.slice(0, Math.min(5, games))) {
  const board = createBoard();
  const bag = new Bag(seed);
  let current = bag.next();
  let next = bag.next();
  for (let i = 0; i < cap; i++) {
    const move = chooseCloned(board, current, next);
    if (move === null) break;
    const want = chooseMove(board, current, next, weights, { lookahead: true });
    if (want && want.score !== -Infinity) {
      total++;
      if (moveClass(want.rotation, want.col) === moveClass(move.rotation, move.col)) agree++;
    }
    placePiece(board, PIECES[current].rotations[move.rotation], move.row, move.col);
    clearLines(board);
    current = next;
    next = bag.next();
  }
}

console.log(`\nagreement with the expert, move for move:`);
if (!valueFile) {
  console.log(`  on boards the EXPERT reached (training distribution)   ${(model.testAccuracy * 100).toFixed(1)}%`);
}
console.log(`  on boards the NETWORK reaches (what actually happens)  ${((agree / Math.max(total, 1)) * 100).toFixed(1)}%`);

if (!valueFile) {
  console.log(
    `\nIf the second number is clearly lower, that is covariate shift, and it is\n` +
      `the reason the network plays worse than its accuracy suggests. The fix:\n` +
      `  node collect.js --policy ${modelFile} --games 60 --out data/dagger.csv\n` +
      `then append that to your training data and retrain.\n`
  );
} else {
  console.log(
    `\nThe value network never had to learn the rules - it only judges boards we\n` +
      `hand it, so it cannot propose an illegal or absurd move. That is why it\n` +
      `survives where the move-classifier does not.\n`
  );
}
