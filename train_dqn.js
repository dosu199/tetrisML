/**
 * train_dqn.js - teach the network to play by playing.
 *
 *   node train_dqn.js                       # defaults below, a few minutes
 *   node train_dqn.js --epochs 4000 --out dqn-model.json
 *
 * Deep Q-learning on "afterstates", following
 * vietnh1009/Tetris-deep-Q-learning-pytorch (same network, reward and
 * hyperparameters), reimplemented in plain JavaScript.
 *
 * There is no teacher and no dataset. The only signal is the game's reward:
 *
 *   reward per piece = 1 + 10 x (lines cleared)^2
 *
 * Surviving earns a little every piece; clearing lines earns a lot, and four
 * at once (a "tetris") earns 16 times a single. Losing simply ends the reward.
 *
 * HOW IT LEARNS
 * -------------
 * The network V(board) estimates the total future reward from a board. After
 * every move we have a transition:
 *
 *   previous board -> (this move's reward) -> new board
 *
 * and the network is nudged so that
 *
 *   V(previous board)  ~=  reward + 0.99 x V(new board)
 *
 * or just `reward` if the game ended. That is the Bellman equation: a board is
 * worth what it pays now plus what the next board is worth. Repeated over
 * thousands of moves, value flows backwards from line clears to the boards
 * that set them up, and from deaths to the boards that caused them.
 *
 * EXPLORATION
 * -----------
 * At first the network knows nothing, so it plays mostly at random (epsilon
 * = 1) to see what happens. Epsilon falls linearly to 0.001 over `--decay`
 * games; by the end it almost always plays its best move.
 *
 * REPLAY MEMORY
 * -------------
 * Transitions go into a memory of the last 30,000. After each game, one
 * training step uses a random batch of 512 of them. Random batches break the
 * correlation between consecutive moves, which keeps training stable.
 */

import { writeFileSync } from 'node:fs';
import { createBoard, Bag, MemorylessGen, mulberry32, placements, playEpisode } from './src/tetris-core.js';
import { MLP, FEATURE_SETS, boardState, nextStates, chooseDqnMove } from './src/tetris-dqn.js';

const argv = process.argv.slice(2);
const flag = (n, d) => {
  const i = argv.indexOf(`--${n}`);
  return i === -1 ? d : argv[i + 1];
};

const EPOCHS = Number(flag('epochs', 3000)); // training games (after warm-up)
const DECAY = Number(flag('decay', 2000)); // games over which epsilon falls
const BATCH = Number(flag('batch', 512));
const LR = Number(flag('lr', 1e-3));
// --lr-end: learning rate at the last game. The rate stays at --lr while the
// network explores (the first --decay games, which is when it learns the
// most), then falls smoothly (exponentially) to --lr-end, so the end of
// training only fine-tunes instead of knocking a good network over.
// Decaying from the very start stops it learning at all. Default: no decay.
const LR_END = Number(flag('lr-end', LR));
// --avg B: also keep a running average of the weights,
//   avg = B x avg + (1 - B) x current     after every training step,
// and test/save that average instead of the raw network. An average of many
// recent networks wobbles much less than any one of them. 0 = off (default);
// 0.99 averages roughly the last 100 steps.
const AVG = Number(flag('avg', 0));
// --updates N: training steps after each game (default 1, like the reference).
// With 1, the whole run is only ~3,000 steps, which is very little learning.
const UPDATES = Number(flag('updates', 1));
const GAMMA = Number(flag('gamma', 0.99));
const MEMORY = Number(flag('memory', 30000));
const EPS_START = Number(flag('eps-start', 1));
const EPS_END = Number(flag('eps-end', 1e-3));
const MAX_PIECES = Number(flag('max-pieces', 2000)); // per training game
const SEED = Number(flag('seed', 8)); // the defaults train the shipped dqn-model.json
const OUT = flag('out', 'dqn-model.json');
const LOG = Number(flag('log', 100));
// Which pieces to train on:
//   bag     the normal 7-bag
//   random  every piece independent and random ("memoryless"; default, the
//           shipped model - it plays normal pieces just as well)
//   mix     alternate: even games 7-bag, odd games random
const PIECES_MODE = flag('pieces', 'random');
if (!['bag', 'random', 'mix'].includes(PIECES_MODE)) throw new Error('--pieces must be bag, random or mix');
const useRandom = (game) => PIECES_MODE === 'random' || (PIECES_MODE === 'mix' && game % 2 === 1);
const EVAL_GAMES = Number(flag('eval-games', 5)); // greedy test games per check
const EVAL_CAP = Number(flag('eval-cap', 1000));

// --features basic (the reference's 4, default) or full (all 8 bot features).
// 'full' needs feature scaling to learn at all, so it switches scaling on;
// --scale turns it on for 'basic' too.
const FEATURE_KIND = flag('features', 'basic');
const FEATURES = FEATURE_SETS[FEATURE_KIND];
if (!FEATURES) throw new Error('--features must be basic or full');
const SCALED = FEATURE_KIND === 'full' || argv.includes('--scale');
const STATE_SIZE = FEATURES.length;

const rand = mulberry32(SEED);
const net = new MLP([STATE_SIZE, 64, 64, 1], rand, FEATURES, SCALED);
const IDX = net.input;
// The averaged network (see --avg). It starts as a copy of the network.
const avgNet = AVG > 0 ? MLP.fromJSON(net.toJSON()) : null;
// The network we test and save: the average if there is one.
const keeper = () => avgNet ?? net;


/* ---------------------------------------------------------- replay memory */

const memS = new Float64Array(MEMORY * STATE_SIZE);
const memN = new Float64Array(MEMORY * STATE_SIZE);
const memR = new Float64Array(MEMORY);
const memD = new Uint8Array(MEMORY);
let memCount = 0;
let memPos = 0;

function remember(state, reward, next, done) {
  memS.set(state, memPos * STATE_SIZE);
  memN.set(next, memPos * STATE_SIZE);
  memR[memPos] = reward;
  memD[memPos] = done ? 1 : 0;
  memPos = (memPos + 1) % MEMORY;
  if (memCount < MEMORY) memCount++;
}

function trainOnBatch() {
  const B = Math.min(BATCH, memCount);
  const X = new Float64Array(B * STATE_SIZE);
  const y = new Float64Array(B);
  const next = new Float64Array(STATE_SIZE);
  for (let n = 0; n < B; n++) {
    const k = Math.floor(rand() * memCount);
    X.set(memS.subarray(k * STATE_SIZE, (k + 1) * STATE_SIZE), n * STATE_SIZE);
    next.set(memN.subarray(k * STATE_SIZE, (k + 1) * STATE_SIZE));
    // Bellman target: reward now, plus the discounted value of where we went
    y[n] = memD[k] ? memR[k] : memR[k] + GAMMA * net.predict(next);
  }
  const late = Math.max(0, epoch - DECAY) / Math.max(1, EPOCHS - DECAY);
  const lr = LR * (LR_END / LR) ** late;
  const loss = net.trainStep(X, y, lr);
  if (avgNet) {
    for (let l = 0; l < net.W.length; l++) {
      for (const [a, w] of [[avgNet.W[l], net.W[l]], [avgNet.b[l], net.b[l]]]) {
        for (let k = 0; k < w.length; k++) a[k] = AVG * a[k] + (1 - AVG) * w[k];
      }
    }
  }
  return loss;
}

/* ------------------------------------------------------------------ games */

/** One game with the epsilon-greedy policy; every move goes into memory. */
function playTrainingGame(seed, epsilon, random) {
  const board0 = createBoard();
  const bag = random ? new MemorylessGen(seed) : new Bag(seed);
  let board = board0;
  let state = boardState(board, 0, IDX);
  let current = bag.next();
  let pieces = 0;
  let lines = 0;
  let score = 0;

  while (true) {
    const options = nextStates(board, current, IDX);
    if (options.length === 0) break; // cannot happen on the first piece

    let pick;
    if (rand() < epsilon) {
      pick = options[Math.floor(rand() * options.length)];
    } else {
      let best = -Infinity;
      for (const o of options) {
        const v = net.predict(o.state);
        if (v > best) {
          best = v;
          pick = o;
        }
      }
    }

    const reward = 1 + 10 * pick.lines * pick.lines;
    board = pick.board;
    pieces++;
    lines += pick.lines;
    score += reward;
    current = bag.next();

    // The game is over when the next piece has nowhere to go.
    const done = placements(board, current).length === 0;
    remember(state, reward, pick.state, done);
    state = pick.state;
    if (done) return { pieces, lines, score, died: true };
    if (pieces >= MAX_PIECES) return { pieces, lines, score, died: false };
  }
  return { pieces, lines, score, died: true };
}

/* -------------------------------------------------------------- checkpoint */

/**
 * Play a few games with NO exploration, on fixed seeds, and return the average
 * number of pieces survived. Deep Q-learning is not monotonic: performance
 * climbs, then often sags or collapses as the network chases its own moving
 * targets. Saving the last network can therefore throw away a much better
 * one. So every LOG games we measure, and OUT always holds the best so far.
 * The seeds (500000+) are separate from training (1000000+) and from the
 * benchmark seeds in hardmode.js (20000+, 90000+).
 */
function evaluateGreedy() {
  // Test on the same kind of pieces the network trains on ('mix' tests both).
  const kinds = PIECES_MODE === 'mix' ? ['bag', 'memoryless'] : [PIECES_MODE === 'random' ? 'memoryless' : 'bag'];
  let total = 0;
  for (const randomizer of kinds) {
    for (let g = 0; g < EVAL_GAMES; g++) {
      total += playEpisode((b, c) => chooseDqnMove(b, c, keeper()), 500000 + g, { cap: EVAL_CAP, randomizer }).pieces;
    }
  }
  return total / (EVAL_GAMES * kinds.length);
}

/* ------------------------------------------------------------------- main */

console.log(`\ntraining a DQN to play Tetris`);
console.log(`  network   ${net.sizes.join(' -> ')}  (input: ${FEATURES.join(', ')}${SCALED ? ', scaled' : ''})`);
console.log(`  games     ${EPOCHS} after warm-up, epsilon ${EPS_START} -> ${EPS_END} over ${DECAY}`);
console.log(`  learning  batch ${BATCH}, lr ${LR}${LR_END !== LR ? ` -> ${LR_END}` : ''}, gamma ${GAMMA}, memory ${MEMORY}${UPDATES !== 1 ? `, ${UPDATES} updates per game` : ''}${AVG ? `, weight average ${AVG}` : ''}`);
console.log(`  pieces    ${PIECES_MODE}${PIECES_MODE === 'mix' ? ' (half 7-bag, half random)' : ''}`);
console.log(`  cap       ${MAX_PIECES} pieces per training game\n`);
console.log('   game   epsilon    pieces (avg last 100)   lines (avg)   greedy test   elapsed');
console.log('  -----   -------    ---------------------   -----------   -----------   -------');

const started = Date.now();
let epoch = 0;
let game = 0;
const recent = [];
let bestGreedy = -1;
let bestEpoch = 0;

while (epoch < EPOCHS) {
  const epsilon = EPS_END + (Math.max(DECAY - epoch, 0) * (EPS_START - EPS_END)) / DECAY;
  const result = playTrainingGame(1_000_000 + game, epsilon, useRandom(game));
  game++;

  // Like the reference: no learning until the memory holds a tenth of its size.
  if (memCount < MEMORY / 10) continue;

  for (let u = 0; u < UPDATES; u++) trainOnBatch();
  epoch++;
  recent.push(result);
  if (recent.length > 100) recent.shift();

  if (epoch % LOG === 0 || epoch === EPOCHS) {
    const avgP = recent.reduce((a, r) => a + r.pieces, 0) / recent.length;
    const avgL = recent.reduce((a, r) => a + r.lines, 0) / recent.length;
    const greedy = evaluateGreedy();
    const improved = greedy > bestGreedy;
    if (improved) {
      bestGreedy = greedy;
      bestEpoch = epoch;
      writeFileSync(
        OUT,
        JSON.stringify({ ...keeper().toJSON(), trainedGames: epoch, greedyTest: greedy, reward: '1 + 10 * lines^2 per piece' })
      );
    }
    console.log(
      `  ${String(epoch).padStart(5)}   ${epsilon.toFixed(3).padStart(7)}    ` +
        `${avgP.toFixed(1).padStart(21)}   ${avgL.toFixed(1).padStart(11)}   ` +
        `${greedy.toFixed(0).padStart(11)}   ${((Date.now() - started) / 1000).toFixed(0).padStart(6)}s` +
        (improved ? '   <- best, saved' : '')
    );
  }
}

writeFileSync(OUT.replace(/\.json$/, '-last.json'), JSON.stringify(keeper().toJSON()));
console.log(`\nbest network: game ${bestEpoch}, ${bestGreedy.toFixed(0)} pieces in the greedy test -> ${OUT}`);
console.log(`last network saved separately as ${OUT.replace(/\.json$/, '-last.json')}`);
console.log(`${game} games in ${((Date.now() - started) / 60000).toFixed(1)} min`);
console.log(`next: node hardmode.js --model ${OUT} --games 50 --seed 90000\n`);
