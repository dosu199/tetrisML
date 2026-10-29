/**
 * train-dqn.js - teach the network to play by playing.
 *
 *   node scripts/train-dqn.js                    # the defaults train the shipped models/dqn-model.json
 *   node scripts/train-dqn.js --epochs 4000 --out models/test.json
 *   node scripts/train-dqn.js --help             # every flag and its default
 *
 * Deep Q-learning on "afterstates", in plain JavaScript.
 *
 * There is no teacher and no dataset. The only signal is the game's reward
 * (moveReward in src/tetris-dqn.js):
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
 * Transitions go into a memory of the last 30,000 (scripts/lib/replay-memory.js).
 * After each game, one training step uses a random batch of 512 of them.
 *
 * EXPERIMENTS
 * -----------
 * The flags below the defaults (--nstep, --hold, --garbage, --features full,
 * --lr-end, --avg, --updates, --pieces) are experiments that were tried and
 * did not beat the defaults; they are kept so the results can be reproduced.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { createBoard, createPieceGenerator, garbageSeed, mulberry32, placements, playEpisode, addGarbageRow, PLAY_COLS } from '../src/tetris-core.js';
import { MLP, FEATURE_SETS, boardState, nextStates, greedyPick, moveReward, chooseDqnMove, chooseDqnAction } from '../src/tetris-dqn.js';
import { parseFlags, oneOf, numberList, STRING, NUMBER } from './lib/cli.js';
import { ReplayMemory } from './lib/replay-memory.js';
import { SEED_BASE } from './lib/seeds.js';
import { mean } from './lib/stats.js';

/* ----------------------------------------------------------------- config */

const opts = parseFlags({
  epochs: 3000, // training games (after warm-up)
  decay: 2000, // games over which epsilon falls
  'eps-start': 1,
  'eps-end': 1e-3,
  batch: 512,
  lr: 1e-3,
  gamma: 0.99,
  memory: 30000,
  'max-pieces': 2000, // per training game
  seed: 8, // the defaults train the shipped dqn-model.json
  out: 'models/dqn-model.json',
  log: 100, // test (and maybe save) the network every this many games
  'eval-games': 5, // greedy test games per check
  'eval-cap': 1000,
  // --snapshots 0,1000,2800: also save the network as it was after these many
  // training games (0 = before any training) into --snapshot-dir. Used by
  // learning.html to replay how the network improved. Does not change training.
  snapshots: STRING,
  'snapshot-dir': 'models/snapshots',
  // Which pieces to train on:
  //   random  every piece independent and random ("memoryless"): the shipped
  //           model; it plays the normal 7-bag just as well
  //   bag     the normal 7-bag
  //   mix     even games 7-bag, odd games random
  pieces: 'random',
  // --lr-end: learning rate at the last game. The rate stays at --lr while
  // epsilon decays, then falls exponentially to --lr-end. Default: no decay.
  'lr-end': NUMBER,
  // --avg B: test and save a running average of the weights instead,
  //   avg = B x avg + (1 - B) x current   after every training step.
  // 0 = off; 0.99 averages roughly the last 100 steps.
  avg: 0,
  updates: 1, // training steps after each game
  // --nstep N: learn from the next N moves' rewards instead of just one:
  //   1-step:  V(s0) ~ r0 + 0.99 V(s1)
  //   3-step:  V(s0) ~ r0 + 0.99 r1 + 0.99^2 r2 + 0.99^3 V(s3)
  nstep: 1,
  // --garbage N: in some games a garbage row rises every N pieces;
  // --garbage-share: the fraction of games that get it.
  garbage: 0,
  'garbage-share': 0.5,
  // --hold: train with the hold box. The network also sees which piece is in
  // the box (7 extra inputs), and every turn it may place the current piece
  // or hold it and place the other one.
  hold: false,
  // --features basic (default) | full (all 8 bot features) | hold (with --hold).
  // 'full' needs input scaling to learn at all, so it switches it on;
  // --scale switches it on for the others too.
  features: STRING,
  scale: false,
});

const HOLD = opts.hold;
const FEATURE_KIND = oneOf(opts.features ?? (HOLD ? 'hold' : 'basic'), 'features', Object.keys(FEATURE_SETS));
const FEATURES = FEATURE_SETS[FEATURE_KIND];
const SCALED = FEATURE_KIND === 'full' || opts.scale;
const PIECES_MODE = oneOf(opts.pieces, 'pieces', ['bag', 'random', 'mix']);
const SNAPSHOTS = new Set(numberList(opts.snapshots ?? '', 'snapshots'));
const LR_END = opts.lrEnd ?? opts.lr;
const GAMMA = opts.gamma;
const LAST_OUT = opts.out.endsWith('.json') ? `${opts.out.slice(0, -5)}-last.json` : `${opts.out}-last`;

/** The piece generator of training game number `game`. */
const randomizerFor = (game) => (PIECES_MODE === 'random' || (PIECES_MODE === 'mix' && game % 2 === 1) ? 'memoryless' : 'bag');

/** Garbage every opts.garbage pieces in a share of the games, spread evenly (share 0.5: every second game). */
const garbageFor = (game) =>
  opts.garbage > 0 && Math.floor((game + 1) * opts.garbageShare) !== Math.floor(game * opts.garbageShare) ? opts.garbage : 0;

/* ---------------------------------------------------------------- network */

// One random stream for everything that is random in training (initial
// weights, exploration, batch sampling), so a seed reproduces a run exactly.
const rand = mulberry32(opts.seed);
const net = new MLP([FEATURES.length, 64, 64, 1], { rand, features: FEATURES, scaled: SCALED });
const spec = net.inputSpec;
const memory = new ReplayMemory(opts.memory, FEATURES.length);
// The averaged network (see --avg): starts as a copy of the network.
const avgNet = opts.avg > 0 ? MLP.fromJSON(net.toJSON()) : null;
/** The network we test and save: the average if there is one. */
const keeper = () => avgNet ?? net;

function saveSnapshot(games) {
  mkdirSync(opts.snapshotDir, { recursive: true });
  writeFileSync(`${opts.snapshotDir}/game-${games}.json`, JSON.stringify({ ...net.toJSON(), trainedGames: games }));
}

/** The learning rate: --lr while epsilon decays, then exponentially towards --lr-end. */
function learningRate(epoch) {
  const late = Math.max(0, epoch - opts.decay) / Math.max(1, opts.epochs - opts.decay);
  return opts.lr * (LR_END / opts.lr) ** late;
}

/** avg = beta x avg + (1 - beta) x current, for every weight and bias. */
function blendWeights(avg, current, beta) {
  for (let l = 0; l < current.W.length; l++) {
    for (const [a, w] of [[avg.W[l], current.W[l]], [avg.b[l], current.b[l]]]) {
      for (let k = 0; k < w.length; k++) a[k] = beta * a[k] + (1 - beta) * w[k];
    }
  }
}

function trainOnBatch(epoch) {
  const { X, y } = memory.sampleBatch(opts.batch, rand, (state) => net.predict(state));
  const loss = net.trainStep(X, y, learningRate(epoch));
  if (avgNet) blendWeights(avgNet, net, opts.avg);
  return loss;
}

/* ------------------------------------------------------------------ games */

/**
 * Every option this turn: each placement of the current piece and, with
 * hold, each placement of the other piece after holding (flagged hold: true).
 * A hold network's state also says which piece each option leaves in the box.
 */
function moveOptions(board, current, next, held) {
  if (!HOLD) return nextStates(board, current, spec);
  const options = nextStates(board, current, spec, held);
  for (const o of options) o.hold = false;
  for (const o of nextStates(board, held ?? next, spec, current)) {
    o.hold = true;
    options.push(o);
  }
  return options;
}

/** Epsilon-greedy: a random option with probability epsilon, otherwise the network's best. */
function pickEpsilonGreedy(options, epsilon) {
  if (rand() < epsilon) return options[Math.floor(rand() * options.length)];
  return greedyPick(options, net);
}

/**
 * N-step returns. `pending` holds the last moves still waiting for theirs.
 * Once it holds N, the oldest gets r0 + g r1 + ... + g^(N-1) r(N-1), with
 * g^N x V(state after them) still to come. At game over every waiting move
 * is stored with the rewards it had left and nothing to come.
 */
function storeReturns(pending, nextState, done) {
  if (done) {
    for (let i = 0; i < pending.length; i++) {
      let ret = 0;
      for (let j = pending.length - 1; j >= i; j--) ret = pending[j].reward + GAMMA * ret;
      memory.add(pending[i].state, ret, nextState, true, GAMMA);
    }
    pending.length = 0;
  } else if (pending.length === opts.nstep) {
    let ret = 0;
    for (let j = opts.nstep - 1; j >= 0; j--) ret = pending[j].reward + GAMMA * ret;
    memory.add(pending[0].state, ret, nextState, false, GAMMA ** opts.nstep);
    pending.shift();
  }
}

/**
 * One training game with the epsilon-greedy policy; every move goes into
 * the replay memory. Garbage (if any) is part of what happens after a move,
 * so the network learns what a board is worth with garbage still to come.
 */
function playTrainingGame(seed, epsilon, randomizer, garbageEvery) {
  const gen = createPieceGenerator(randomizer, seed);
  const holeRand = mulberry32(garbageSeed(seed));
  let board = createBoard();
  let current = gen.next();
  let next = gen.next();
  let held = null;
  let state = boardState(board, 0, spec, held);
  let pieces = 0;
  let lines = 0;
  let score = 0;
  const pending = []; // moves waiting for their N-step return

  for (;;) {
    const options = moveOptions(board, current, next, held);
    if (options.length === 0) return { pieces, lines, score, died: true }; // unreachable: `done` ends the game first
    const pick = pickEpsilonGreedy(options, epsilon);
    if (pick.hold) {
      if (held == null) {
        held = current;
        current = next;
        next = gen.next();
      } else {
        [held, current] = [current, held];
      }
    }

    const reward = moveReward(pick.lines);
    board = pick.board;
    pieces++;
    lines += pick.lines;
    score += reward;
    current = next;
    next = gen.next();

    let buried = false;
    if (garbageEvery > 0 && pieces % garbageEvery === 0) {
      buried = !addGarbageRow(board, Math.floor(holeRand() * PLAY_COLS));
    }
    // over when the stack is pushed out of the top, or when nothing fits:
    // the next piece (and with hold, the piece holding would give)
    const stuck = placements(board, current).length === 0 && (!HOLD || placements(board, held ?? next).length === 0);
    const done = buried || stuck;

    pending.push({ state, reward });
    storeReturns(pending, pick.state, done);
    state = pick.state;
    if (done) return { pieces, lines, score, died: true };
    if (pieces >= opts.maxPieces) return { pieces, lines, score, died: false };
  }
}

/* -------------------------------------------------------------- checkpoint */

/**
 * Plays a few games with NO exploration, on fixed seeds, and returns the
 * average number of pieces survived. Deep Q-learning is not monotonic:
 * performance climbs, then often sags or collapses as the network chases its
 * own moving targets, so saving the last network can throw away a much
 * better one. Every --log games we measure, and --out always holds the best.
 */
function evaluateGreedy() {
  // test on the kind of pieces the network trains on ('mix' tests both)
  const kinds = PIECES_MODE === 'mix' ? ['bag', 'memoryless'] : [PIECES_MODE === 'random' ? 'memoryless' : 'bag'];
  const games = opts.evalGames;
  // With hold, ordinary games almost never end, so they cannot tell good
  // checkpoints apart. Test on fast garbage instead (a row every 3 pieces),
  // where every game ends and longer survival means a better network.
  if (HOLD) {
    let total = 0;
    for (let g = 0; g < games; g++) {
      total += playEpisode((b, c, n, h) => chooseDqnAction(b, c, n, keeper(), h), SEED_BASE.checkpoint + g, {
        cap: opts.evalCap,
        randomizer: kinds[g % kinds.length],
        garbageEvery: 3,
        hold: true,
      }).pieces;
    }
    return total / games;
  }
  const greedy = (b, c) => chooseDqnMove(b, c, null, keeper());
  let total = 0;
  for (const randomizer of kinds) {
    for (let g = 0; g < games; g++) {
      total += playEpisode(greedy, SEED_BASE.checkpoint + g, { cap: opts.evalCap, randomizer }).pieces;
    }
  }
  let score = total / (games * kinds.length);
  // With --garbage, add the average survival in garbage games. Normal games
  // of a good network mostly hit the cap; these tell good checkpoints apart.
  if (opts.garbage > 0) {
    let garbageTotal = 0;
    for (let k = 0; k < games; k++) {
      garbageTotal += playEpisode(greedy, SEED_BASE.checkpointGarbage + k, {
        cap: opts.evalCap,
        randomizer: kinds[k % kinds.length],
        garbageEvery: opts.garbage,
      }).pieces;
    }
    score += garbageTotal / games;
  }
  return score;
}

/* ------------------------------------------------------------------- main */

function printSettings() {
  const learning = [
    `batch ${opts.batch}`,
    `lr ${opts.lr}${LR_END !== opts.lr ? ` -> ${LR_END}` : ''}`,
    `gamma ${GAMMA}`,
    `memory ${opts.memory}`,
    ...(opts.nstep !== 1 ? [`${opts.nstep}-step targets`] : []),
    ...(opts.updates !== 1 ? [`${opts.updates} updates per game`] : []),
    ...(opts.avg ? [`weight average ${opts.avg}`] : []),
  ];
  console.log(`\ntraining a DQN to play Tetris`);
  console.log(`  network   ${net.sizes.join(' -> ')}  (input: ${FEATURES.join(', ')}${SCALED ? ', scaled' : ''})`);
  console.log(`  games     ${opts.epochs} after warm-up, epsilon ${opts.epsStart} -> ${opts.epsEnd} over ${opts.decay}`);
  console.log(`  learning  ${learning.join(', ')}`);
  console.log(`  pieces    ${PIECES_MODE}${PIECES_MODE === 'mix' ? ' (half 7-bag, half random)' : ''}`);
  console.log(`  cap       ${opts.maxPieces} pieces per training game`);
  console.log(`  hold      ${HOLD ? 'on (the network also sees the hold box)' : 'off'}`);
  console.log(`  garbage   ${opts.garbage > 0 ? `a row every ${opts.garbage} pieces in ${Math.round(opts.garbageShare * 100)}% of games` : 'off'}\n`);
  console.log('   game   epsilon    pieces (avg last 100)   lines (avg)   greedy test   elapsed');
  console.log('  -----   -------    ---------------------   -----------   -----------   -------');
}

function train() {
  printSettings();
  if (SNAPSHOTS.has(0)) saveSnapshot(0);
  const started = Date.now();
  const recent = []; // results of the last 100 games
  let epoch = 0; // games that trained the network (after warm-up)
  let game = 0; // all games played
  let bestGreedy = -1;
  let bestEpoch = 0;

  while (epoch < opts.epochs) {
    const epsilon = opts.epsEnd + (Math.max(opts.decay - epoch, 0) * (opts.epsStart - opts.epsEnd)) / opts.decay;
    const result = playTrainingGame(SEED_BASE.training + game, epsilon, randomizerFor(game), garbageFor(game));
    game++;

    // warm-up: no learning until the memory holds a tenth of its capacity
    if (memory.size < opts.memory / 10) continue;

    for (let u = 0; u < opts.updates; u++) trainOnBatch(epoch);
    epoch++;
    if (SNAPSHOTS.has(epoch)) saveSnapshot(epoch);
    recent.push(result);
    if (recent.length > 100) recent.shift();

    if (epoch % opts.log === 0 || epoch === opts.epochs) {
      const greedy = evaluateGreedy();
      const improved = greedy > bestGreedy;
      if (improved) {
        bestGreedy = greedy;
        bestEpoch = epoch;
        const model = { ...keeper().toJSON(), trainedGames: epoch, greedyTest: greedy, reward: '1 + 10 * lines^2 per piece' };
        writeFileSync(opts.out, JSON.stringify(model));
      }
      console.log(
        `  ${String(epoch).padStart(5)}   ${epsilon.toFixed(3).padStart(7)}    ` +
          `${mean(recent.map((r) => r.pieces)).toFixed(1).padStart(21)}   ${mean(recent.map((r) => r.lines)).toFixed(1).padStart(11)}   ` +
          `${greedy.toFixed(0).padStart(11)}   ${((Date.now() - started) / 1000).toFixed(0).padStart(6)}s` +
          (improved ? '   <- best, saved' : '')
      );
    }
  }

  writeFileSync(LAST_OUT, JSON.stringify(keeper().toJSON()));
  console.log(`\nbest network: game ${bestEpoch}, ${bestGreedy.toFixed(0)} pieces in the greedy test -> ${opts.out}`);
  console.log(`last network saved separately as ${LAST_OUT}`);
  console.log(`${game} games in ${((Date.now() - started) / 60000).toFixed(1)} min`);
  console.log(`next: node scripts/hardmode.js --model ${opts.out} --games 50 --seed ${SEED_BASE.benchmark}\n`);
}

train();
