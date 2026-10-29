/**
 * tetris-dqn.js - the neural network that plays, and the search around it.
 * No DOM: the same code trains in Node (scripts/train-dqn.js) and plays in
 * the browser.
 *
 *   1. For the current piece, try every rotation x column.
 *   2. Describe each resulting board with a few numbers (the shipped model
 *      uses 4: lines cleared, holes, bumpiness, total height).
 *   3. A small network (4 -> 64 -> 64 -> 1) turns those numbers into one
 *      score V(board): "how much reward is still to come from this board".
 *   4. Play the move with the best score (1-ply), or look 1 or 2 pieces
 *      further ahead with the same network (2-ply, 3-ply).
 *
 * The network learns V from the game's reward alone (see moveReward); nobody
 * tells it what a good board looks like.
 *
 * Contents: inputs (which features the network sees), the MLP itself, and
 * the choosers: chooseDqnMove (placement only) and chooseDqnAction (may hold).
 */

import { PIECES, createBoard, copyBoard, playMove, placements, HOLD } from './tetris-core.js';
import { extractFeatures, decideWithHold, FEATURE_NAMES, NUM_FEATURES } from './tetris-ai.js';

/* ---------------------------------------------------------------- inputs */

const HELD_PREFIX = 'held:';

/**
 * Which numbers a network sees, by name (FEATURE_NAMES in tetris-ai.js). A
 * trained model stores its own list, so models trained on any set keep working.
 */
export const FEATURE_SETS = Object.freeze({
  // the 4 classic afterstate features; the shipped model uses these
  basic: Object.freeze(['linesCleared', 'holes', 'bumpiness', 'aggregateHeight']),
  // all 8 the search bot uses: the 4 above plus the shape of the surface
  full: Object.freeze(['linesCleared', 'holes', 'bumpiness', 'aggregateHeight', 'maxHeight', 'rowTransitions', 'colTransitions', 'wells']),
  // the basic 4 plus 7 inputs saying which piece is in the hold box
  hold: Object.freeze(['linesCleared', 'holes', 'bumpiness', 'aggregateHeight', ...PIECES.map((p) => HELD_PREFIX + p.name)]),
});

const ALIASES = Object.freeze({ totalHeight: 'aggregateHeight' }); // name used by older models

/**
 * Optional input scaling: each feature divided by roughly its largest value,
 * so every input lands around 0-1. The 'full' set needs it to learn at all;
 * the shipped 'basic' model learned best without it (scaled: false).
 */
const FEATURE_SCALE = Object.freeze({
  linesCleared: 4,
  holes: 100,
  bumpiness: 30,
  aggregateHeight: 240,
  maxHeight: 24,
  rowTransitions: 120,
  colTransitions: 60,
  wells: 50,
});

function featureIndexOf(name) {
  const i = FEATURE_NAMES.indexOf(ALIASES[name] ?? name);
  if (i === -1) throw new Error(`unknown feature: ${name}`);
  return i;
}

/**
 * How a network reads a board, worked out once per network:
 *   board[]  { pos, fi, scale }: input `pos` = feature `fi` / scale
 *   held[]   { pos, piece }:     input `pos` = 1 if `piece` is in the hold box
 */
function makeInputSpec(names, scaled) {
  const spec = { n: names.length, board: [], held: [] };
  names.forEach((name, pos) => {
    if (name.startsWith(HELD_PREFIX)) {
      const piece = PIECES.findIndex((p) => p.name === name.slice(HELD_PREFIX.length));
      if (piece < 0) throw new Error(`unknown piece in input: ${name}`);
      spec.held.push({ pos, piece });
    } else {
      const scale = scaled ? FEATURE_SCALE[ALIASES[name] ?? name] : 1;
      spec.board.push({ pos, fi: featureIndexOf(name), scale });
    }
  });
  return spec;
}

const featureScratch = new Float64Array(NUM_FEATURES);

/**
 * The network's input for a board after a move: `spec` is net.inputSpec;
 * `held` is the piece left in the hold box (only read by hold networks).
 */
export function boardState(board, linesCleared, spec, held = null, out = new Float64Array(spec.n)) {
  const f = extractFeatures(board, linesCleared, featureScratch);
  for (const b of spec.board) out[b.pos] = f[b.fi] / b.scale;
  for (const h of spec.held) out[h.pos] = held === h.piece ? 1 : 0;
  return out;
}

/** Every legal move for a piece: [{ move, board, lines, state }], in placements() order. */
export function nextStates(board, pieceIdx, spec, held = null) {
  const out = [];
  for (const move of placements(board, pieceIdx)) {
    const after = copyBoard(board);
    const lines = playMove(after, pieceIdx, move);
    out.push({ move, board: after, lines, state: boardState(after, lines, spec, held) });
  }
  return out;
}

/**
 * The option ({ state, ... } from nextStates) the network values most, i.e.
 * the 1-ply choice; the first one on a tie. Undefined if there are none.
 */
export function greedyPick(options, net) {
  let pick;
  let best = -Infinity;
  for (const o of options) {
    const v = net.predict(o.state);
    if (v > best) {
      best = v;
      pick = o;
    }
  }
  return pick;
}

/* --------------------------------------------------------------- network */

const ADAM_BETA1 = 0.9;
const ADAM_BETA2 = 0.999;
const ADAM_EPSILON = 1e-8;

/**
 * A plain multi-layer perceptron with ReLU between layers and one output,
 * plus the backward pass and the Adam optimiser needed to train it. A forward
 * pass is "multiply by a matrix, add a bias, ReLU, repeat".
 */
export class MLP {
  /**
   * @param {number[]} sizes  layer sizes, e.g. [4, 64, 64, 1]
   * @param {object} [opts]
   * @param {() => number} [opts.rand=Math.random]  for the initial weights
   * @param {readonly string[]} [opts.features=FEATURE_SETS.basic]  input names, in order
   * @param {boolean} [opts.scaled=false]  divide inputs by FEATURE_SCALE
   */
  constructor(sizes, { rand = Math.random, features = FEATURE_SETS.basic, scaled = false } = {}) {
    if (features.length !== sizes[0]) throw new Error(`${features.length} features but ${sizes[0]} inputs`);
    if (sizes.at(-1) !== 1) throw new Error('the network must have exactly one output');
    this.sizes = sizes;
    this.features = features;
    this.scaled = scaled;
    this.inputSpec = makeInputSpec(features, scaled);
    this.W = []; // W[l] is sizes[l] x sizes[l+1], row-major
    this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l];
      const nOut = sizes[l + 1];
      const limit = Math.sqrt(6 / (nIn + nOut)); // Xavier-uniform initialisation
      const w = new Float64Array(nIn * nOut);
      for (let i = 0; i < w.length; i++) w[i] = (rand() * 2 - 1) * limit;
      this.W.push(w);
      this.b.push(new Float64Array(nOut));
    }
    // Adam state: step count and running moments of every weight and bias
    this.t = 0;
    this.mW = this.W.map((w) => new Float64Array(w.length));
    this.vW = this.W.map((w) => new Float64Array(w.length));
    this.mb = this.b.map((b) => new Float64Array(b.length));
    this.vb = this.b.map((b) => new Float64Array(b.length));
    // Reused buffers: predict() runs hundreds of thousands of times a second.
    this.layerBuffers = sizes.slice(1).map((n) => new Float64Array(n));
    this.stateScratch = new Float64Array(sizes[0]);
  }

  /** V(input): the network's score for one input vector. */
  predict(x) {
    let a = x;
    for (let l = 0; l < this.W.length; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const out = this.layerBuffers[l];
      const W = this.W[l];
      const bias = this.b[l];
      for (let j = 0; j < nOut; j++) out[j] = bias[j];
      for (let i = 0; i < nIn; i++) {
        const xi = a[i];
        if (xi === 0) continue;
        const row = i * nOut;
        for (let j = 0; j < nOut; j++) out[j] += xi * W[row + j];
      }
      if (l < this.W.length - 1) for (let j = 0; j < nOut; j++) if (out[j] < 0) out[j] = 0; // ReLU
      a = out;
    }
    return a[0];
  }

  /**
   * One Adam step on a batch, minimising the mean squared error between the
   * outputs and `targets`. X is a flat array of batch x inputs. Returns the
   * loss before the step.
   */
  trainStep(X, targets, lr) {
    const B = targets.length;
    const L = this.W.length;

    // forward, keeping every layer's activations
    const acts = [X];
    for (let l = 0; l < L; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const W = this.W[l];
      const bias = this.b[l];
      const prev = acts[l];
      const out = new Float64Array(B * nOut);
      for (let n = 0; n < B; n++) {
        const o = n * nOut;
        for (let j = 0; j < nOut; j++) out[o + j] = bias[j];
        for (let i = 0; i < nIn; i++) {
          const xi = prev[n * nIn + i];
          if (xi === 0) continue;
          const row = i * nOut;
          for (let j = 0; j < nOut; j++) out[o + j] += xi * W[row + j];
        }
        if (l < L - 1) for (let j = 0; j < nOut; j++) if (out[o + j] < 0) out[o + j] = 0;
      }
      acts.push(out);
    }

    // loss, and its gradient with respect to the output
    const pred = acts[L];
    let loss = 0;
    let grad = new Float64Array(B);
    for (let n = 0; n < B; n++) {
      const d = pred[n] - targets[n];
      loss += d * d;
      grad[n] = (2 * d) / B;
    }
    loss /= B;

    // backward: the chain rule, layer by layer
    const gW = [];
    const gb = [];
    for (let l = L - 1; l >= 0; l--) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const prev = acts[l];
      const W = this.W[l];
      const dW = new Float64Array(nIn * nOut);
      const db = new Float64Array(nOut);
      const dPrev = new Float64Array(B * nIn);
      for (let n = 0; n < B; n++) {
        for (let j = 0; j < nOut; j++) {
          const g = grad[n * nOut + j];
          if (g === 0) continue;
          db[j] += g;
          for (let i = 0; i < nIn; i++) {
            dW[i * nOut + j] += prev[n * nIn + i] * g;
            dPrev[n * nIn + i] += W[i * nOut + j] * g;
          }
        }
      }
      // the ReLU below lets gradient through only where its output was > 0
      if (l > 0) for (let k = 0; k < dPrev.length; k++) if (prev[k] <= 0) dPrev[k] = 0;
      gW[l] = dW;
      gb[l] = db;
      grad = dPrev;
    }

    // Adam: per-weight step sizes from running averages of the gradient (m)
    // and of its square (v), with bias correction for the first steps
    this.t++;
    const c1 = 1 - ADAM_BETA1 ** this.t;
    const c2 = 1 - ADAM_BETA2 ** this.t;
    for (let l = 0; l < L; l++) {
      for (const [p, g, m, v] of [
        [this.W[l], gW[l], this.mW[l], this.vW[l]],
        [this.b[l], gb[l], this.mb[l], this.vb[l]],
      ]) {
        for (let k = 0; k < p.length; k++) {
          m[k] = ADAM_BETA1 * m[k] + (1 - ADAM_BETA1) * g[k];
          v[k] = ADAM_BETA2 * v[k] + (1 - ADAM_BETA2) * g[k] * g[k];
          p[k] -= (lr * (m[k] / c1)) / (Math.sqrt(v[k] / c2) + ADAM_EPSILON);
        }
      }
    }
    return loss;
  }

  /** The model file format. The input names are stored under `state`. */
  toJSON() {
    return {
      kind: 'dqn',
      state: this.features,
      scaled: this.scaled,
      sizes: this.sizes,
      W: this.W.map((w) => Array.from(w)),
      b: this.b.map((b) => Array.from(b)),
    };
  }

  static fromJSON(json) {
    // the initial weights are thrown away, so any rand will do
    const net = new MLP(json.sizes, { rand: () => 0.5, features: json.state, scaled: json.scaled === true });
    net.W = json.W.map((w) => Float64Array.from(w));
    net.b = json.b.map((b) => Float64Array.from(b));
    return net;
  }
}

/** Loads a trained model (models/dqn-model.json). */
export function loadDqn(json) {
  if (json.kind !== 'dqn') throw new Error(`not a DQN model (kind: ${json.kind})`);
  return MLP.fromJSON(json);
}

/* ---------------------------------------------------------------- search */

/** Reward for one move. Training uses exactly this, so search and learning agree. */
export const moveReward = (lines) => 1 + 10 * lines * lines;

/** Discount per move: a reward one move later is worth 0.99 of one now. */
export const GAMMA = 0.99;

/**
 * Score of a move after which the next piece has nowhere to go. Finite (not
 * -Infinity) so that averages over the 7 possible pieces stay comparable:
 * a move that loses to one piece in seven still beats one that loses to two.
 */
export const LOST_GAME_SCORE = -1e9;

/** How many candidates get the expensive deeper look in 3-ply search, per level. */
const DEEP_BEAM = Object.freeze({ first: 6, second: 4 });

// Scratch board for bestFollow. Its input buffer lives on the network
// (net.stateScratch), sized for that network's inputs.
const followBoard = createBoard();

/**
 * The best the `piece` can do on `board`:
 *   max over its placements of  reward + gamma x V(board after it),
 * or -Infinity if it has nowhere to go. Same as looping over nextStates(),
 * but with one scratch board instead of a new board per placement.
 */
export function bestFollow(board, piece, net, gamma = GAMMA) {
  let best = -Infinity;
  for (const m of placements(board, piece)) {
    followBoard.set(board);
    const lines = playMove(followBoard, piece, m);
    const v = moveReward(lines) + gamma * net.predict(boardState(followBoard, lines, net.inputSpec, null, net.stateScratch));
    if (v > best) best = v;
  }
  return best;
}

/**
 * Expected value of `board` when the next piece is unknown: the best each of
 * the 7 pieces could do, averaged. A piece with nowhere to go counts as
 * LOST_GAME_SCORE.
 */
export function expectedFollow(board, net, gamma = GAMMA) {
  let sum = 0;
  for (let p = 0; p < PIECES.length; p++) {
    const f = bestFollow(board, p, net, gamma);
    sum += f === -Infinity ? LOST_GAME_SCORE : f;
  }
  return sum / PIECES.length;
}

/** reward + gamma x follow, or LOST_GAME_SCORE when the follower had nowhere to go. */
const withFollow = (lines, follow, gamma) => (follow === -Infinity ? LOST_GAME_SCORE : moveReward(lines) + gamma * follow);

/**
 * The network's best placement for one piece, as a function the hold
 * decision can call: (piece, follower) -> { move, score } | null.
 *
 *   1-ply:  score = V(board after the move)
 *   2-ply:  score = reward + gamma x (the best the follower can do), or
 *           reward + gamma x V(board after) when the follower is unknown
 *   3-ply:  see bestPlacementDeep
 */
function bestPlacement(net, board, { lookahead = false, deep = false, gamma = GAMMA } = {}) {
  if (deep) return bestPlacementDeep(net, board, gamma);
  return (piece, follower) => {
    let best = null;
    for (const s of nextStates(board, piece, net.inputSpec)) {
      let v;
      if (!lookahead) v = net.predict(s.state);
      else if (follower == null) v = moveReward(s.lines) + gamma * net.predict(s.state);
      else v = withFollow(s.lines, bestFollow(s.board, follower, net, gamma), gamma);
      if (!best || v > best.score) best = { move: s.move, score: v };
    }
    return best;
  };
}

/**
 * The best `piece` can do on `board` looking one more piece ahead:
 *   max over its placements of  reward + gamma x expectedFollow(board after).
 * Only the `beam` most promising placements (by reward + gamma x V) get that
 * deeper look; that is what keeps it fast enough to play.
 */
function bestFollowDeep(board, piece, net, gamma, beam) {
  const cands = [];
  for (const m of placements(board, piece)) {
    const after = copyBoard(board);
    const lines = playMove(after, piece, m);
    const quick = moveReward(lines) + gamma * net.predict(boardState(after, lines, net.inputSpec, null, net.stateScratch));
    cands.push({ after, lines, quick });
  }
  if (cands.length === 0) return -Infinity;
  cands.sort((a, b) => b.quick - a.quick);
  let best = -Infinity;
  for (const c of cands.slice(0, beam)) {
    const v = moveReward(c.lines) + gamma * expectedFollow(c.after, net, gamma);
    if (v > best) best = v;
  }
  return best;
}

/**
 * 3-ply bestPlacement: (piece, follower) -> { move, score } | null.
 * For each placement of `piece`, the follower (the visible next piece) is
 * placed, and then the unknown piece after it is averaged over all 7:
 *
 *   score = reward + gamma x max over follower placements of
 *             [ reward + gamma x average over 7 pieces of their best
 *                 ( reward + gamma x V(board) ) ]
 *
 * Beam search: every placement is first scored the 2-ply way, and only the
 * best DEEP_BEAM.first of them (and DEEP_BEAM.second follower placements
 * each) get the deeper look. Without a known follower it is 2-ply, with the
 * follower averaged over all 7 pieces.
 */
function bestPlacementDeep(net, board, gamma) {
  return (piece, follower) => {
    const states = nextStates(board, piece, net.inputSpec);
    if (states.length === 0) return null;
    for (const s of states) {
      s.shallow =
        follower == null
          ? moveReward(s.lines) + gamma * expectedFollow(s.board, net, gamma)
          : withFollow(s.lines, bestFollow(s.board, follower, net, gamma), gamma);
    }
    if (follower == null) {
      let best = states[0];
      for (const s of states) if (s.shallow > best.shallow) best = s;
      return { move: best.move, score: best.shallow };
    }
    states.sort((a, b) => b.shallow - a.shallow);
    let best = null;
    for (const s of states.slice(0, DEEP_BEAM.first)) {
      const v = withFollow(s.lines, bestFollowDeep(s.board, follower, net, gamma, DEEP_BEAM.second), gamma);
      if (!best || v > best.score) best = { move: s.move, score: v };
    }
    return best;
  };
}

/* --------------------------------------------------------------- choosers */

/**
 * The network's placement for this piece, or null if nothing fits.
 *
 * 1-ply by default. With { lookahead: true } and a known `next` piece it is
 * 2-ply: every placement of the current piece is scored as
 *
 *     reward(move 1) + gamma x best over move 2 of [ reward(move 2) + gamma x V(board after both) ]
 *
 * which is the Bellman equation from training, unrolled one step with the
 * next piece known instead of guessed. Same network, no retraining. Without
 * a known next piece, lookahead has nothing to look at and it plays 1-ply.
 */
export function chooseDqnMove(board, pieceIdx, next, net, { lookahead = false, gamma = GAMMA } = {}) {
  const best = bestPlacement(net, board, { lookahead: lookahead && next != null, gamma });
  return best(pieceIdx, next)?.move ?? null;
}

/**
 * The network's action: a placement, HOLD, or null (game over). Pass the
 * chooser's 4th argument ({ held, canHold }) as `holdCtx`; without it this
 * never holds. { lookahead: true } is 2-ply, { deep: true } is 3-ply.
 *
 * A network trained with hold inputs decides 1-ply only when it may hold
 * (lookahead and deep are not used there).
 */
export function chooseDqnAction(board, pieceIdx, next, net, holdCtx = null, { lookahead = false, deep = false } = {}) {
  if (holdCtx && net.inputSpec.held.length > 0) return chooseWithHoldInputs(board, pieceIdx, next, net, holdCtx);
  const best = bestPlacement(net, board, { lookahead: lookahead || deep, deep });
  if (!holdCtx) return best(pieceIdx, next)?.move ?? null;
  return decideWithHold(best, pieceIdx, next, holdCtx, lookahead || deep);
}

/**
 * The hold decision for a network that also sees which piece is in the hold
 * box (trained with --hold). Every option is scored on the board it leaves
 * AND the piece it leaves in the box:
 *   place the current piece  -> the box keeps what it had
 *   hold, place the other    -> the box now holds the current piece
 * When the piece we would get is unknown (empty box, no preview), the hold
 * side is the average over all 7 pieces.
 */
function chooseWithHoldInputs(board, current, next, net, { held, canHold }) {
  const best = (piece, heldAfter) => {
    let b = null;
    for (const s of nextStates(board, piece, net.inputSpec, heldAfter)) {
      const v = net.predict(s.state);
      if (!b || v > b.score) b = { move: s.move, score: v };
    }
    return b;
  };
  const place = best(current, held);
  if (!canHold) return place ? place.move : null;
  let holdScore;
  if (held != null) holdScore = best(held, current)?.score ?? -Infinity;
  else if (next != null) holdScore = best(next, current)?.score ?? -Infinity;
  else {
    holdScore = 0;
    // each term divided before adding (not the sum divided): keeps the trained results bit-identical
    for (let p = 0; p < PIECES.length; p++) holdScore += (best(p, current)?.score ?? -Infinity) / PIECES.length;
  }
  if (!place) return holdScore > -Infinity ? HOLD : null;
  return holdScore > place.score ? HOLD : place.move;
}
