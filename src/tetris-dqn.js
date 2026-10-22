/**
 * tetris-dqn.js - the neural network that plays, and the code to run it.
 *
 * The approach follows vietnh1009/Tetris-deep-Q-learning-pytorch, rewritten
 * here in plain JavaScript so it needs no PyTorch and runs in the browser:
 *
 *   1. For the current piece, try every rotation x column.
 *   2. For each resulting board, compute 4 numbers:
 *        lines cleared, holes, bumpiness, total height
 *   3. A small neural network (4 -> 64 -> 64 -> 1) turns those 4 numbers into
 *      one score: "how good is this board for the rest of the game".
 *   4. Play the move with the highest score.
 *
 * The network learns that score by itself, from the game's reward - see
 * train_dqn.js. Nobody tells it what a good board looks like.
 *
 * No DOM access in this file, so the same code trains in node and plays in
 * the browser.
 */

import { PIECES, copyBoard, createBoard, placePiece, clearLines, placements, decideWithHold, HOLD } from './tetris-core.js';
import { extractFeatures, FEATURE_NAMES, NUM_FEATURES } from './tetris-ai.js';

/* ----------------------------------------------------------------- state */

// Which board numbers the network sees. Names come from FEATURE_NAMES in
// tetris-ai.js; a trained model stores its list, so old models keep working.
export const FEATURE_SETS = {
  // the 4 from the reference implementation
  basic: ['linesCleared', 'holes', 'bumpiness', 'aggregateHeight'],
  // all 8 the search bot uses: the 4 above plus the shape of the surface
  full: ['linesCleared', 'holes', 'bumpiness', 'aggregateHeight', 'maxHeight', 'rowTransitions', 'colTransitions', 'wells'],
};
// the 4 basic numbers plus 7 inputs saying which piece is in the hold box
FEATURE_SETS.hold = [...FEATURE_SETS.basic, ...PIECES.map((p) => `held:${p.name}`)];
export const DEFAULT_FEATURES = FEATURE_SETS.full;

const ALIASES = { totalHeight: 'aggregateHeight' }; // name used by v0.4 models

// Feature scaling: divide each feature by roughly its largest value, so every
// input lands around 0-1. Without this, big inputs (total height up to 240)
// drown out small ones (lines 0-4) and training barely moves.
export const FEATURE_SCALE = {
  linesCleared: 4,
  holes: 100,
  bumpiness: 30,
  aggregateHeight: 240,
  maxHeight: 24,
  rowTransitions: 120,
  colTransitions: 60,
  wells: 50,
};

/** Positions in extractFeatures' output for a list of feature names. */
export function featureIndex(names) {
  return names.map((n) => {
    const i = FEATURE_NAMES.indexOf(ALIASES[n] ?? n);
    if (i === -1) throw new Error(`unknown feature: ${n}`);
    return i;
  });
}

const scratch = new Float64Array(NUM_FEATURES);

/**
 * How a network reads a board: which features (idx) and what to divide each
 * one by (scale). Models from v0.4 were trained unscaled, so scale is 1 there.
 */
export function makeInput(names, scaled = true) {
  const input = { n: names.length, board: [], held: [] };
  names.forEach((name, pos) => {
    if (name.startsWith('held:')) {
      // one input per piece type: 1 if that piece is in the hold box, else 0
      input.held.push({ pos, piece: PIECES.findIndex((p) => p.name === name.slice(5)) });
    } else {
      const [fi] = featureIndex([name]);
      input.board.push({ pos, fi, scale: scaled ? FEATURE_SCALE[ALIASES[name] ?? name] : 1 });
    }
  });
  if (input.held.some((h) => h.piece < 0)) throw new Error(`unknown piece in ${names.join(', ')}`);
  return input;
}

/** The numbers the network sees for a board after a move (and, for a network
 * with hold inputs, the piece left in the hold box). */
export function boardState(board, linesCleared, input, out = new Float64Array(input.n), held = null) {
  const f = extractFeatures(board, linesCleared, scratch);
  for (const b of input.board) out[b.pos] = f[b.fi] / b.scale;
  for (const h of input.held) out[h.pos] = held === h.piece ? 1 : 0;
  return out;
}

/** Every legal move for a piece, with the board it produces and its state. */
export function nextStates(board, pieceIdx, input, held = null) {
  const out = [];
  for (const move of placements(board, pieceIdx)) {
    const after = copyBoard(board);
    placePiece(after, PIECES[pieceIdx].rotations[move.rotation], move.row, move.col);
    const lines = clearLines(after);
    out.push({ move, board: after, lines, state: boardState(after, lines, input, undefined, held) });
  }
  return out;
}

/* ---------------------------------------------------------------- network */

/**
 * A plain multi-layer perceptron with ReLU between layers, plus the backward
 * pass and Adam optimiser needed to train it. Small enough to read in one go:
 * a forward pass is "multiply by a matrix, add a bias, ReLU, repeat".
 */
export class MLP {
  constructor(sizes, rand = Math.random, features = DEFAULT_FEATURES, scaled = true) {
    this.sizes = sizes;
    this.features = features; // names of the inputs, in order
    this.scaled = scaled;
    this.input = makeInput(features, scaled);
    if (features.length !== sizes[0]) throw new Error(`${features.length} features but ${sizes[0]} inputs`);
    this.W = []; // W[l] is sizes[l] x sizes[l+1], row-major
    this.b = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const nIn = sizes[l];
      const nOut = sizes[l + 1];
      // Xavier-uniform initialisation, same as the reference implementation
      const limit = Math.sqrt(6 / (nIn + nOut));
      const w = new Float64Array(nIn * nOut);
      for (let i = 0; i < w.length; i++) w[i] = (rand() * 2 - 1) * limit;
      this.W.push(w);
      this.b.push(new Float64Array(nOut));
    }
    this.t = 0;
    this.mW = this.W.map((w) => new Float64Array(w.length));
    this.vW = this.W.map((w) => new Float64Array(w.length));
    this.mb = this.b.map((b) => new Float64Array(b.length));
    this.vb = this.b.map((b) => new Float64Array(b.length));
  }

  /** Score one input vector. */
  predict(x) {
    // Reuses one buffer per layer instead of allocating new arrays on every
    // call; predict runs hundreds of thousands of times per second.
    if (!this._bufs) this._bufs = this.sizes.slice(1).map((n) => new Float64Array(n));
    let a = x;
    for (let l = 0; l < this.W.length; l++) {
      const nIn = this.sizes[l];
      const nOut = this.sizes[l + 1];
      const out = this._bufs[l];
      const W = this.W[l];
      const bias = this.b[l];
      for (let j = 0; j < nOut; j++) out[j] = bias[j];
      for (let i = 0; i < nIn; i++) {
        const xi = a[i];
        if (xi === 0) continue;
        const row = i * nOut;
        for (let j = 0; j < nOut; j++) out[j] += xi * W[row + j];
      }
      if (l < this.W.length - 1) for (let j = 0; j < nOut; j++) if (out[j] < 0) out[j] = 0;
      a = out;
    }
    return a[0];
  }


  /**
   * One gradient step on a batch, minimising mean squared error between the
   * network's outputs and `targets`. X is a flat array of batch x inputs.
   * Returns the loss before the step.
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
      const prev = acts[l];
      const out = new Float64Array(B * nOut);
      for (let n = 0; n < B; n++) {
        const o = n * nOut;
        for (let j = 0; j < nOut; j++) out[o + j] = this.b[l][j];
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

    // loss and its gradient with respect to the output
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
      // ReLU of the layer below lets gradient through only where it was > 0
      if (l > 0) for (let k = 0; k < dPrev.length; k++) if (prev[k] <= 0) dPrev[k] = 0;
      gW[l] = dW;
      gb[l] = db;
      grad = dPrev;
    }

    // Adam
    this.t++;
    const b1 = 0.9;
    const b2 = 0.999;
    const c1 = 1 - b1 ** this.t;
    const c2 = 1 - b2 ** this.t;
    for (let l = 0; l < L; l++) {
      for (const [p, g, m, v] of [
        [this.W[l], gW[l], this.mW[l], this.vW[l]],
        [this.b[l], gb[l], this.mb[l], this.vb[l]],
      ]) {
        for (let k = 0; k < p.length; k++) {
          m[k] = b1 * m[k] + (1 - b1) * g[k];
          v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k];
          p[k] -= (lr * (m[k] / c1)) / (Math.sqrt(v[k] / c2) + 1e-8);
        }
      }
    }
    return loss;
  }

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
    const net = new MLP(json.sizes, () => 0.5, json.state, json.scaled === true);
    net.W = json.W.map((w) => Float64Array.from(w));
    net.b = json.b.map((b) => Float64Array.from(b));
    return net;
  }
}

/* ------------------------------------------------------------------ play */

/** Load a trained model (dqn-model.json). */
export function loadDqn(json) {
  if (json.kind !== 'dqn') throw new Error(`not a DQN model (kind: ${json.kind})`);
  return MLP.fromJSON(json);
}

const GAMMA_DEFAULT = 0.99;

const followBoard = createBoard();
const followState = new Float64Array(NUM_FEATURES);

/**
 * The best the follower piece can do on `board`:
 *   max over its placements of  reward + gamma x V(board after it),
 * or -Infinity if it has nowhere to go. Same result as looping over
 * nextStates(), but reuses one scratch board instead of allocating a board
 * and an object per placement, which makes the 2-ply search much faster.
 */
export function bestFollow(board, piece, net, gamma = GAMMA_DEFAULT) {
  let best = -Infinity;
  const rotations = PIECES[piece].rotations;
  for (const m of placements(board, piece)) {
    followBoard.set(board);
    placePiece(followBoard, rotations[m.rotation], m.row, m.col);
    const lines = clearLines(followBoard);
    const v = moveReward(lines) + gamma * net.predict(boardState(followBoard, lines, net.input, followState));
    if (v > best) best = v;
  }
  return best;
}

/** Reward for one move, the same formula training uses. */
export const moveReward = (lines) => 1 + 10 * lines * lines;

/**
 * The network's move for this piece, or null if no move fits (game over).
 *
 * Without `next` (or with lookahead off) it is 1-ply: score every placement of
 * the current piece with the network and take the best.
 *
 * With `next` and { lookahead: true } it is 2-ply: for every placement of the
 * current piece, also try every placement of the next piece, and score the
 * pair as
 *
 *     reward(move 1) + 0.99 x best over move 2 of [ reward(move 2) + 0.99 x V(board after both) ]
 *
 * which is the Bellman equation from training, unrolled one step with the
 * next piece known instead of guessed. Same network, no retraining - only the
 * way moves are chosen changes. About 30x slower, still fast enough to watch.
 */
export function chooseDqnMove(board, pieceIdx, net, next = null, { lookahead = false, gamma = 0.99 } = {}) {
  let best = null;
  let bestScore = -Infinity;
  const twoPly = lookahead && next != null;
  for (const s of nextStates(board, pieceIdx, net.input)) {
    let v;
    if (!twoPly) {
      v = net.predict(s.state);
    } else {
      const follow = bestFollow(s.board, next, net, gamma);
      // no room for the next piece: this move loses the game
      v = follow === -Infinity ? -1e9 : moveReward(s.lines) + gamma * follow;
    }
    if (v > bestScore) {
      bestScore = v;
      best = s.move;
    }
  }
  return best;
}

/**
 * The network's best placement for one piece, as a function the hold
 * decision can call: (piece, follower) -> { move, score } | null.
 *
 *   no lookahead:   score = V(board after the move)
 *   lookahead:      score = reward + 0.99 x (the best the follower can do), or
 *                   reward + 0.99 x V(board after) when the follower is unknown
 */
export function dqnBest(net, board, { lookahead = false, gamma = GAMMA_DEFAULT } = {}) {
  return (piece, follower) => {
    let best = null;
    for (const s of nextStates(board, piece, net.input)) {
      let v;
      if (!lookahead) {
        v = net.predict(s.state);
      } else if (follower == null) {
        v = moveReward(s.lines) + gamma * net.predict(s.state);
      } else {
        const follow = bestFollow(s.board, follower, net, gamma);
        v = follow === -Infinity ? -1e9 : moveReward(s.lines) + gamma * follow;
      }
      if (!best || v > best.score) best = { move: s.move, score: v };
    }
    return best;
  };
}

/**
 * Like chooseDqnMove, but understands hold: pass the 4th chooser argument
 * ({ held, canHold }) as `holdCtx`. Returns a placement or HOLD.
 * Without holdCtx it plays exactly like chooseDqnMove.
 */
export function chooseDqnAction(board, pieceIdx, net, next = null, holdCtx = null, { lookahead = false } = {}) {
  // a network trained with hold sees the hold box: decide with that knowledge
  if (holdCtx && net.input.held.length > 0) return chooseWithHoldInputs(board, pieceIdx, net, next, holdCtx);
  const best = dqnBest(net, board, { lookahead });
  if (!holdCtx) return best(pieceIdx, next)?.move ?? null;
  return decideWithHold(best, pieceIdx, next, holdCtx.held, holdCtx.canHold, lookahead);
}

/**
 * The hold decision for a network that also sees which piece is in the hold
 * box (trained with --hold). Every option is scored on the board it leaves
 * AND the piece it leaves in the box:
 *   place the current piece  -> the box keeps what it had
 *   hold, place the other    -> the box now holds the current piece
 * When the piece we'd get is unknown (empty box, no preview) we average over
 * all 7. Decides on the current position only (no 2-ply lookahead yet).
 */
function chooseWithHoldInputs(board, cur, net, next, { held, canHold }) {
  const best = (piece, heldAfter) => {
    let b = null;
    for (const s of nextStates(board, piece, net.input, heldAfter)) {
      const v = net.predict(s.state);
      if (!b || v > b.score) b = { move: s.move, score: v };
    }
    return b;
  };
  const place = best(cur, held);
  if (!canHold) return place ? place.move : null;
  let holdScore;
  if (held != null) holdScore = best(held, cur)?.score ?? -Infinity;
  else if (next != null) holdScore = best(next, cur)?.score ?? -Infinity;
  else {
    holdScore = 0;
    for (let p = 0; p < PIECES.length; p++) holdScore += (best(p, cur)?.score ?? -Infinity) / PIECES.length;
  }
  if (!place) return holdScore > -Infinity ? HOLD : null;
  return holdScore > place.score ? HOLD : place.move;
}
