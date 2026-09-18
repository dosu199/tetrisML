/**
 * tetris-policy.js - the neural network's side of Phase 2.
 *
 * Phase 1 learned eight numbers inside a scoring function that a human wrote.
 * Phase 2 throws that scoring function away and learns the *decision itself*:
 * a model that looks at a board and names the move, with no features, no
 * search, and no evaluation function in between.
 *
 * This file owns three things, and it owns them because they must agree
 * exactly between the JavaScript that generates the data, the Python that
 * trains on it, and the JavaScript that plays with the result:
 *
 *   1. How a board becomes a vector of numbers      -> buildInput()
 *   2. How a placement becomes a class label        -> moveClass()
 *   3. How a trained model picks a legal move       -> choosePolicyMove()
 *
 * A mismatch in any of those three is the classic way this phase fails
 * silently: the model trains to 90% accuracy and then plays like it has never
 * seen Tetris, because column 7 in training means something different from
 * column 7 at inference. Keeping all three definitions in one file, imported
 * by everything else, is the defence.
 */

import {
  PLAY_LEFT,
  PLAY_RIGHT,
  PLAY_COLS,
  PLAY_BOTTOM,
  PIECES,
  idx,
  placements,
} from './tetris-core.js';

export const NUM_PIECES = PIECES.length; // 7
export const MAX_ROTATIONS = 4;
export const NUM_CLASSES = MAX_ROTATIONS * PLAY_COLS; // 40

/* ------------------------------------------------------------------- input */

/**
 * Board cells, then a one-hot for the current piece, then one for the next.
 *
 *   240  playfield occupancy, row-major from the top, 1 = filled
 *     7  current piece
 *     7  next piece
 *   ---
 *   254
 *
 * Occupancy is deliberately binary rather than the colour value. Which colour
 * a settled block used to be has no bearing on where the next piece should go,
 * and feeding it in would invite the model to learn superstitions about it.
 */
export const BOARD_CELLS = (PLAY_BOTTOM + 1) * PLAY_COLS; // 24 * 10 = 240
export const INPUT_SIZE = BOARD_CELLS + NUM_PIECES * 2; // 254

export function buildInput(board, pieceIdx, nextIdx, out = new Float64Array(INPUT_SIZE)) {
  out.fill(0);
  let k = 0;
  for (let r = 0; r <= PLAY_BOTTOM; r++) {
    for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
      out[k++] = board[idx(r, c)] !== 0 ? 1 : 0;
    }
  }
  out[k + pieceIdx] = 1;
  k += NUM_PIECES;
  if (nextIdx != null) out[k + nextIdx] = 1;
  return out;
}

/** CSV header, in exactly the order buildInput() writes. */
export function inputColumns() {
  const names = [];
  for (let r = 0; r <= PLAY_BOTTOM; r++) {
    for (let c = 0; c < PLAY_COLS; c++) names.push(`b${r}_${c}`);
  }
  for (let p = 0; p < NUM_PIECES; p++) names.push(`piece_${PIECES[p].name}`);
  for (let p = 0; p < NUM_PIECES; p++) names.push(`next_${PIECES[p].name}`);
  return names;
}

/* ------------------------------------------------------------------ labels */

/** (rotation, absolute column) -> a single class in 0..39. */
export const moveClass = (rotation, col) => rotation * PLAY_COLS + (col - PLAY_LEFT);

export const decodeClass = (k) => ({
  rotation: Math.floor(k / PLAY_COLS),
  col: (k % PLAY_COLS) + PLAY_LEFT,
});

/* --------------------------------------------------------------- inference */

/**
 * Loads a model exported by train_policy.py. Deliberately a hand-written
 * forward pass rather than a library: a multi-layer perceptron is a matrix
 * multiply, an activation, and repeat, and writing it out once makes that
 * concrete. It also means the browser needs no dependencies at all.
 */
export function loadPolicy(json) {
  const { layers, scaler } = json;

  function forward(x) {
    let a = x;
    if (scaler) {
      a = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) a[i] = (x[i] - scaler.mean[i]) / scaler.scale[i];
    }

    for (const layer of layers) {
      const { W, b, activation } = layer;
      const out = new Float64Array(b.length);
      for (let j = 0; j < b.length; j++) out[j] = b[j];
      for (let i = 0; i < a.length; i++) {
        const xi = a[i];
        if (xi === 0) continue; // the board is mostly zeros; skip the row
        const row = W[i];
        for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
      }
      if (activation === 'relu') {
        for (let j = 0; j < out.length; j++) if (out[j] < 0) out[j] = 0;
      } else if (activation === 'tanh') {
        for (let j = 0; j < out.length; j++) out[j] = Math.tanh(out[j]);
      }
      a = out;
    }
    return a; // raw scores; we only ever argmax them, so softmax is redundant
  }

  return {
    classes: json.classes,
    scores(board, pieceIdx, nextIdx) {
      return forward(buildInput(board, pieceIdx, nextIdx));
    },
  };
}

/**
 * Picks the model's highest-scoring move *among the legal ones*.
 *
 * The masking matters more than it looks. A network asked to choose among 40
 * classes will happily name a placement that does not fit on this board, and
 * an unmasked policy spends much of its life proposing impossible moves. The
 * legal set is cheap to compute - we already have placements() - so there is
 * no reason to let the model guess at legality. What we are asking it to learn
 * is judgement, not the rules.
 */
export function choosePolicyMove(board, pieceIdx, nextIdx, policy) {
  const legal = placements(board, pieceIdx);
  if (legal.length === 0) return null;

  const raw = policy.scores(board, pieceIdx, nextIdx);
  const classIndex = new Map(policy.classes.map((c, i) => [c, i]));

  let best = null;
  let bestScore = -Infinity;
  for (const m of legal) {
    const i = classIndex.get(moveClass(m.rotation, m.col));
    // A class the training set never contained cannot be scored; treat it as
    // the worst option rather than crashing.
    const score = i === undefined ? -Infinity : raw[i];
    if (score > bestScore) {
      bestScore = score;
      best = m;
    }
  }
  return best ?? legal[0];
}

/* ------------------------------------------------- the value-network variant

 * Behaviour cloning above asks: "board in, which of 40 moves out?" That turns
 * out to be a brutal thing to learn from raw cells, because the label is an
 * abstract action index whose meaning changes with the piece, and the network
 * has to infer the whole geometry of fitting from scratch.
 *
 * This variant asks a much easier question: "here is a board that COULD
 * result from a move - how good is it?" Then at play time we enumerate the
 * legal placements ourselves, build each resulting board, score them all, and
 * take the best. The network never has to know the rules; it only has to
 * develop taste.
 *
 * This is knowledge distillation: the target is the expert's own evaluation
 * score, so the network is learning to reproduce a function we already have.
 * That sounds circular until you notice what it must do to succeed - it has to
 * learn to see holes, bumpiness and stack height in a grid of raw cells, with
 * nobody telling it those concepts exist.
 *
 * It is also the exact shape Phase 3 needs. A DQN over afterstates is this
 * network with a different training signal.
 */

import { copyBoard, placePiece, clearLines } from './tetris-core.js';

export const VALUE_INPUT_SIZE = BOARD_CELLS + 1; // 240 cells + lines cleared

export function buildValueInput(board, linesCleared, out = new Float64Array(VALUE_INPUT_SIZE)) {
  let k = 0;
  for (let r = 0; r <= PLAY_BOTTOM; r++) {
    for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
      out[k++] = board[idx(r, c)] !== 0 ? 1 : 0;
    }
  }
  out[k] = linesCleared;
  return out;
}

export function valueColumns() {
  const names = [];
  for (let r = 0; r <= PLAY_BOTTOM; r++) {
    for (let c = 0; c < PLAY_COLS; c++) names.push(`a${r}_${c}`);
  }
  names.push('lines');
  return names;
}

export function loadValueNet(json) {
  const { layers } = json;
  const scratch = new Float64Array(VALUE_INPUT_SIZE);

  function forward(x) {
    let a = x;
    for (const { W, b, activation } of layers) {
      const out = new Float64Array(b.length);
      for (let j = 0; j < b.length; j++) out[j] = b[j];
      for (let i = 0; i < a.length; i++) {
        const xi = a[i];
        if (xi === 0) continue;
        const row = W[i];
        for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
      }
      if (activation === 'relu') for (let j = 0; j < out.length; j++) if (out[j] < 0) out[j] = 0;
      a = out;
    }
    return a[0];
  }

  return { score: (board, lines) => forward(buildValueInput(board, lines, scratch)) };
}

/** Enumerate placements, build each afterstate, let the network judge. */
export function chooseValueMove(board, pieceIdx, valueNet) {
  const legal = placements(board, pieceIdx);
  if (legal.length === 0) return null;

  let best = null;
  let bestScore = -Infinity;
  for (const m of legal) {
    const after = copyBoard(board);
    placePiece(after, PIECES[pieceIdx].rotations[m.rotation], m.row, m.col);
    const lines = clearLines(after);
    const score = valueNet.score(after, lines);
    if (score > bestScore) {
      bestScore = score;
      best = m;
    }
  }
  return best;
}
