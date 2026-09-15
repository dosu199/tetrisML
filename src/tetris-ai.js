/**
 * tetris-ai.js - board features, the linear evaluation function, and the
 * placement search. Still no DOM: this runs identically in node and in the
 * browser, which is what lets you train it.
 *
 * The evaluation is a plain dot product:
 *
 *     score(board) = sum over i of  weights[i] * features[i]
 *
 * In Phase 1 you stop hand-picking `weights` and let a genetic algorithm or
 * the cross-entropy method find them. Nothing else in this file has to change
 * for that to happen - that is the whole point of the refactor.
 */

import {
  PIECES,
  PLAY_LEFT,
  PLAY_RIGHT,
  PLAY_COLS,
  PLAY_BOTTOM,
  idx,
  columnHeights,
  copyBoard,
  placePiece,
  clearLines,
  placements,
} from './tetris-core.js';

/* ---------------------------------------------------------------- features */

export const FEATURE_NAMES = [
  'aggregateHeight', // sum of all column heights
  'linesCleared', // rows cleared by the move that produced this board
  'holes', // empty cells with at least one filled cell above them
  'bumpiness', // sum of |height difference| between adjacent columns
  'maxHeight', // tallest column
  'rowTransitions', // filled/empty flips scanning each row
  'colTransitions', // filled/empty flips scanning each column
  'wells', // summed depth of one-wide gaps
];

export const NUM_FEATURES = FEATURE_NAMES.length;

/**
 * The weights you were using. They are the published El-Tetris values, not
 * values you derived - replacing them with learned ones is Phase 1.
 * The four extra features start at 0, so this vector reproduces your current
 * bot's behaviour exactly while leaving the GA room to discover whether the
 * extras earn their keep.
 */
export const PUBLISHED_WEIGHTS = [
  -0.510066, // aggregateHeight
  0.760666, // linesCleared
  -0.35663, // holes
  -0.184483, // bumpiness
  0, // maxHeight
  0, // rowTransitions
  0, // colTransitions
  0, // wells
];

const heightScratch = new Int32Array(PLAY_COLS);

/**
 * Extracts the feature vector for a board.
 *
 * `linesCleared` is passed in rather than measured, because by the time we see
 * the board the rows are already gone.
 *
 * Note on bumpiness: the original compared the *number of filled cells* per
 * column. Once a column contains a hole, cell count and column height are
 * different numbers, so that measured something other than surface roughness.
 * This uses heights.
 */
export function extractFeatures(board, linesCleared, out = new Float64Array(NUM_FEATURES)) {
  const heights = columnHeights(board, heightScratch);

  let aggregate = 0;
  let maxHeight = 0;
  for (let i = 0; i < PLAY_COLS; i++) {
    aggregate += heights[i];
    if (heights[i] > maxHeight) maxHeight = heights[i];
  }

  let bumpiness = 0;
  for (let i = 0; i < PLAY_COLS - 1; i++) {
    bumpiness += Math.abs(heights[i] - heights[i + 1]);
  }

  // holes: empty cells that sit below the top of their column
  let holes = 0;
  for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
    const top = PLAY_BOTTOM - heights[c - PLAY_LEFT] + 1;
    for (let r = top; r <= PLAY_BOTTOM; r++) {
      if (board[idx(r, c)] === 0) holes++;
    }
  }

  // transitions: how ragged the board is. Walls count as filled.
  let rowTransitions = 0;
  for (let r = 0; r <= PLAY_BOTTOM; r++) {
    let prev = 1; // left wall
    for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
      const cur = board[idx(r, c)] !== 0 ? 1 : 0;
      if (cur !== prev) rowTransitions++;
      prev = cur;
    }
    if (prev !== 1) rowTransitions++; // right wall
  }

  let colTransitions = 0;
  for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
    let prev = 0; // open sky above the stack
    for (let r = 0; r <= PLAY_BOTTOM; r++) {
      const cur = board[idx(r, c)] !== 0 ? 1 : 0;
      if (cur !== prev) colTransitions++;
      prev = cur;
    }
    if (prev !== 1) colTransitions++; // floor
  }

  // wells: a column lower than both neighbours, counted by depth
  let wells = 0;
  for (let i = 0; i < PLAY_COLS; i++) {
    const left = i === 0 ? Infinity : heights[i - 1];
    const right = i === PLAY_COLS - 1 ? Infinity : heights[i + 1];
    const depth = Math.min(left, right) - heights[i];
    if (depth > 0) wells += (depth * (depth + 1)) / 2;
  }

  out[0] = aggregate;
  out[1] = linesCleared;
  out[2] = holes;
  out[3] = bumpiness;
  out[4] = maxHeight;
  out[5] = rowTransitions;
  out[6] = colTransitions;
  out[7] = wells;
  return out;
}

const featureScratch = new Float64Array(NUM_FEATURES);

/** score = weights . features. Higher is better. */
export function evaluate(board, linesCleared, weights) {
  const f = extractFeatures(board, linesCleared, featureScratch);
  let score = 0;
  for (let i = 0; i < NUM_FEATURES; i++) score += weights[i] * f[i];
  return score;
}

/* ------------------------------------------------------------------ search */

/** Best achievable score for `pieceIdx` on `board`, carrying earlier clears. */
function bestScoreFor(board, pieceIdx, carriedLines, weights) {
  const moves = placements(board, pieceIdx);
  if (moves.length === 0) return -Infinity; // dead

  let best = -Infinity;
  const rotations = PIECES[pieceIdx].rotations;
  for (let i = 0; i < moves.length; i++) {
    const m = moves[i];
    const next = copyBoard(board);
    placePiece(next, rotations[m.rotation], m.row, m.col);
    const lines = clearLines(next);
    const score = evaluate(next, carriedLines + lines, weights);
    if (score > best) best = score;
  }
  return best;
}

/**
 * Picks where to put the current piece.
 *
 * Returns { rotation, col, row, score } or null when nothing is legal, which
 * is the game-over condition.
 *
 * `nextIdx` enables the 2-ply lookahead you already had. Two important fixes
 * over the original:
 *
 *  1. The simulation runs on a copy. The original passed the live `gameBoard`
 *     into breakeRowsForAi(), so every candidate placement it considered was
 *     deleting rows from the real game and reporting clears from the wrong
 *     board.
 *  2. Line clears from both plies are summed before scoring, so a move that
 *     sets up a clear on the next piece is credited for it.
 *
 * Lookahead costs roughly 40x the search time. Turn it off during training
 * (it barely changes the *ranking* of weight vectors) and back on to play.
 */
export function chooseMove(board, pieceIdx, nextIdx, weights, { lookahead = true } = {}) {
  const moves = placements(board, pieceIdx);
  if (moves.length === 0) return null;

  const rotations = PIECES[pieceIdx].rotations;
  let best = null;
  let bestScore = -Infinity;

  for (let i = 0; i < moves.length; i++) {
    const m = moves[i];
    const after = copyBoard(board);
    placePiece(after, rotations[m.rotation], m.row, m.col);
    const lines = clearLines(after);

    const score =
      lookahead && nextIdx != null
        ? bestScoreFor(after, nextIdx, lines, weights)
        : evaluate(after, lines, weights);

    if (score > bestScore) {
      bestScore = score;
      best = { rotation: m.rotation, col: m.col, row: m.row, score };
    }
  }

  // Every move leads to death; take the first legal one so the game ends
  // cleanly on the next piece rather than throwing.
  if (best === null) {
    const m = moves[0];
    return { rotation: m.rotation, col: m.col, row: m.row, score: -Infinity };
  }
  return best;
}