/**
 * tetris-ai.js - board features and the search bot. No DOM.
 *
 * The bot scores a board with a plain weighted sum of 8 features:
 *
 *     score(board) = sum over i of  weights[i] x features[i]
 *
 * and plays the placement whose resulting board scores highest, optionally
 * looking one piece ahead (2-ply). The weights in models/weights.json were
 * tuned by a genetic algorithm (scripts/legacy-ga/).
 *
 * The features are also the network's inputs (tetris-dqn.js uses 4 of them),
 * and decideWithHold() is shared by the bot and the network.
 */

import {
  PIECES,
  PLAY_LEFT,
  PLAY_RIGHT,
  PLAY_COLS,
  PLAY_BOTTOM,
  idx,
  createBoard,
  columnHeights,
  playMove,
  placements,
  HOLD,
} from './tetris-core.js';

/* --------------------------------------------------------------- features */

export const FEATURE_NAMES = [
  'aggregateHeight', // sum of all column heights
  'linesCleared', // rows cleared by the move that produced this board
  'holes', // empty cells with at least one filled cell above them
  'bumpiness', // sum of |height difference| between neighbouring columns
  'maxHeight', // tallest column
  'rowTransitions', // filled/empty flips scanning each row (walls count as filled)
  'colTransitions', // filled/empty flips scanning each column (floor counts as filled)
  'wells', // for each column lower than both neighbours (a wall counts as
  //          infinitely high) of depth d: 1 + 2 + ... + d, so deep wells cost more
];

export const NUM_FEATURES = FEATURE_NAMES.length;

/** Position of each feature in extractFeatures' output, e.g. FEATURE.holes === 2. */
export const FEATURE = Object.freeze(Object.fromEntries(FEATURE_NAMES.map((name, i) => [name, i])));

/**
 * Hand-tuned starting weights, from Yiyuan Lee's 2013 "near perfect" Tetris
 * bot (4 features); the other four start at 0. Used when models/weights.json
 * cannot be loaded.
 */
export const PUBLISHED_WEIGHTS = Object.freeze([
  -0.510066, // aggregateHeight
  0.760666, // linesCleared
  -0.35663, // holes
  -0.184483, // bumpiness
  0, // maxHeight
  0, // rowTransitions
  0, // colTransitions
  0, // wells
]);

// Scratch buffers. They make these functions non-reentrant, which is fine:
// nothing here calls itself, and JavaScript runs one call at a time.
const heightScratch = new Int32Array(PLAY_COLS);
const featureScratch = new Float64Array(NUM_FEATURES);

/**
 * The feature vector of a board, in FEATURE_NAMES order.
 *
 * `linesCleared` is passed in rather than measured: by the time we see the
 * board, the cleared rows are already gone. Bumpiness, holes and wells use
 * column heights (the top block), not the number of filled cells.
 */
export function extractFeatures(board, linesCleared, out = new Float64Array(NUM_FEATURES)) {
  const heights = columnHeights(board, heightScratch);

  let aggregateHeight = 0;
  let maxHeight = 0;
  for (let i = 0; i < PLAY_COLS; i++) {
    aggregateHeight += heights[i];
    if (heights[i] > maxHeight) maxHeight = heights[i];
  }

  let bumpiness = 0;
  for (let i = 0; i < PLAY_COLS - 1; i++) {
    bumpiness += Math.abs(heights[i] - heights[i + 1]);
  }

  let holes = 0;
  for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
    const top = PLAY_BOTTOM - heights[c - PLAY_LEFT] + 1;
    for (let r = top; r <= PLAY_BOTTOM; r++) {
      if (board[idx(r, c)] === 0) holes++;
    }
  }

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

  let wells = 0;
  for (let i = 0; i < PLAY_COLS; i++) {
    const left = i === 0 ? Infinity : heights[i - 1];
    const right = i === PLAY_COLS - 1 ? Infinity : heights[i + 1];
    const depth = Math.min(left, right) - heights[i];
    if (depth > 0) wells += (depth * (depth + 1)) / 2;
  }

  out[FEATURE.aggregateHeight] = aggregateHeight;
  out[FEATURE.linesCleared] = linesCleared;
  out[FEATURE.holes] = holes;
  out[FEATURE.bumpiness] = bumpiness;
  out[FEATURE.maxHeight] = maxHeight;
  out[FEATURE.rowTransitions] = rowTransitions;
  out[FEATURE.colTransitions] = colTransitions;
  out[FEATURE.wells] = wells;
  return out;
}

/** score = weights . features. Higher is better. */
export function evaluate(board, linesCleared, weights) {
  const f = extractFeatures(board, linesCleared, featureScratch);
  let score = 0;
  for (let i = 0; i < NUM_FEATURES; i++) score += weights[i] * f[i];
  return score;
}

/* ----------------------------------------------------------------- search */

const firstPlyBoard = createBoard();
const secondPlyBoard = createBoard();

/** Best score `pieceIdx` can reach on `board`, counting `carriedLines` from the move before. */
function bestScoreFor(board, pieceIdx, carriedLines, weights) {
  let best = -Infinity; // stays -Infinity when the piece has nowhere to go
  for (const m of placements(board, pieceIdx)) {
    secondPlyBoard.set(board);
    const lines = playMove(secondPlyBoard, pieceIdx, m);
    const score = evaluate(secondPlyBoard, carriedLines + lines, weights);
    if (score > best) best = score;
  }
  return best;
}

/**
 * The bot's move: { rotation, col, row, score }, or null when the game is
 * lost (the piece does not fit, or with lookahead, no placement leaves room
 * for the next piece).
 *
 * With `next` and lookahead on (the default) it is 2-ply: each placement is
 * scored by the best board the next piece can then reach, and the line clears
 * of both moves are added up, so setting up a clear is credited too. About
 * 40x slower than 1-ply.
 *
 * Note: with lookahead the bot gives up one piece earlier than the network
 * would (the network still plays a doomed move); benchmarks undercount the
 * bot by at most one piece per lost game.
 */
export function chooseMove(board, pieceIdx, nextIdx, weights, { lookahead = true } = {}) {
  let best = null;
  for (const m of placements(board, pieceIdx)) {
    firstPlyBoard.set(board);
    const lines = playMove(firstPlyBoard, pieceIdx, m);
    const score =
      lookahead && nextIdx != null
        ? bestScoreFor(firstPlyBoard, nextIdx, lines, weights)
        : evaluate(firstPlyBoard, lines, weights);
    if (score > (best?.score ?? -Infinity)) best = { rotation: m.rotation, col: m.col, row: m.row, score };
  }
  return best;
}

/**
 * Like chooseMove, but may hold: pass the chooser's 4th argument
 * ({ held, canHold }) as `holdCtx`. Returns a placement, HOLD or null.
 * Without holdCtx it is exactly chooseMove.
 */
export function chooseMoveAction(board, pieceIdx, nextIdx, weights, holdCtx = null, { lookahead = true } = {}) {
  if (!holdCtx) return chooseMove(board, pieceIdx, nextIdx, weights, { lookahead });
  const best = (piece, follower) => {
    const m = chooseMove(board, piece, follower, weights, { lookahead });
    return m && { move: m, score: m.score };
  };
  return decideWithHold(best, pieceIdx, nextIdx, holdCtx, lookahead);
}

/* ------------------------------------------------------------------- hold */

/**
 * The hold decision, shared by the bot and the network: "place the current
 * piece" against "hold, then place what you get". Returns a placement, HOLD,
 * or null when neither is possible.
 *
 *   best(piece, follower) -> { move, score } | null
 *     the chooser's best placement for `piece` on the current board, where
 *     `follower` is the piece after it (null = unknown). Higher = better.
 *   usesFollower  whether best() looks at the follower at all (2-ply or more)
 *
 * When the piece you would get, or its follower, is still unknown, the hold
 * side is the average over all 7 pieces (any piece that cannot be placed
 * makes it -Infinity), so both sides are scored the same way.
 */
export function decideWithHold(best, current, next, { held, canHold }, usesFollower = false) {
  const place = best(current, next);
  if (!canHold) return place ? place.move : null;

  const averageOverPieces = (scoreOf) => {
    let sum = 0;
    for (let p = 0; p < PIECES.length; p++) {
      const r = scoreOf(p);
      if (!r) return -Infinity;
      sum += r.score;
    }
    return sum / PIECES.length;
  };
  let holdScore;
  if (held != null) {
    // the held piece comes back; the next piece is unchanged
    holdScore = best(held, next)?.score ?? -Infinity;
  } else if (next != null) {
    // the next piece comes in; the one after it is not known yet
    holdScore = usesFollower ? averageOverPieces((f) => best(next, f)) : best(next, null)?.score ?? -Infinity;
  } else {
    // no preview and an empty box: the piece we would get is unknown
    holdScore = averageOverPieces((p) => best(p, null));
  }

  if (!place) return holdScore > -Infinity ? HOLD : null;
  return holdScore > place.score ? HOLD : place.move;
}
