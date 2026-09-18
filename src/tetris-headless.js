/**
 * tetris-headless.js - plays complete games as fast as the CPU allows.
 *
 * This is the file that makes Phase 1 possible. A genetic algorithm needs
 * something like 20,000 full games to converge; rendered at 600ms per row that
 * is months of wall clock, and here it is minutes.
 */

import { createBoard, Bag, PIECES, placePiece, clearLines } from './tetris-core.js';
import { chooseMove, PUBLISHED_WEIGHTS, NUM_FEATURES } from './tetris-ai.js';

/**
 * Plays one game to the end and reports what happened.
 *
 * maxPieces matters more than it looks. With decent weights this bot does not
 * lose - runs of millions of lines are normal - so "play until death" is not a
 * usable fitness function, it just hangs. Capping pieces turns fitness into
 * "lines cleared in N pieces", which ranks weight vectors perfectly well and
 * terminates. 500 is plenty of signal for early generations; raise it later
 * when the population gets good enough that everyone maxes out.
 */
export function playGame(weights = PUBLISHED_WEIGHTS, seed = 0, options = {}) {
  const { maxPieces = 500, lookahead = true } = options;

  const board = createBoard();
  const bag = new Bag(seed);

  let current = bag.next();
  let next = bag.next();
  let lines = 0;
  let pieces = 0;

  while (pieces < maxPieces) {
    const move = chooseMove(board, current, next, weights, { lookahead });
    if (move === null || move.score === -Infinity) {
      return { lines, pieces, toppedOut: true, seed };
    }

    placePiece(board, PIECES[current].rotations[move.rotation], move.row, move.col);
    lines += clearLines(board);
    pieces++;

    current = next;
    next = bag.next();
  }

  return { lines, pieces, toppedOut: false, seed };
}

/**
 * Fitness for one weight vector, averaged over several fixed seeds.
 *
 * Tetris outcomes are extremely high-variance, so a single game tells you
 * almost nothing about whether one weight vector beats another. Always average
 * over the *same* seeds for every candidate, otherwise you are partly ranking
 * luck. This is the function your GA will call.
 */
export function evaluateWeights(weights, seeds = [1, 2, 3, 4, 5], options = {}) {
  let totalLines = 0;
  let deaths = 0;
  for (const seed of seeds) {
    const r = playGame(weights, seed, { lookahead: false, ...options });
    totalLines += r.lines;
    if (r.toppedOut) deaths++;
  }
  return {
    fitness: totalLines / seeds.length,
    survivalRate: 1 - deaths / seeds.length,
  };
}

export function randomWeights(rand = Math.random) {
  const w = new Float64Array(NUM_FEATURES);
  for (let i = 0; i < NUM_FEATURES; i++) w[i] = rand() * 2 - 1;
  return w;
}

/** Weight vectors are only meaningful up to scale; normalising helps the GA. */
export function normalizeWeights(weights) {
  let norm = 0;
  for (const w of weights) norm += w * w;
  norm = Math.sqrt(norm) || 1;
  return Array.from(weights, (w) => w / norm);
}
