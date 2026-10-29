/**
 * ga.js - the genetic algorithm that tuned the search bot's 8 weights
 * (models/weights.json). Kept for the record: the project's learning now
 * happens in the neural network (scripts/train-dqn.js).
 *
 * The algorithm is a straightforward genetic algorithm:
 *
 *   1. Start with a population of random weight vectors.
 *   2. Score each one by playing a fixed set of games (its "fitness").
 *   3. Repeatedly pick two good ones, blend them, nudge the result randomly,
 *      and use the children to replace the worst of the population.
 *   4. Repeat for a few dozen generations.
 *
 * No gradients, no neural network, no calculus. It is the simplest thing that
 * genuinely deserves to be called machine learning: the parameters are derived
 * from evidence rather than chosen by hand.
 *
 * Three design decisions worth understanding, because they are the difference
 * between this working and this quietly not working:
 *
 * FIXED SEEDS. Every candidate in a generation plays the *same* games. Tetris
 * outcomes vary enormously, so if candidates played different games you would
 * be ranking luck as much as skill, and the population would converge on
 * whoever drew the friendliest pieces.
 *
 * NORMALISED WEIGHTS. Only the direction of the weight vector matters - scaling
 * every weight by 10 picks exactly the same moves. Normalising to unit length
 * stops the population from drifting off to huge magnitudes where mutation has
 * no meaningful effect.
 *
 * AN ADAPTIVE PIECE CAP. With good weights this bot does not lose, so "play
 * until death" never terminates. Fitness is instead "lines cleared within N
 * pieces". But once most of the population maxes that out, fitness stops
 * discriminating and learning stalls - so when the population starts
 * saturating, N doubles.
 */

import { Worker } from 'node:worker_threads';
import { NUM_FEATURES, chooseMove } from '../../src/tetris-ai.js';
import { mulberry32, playEpisode } from '../../src/tetris-core.js';

/* ------------------------------------------------------------------ games */

/**
 * One 7-bag game of the search bot with these weights, capped at `maxPieces`.
 * The cap matters: with decent weights the bot does not lose, so "play until
 * death" would never end. Fitness is "lines cleared within N pieces".
 */
export function playBotGame(weights, seed, { maxPieces = 500, lookahead = true } = {}) {
  const chooser = (board, current, next) => chooseMove(board, current, next, weights, { lookahead });
  const r = playEpisode(chooser, seed, { cap: maxPieces });
  return { lines: r.lines, pieces: r.pieces, toppedOut: r.died };
}

/* ----------------------------------------------------------------- vectors */

/** Scale to unit length. Direction is all that matters; magnitude is not. */
export function normalize(weights) {
  let norm = 0;
  for (const w of weights) norm += w * w;
  norm = Math.sqrt(norm) || 1;
  return Array.from(weights, (w) => w / norm);
}

export function randomVector(rand) {
  const v = new Array(NUM_FEATURES);
  for (let i = 0; i < NUM_FEATURES; i++) v[i] = rand() * 2 - 1;
  return normalize(v);
}

/** Box-Muller: turns two uniforms into a normally distributed sample. */
function gaussian(rand) {
  let u = 0;
  while (u === 0) u = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

/* --------------------------------------------------------------- operators */

/**
 * Blend two parents in proportion to how well they played, then renormalise.
 * A parent that cleared twice as many lines contributes twice as much.
 */
export function crossover(a, b) {
  const child = new Array(NUM_FEATURES);
  const wa = Math.max(a.fitness, 0) + 1e-9;
  const wb = Math.max(b.fitness, 0) + 1e-9;
  for (let i = 0; i < NUM_FEATURES; i++) {
    child[i] = wa * a.weights[i] + wb * b.weights[i];
  }
  return normalize(child);
}

/** Nudge one randomly chosen weight. This is where new ideas come from. */
export function mutate(weights, rand, rate, strength) {
  if (rand() >= rate) return weights;
  const out = weights.slice();
  const i = Math.floor(rand() * NUM_FEATURES);
  out[i] += gaussian(rand) * strength;
  return normalize(out);
}

/**
 * Tournament selection: sample a handful of the population at random and take
 * the best two. Cheaper than sorting the whole population and, more usefully,
 * it lets mediocre candidates occasionally win, which preserves diversity.
 */
export function selectParents(population, rand, tournamentSize) {
  const pool = [];
  for (let i = 0; i < tournamentSize; i++) {
    pool.push(population[Math.floor(rand() * population.length)]);
  }
  pool.sort((x, y) => y.fitness - x.fitness);
  return [pool[0], pool[1] ?? pool[0]];
}

/* -------------------------------------------------------------- evaluation */

/** Single-threaded fitness: play every seed, average the lines cleared. */
export function fitnessOf(weights, seeds, maxPieces) {
  let lines = 0;
  let deaths = 0;
  for (const seed of seeds) {
    const r = playBotGame(weights, seed, { maxPieces, lookahead: false });
    lines += r.lines;
    if (r.toppedOut) deaths++;
  }
  return { fitness: lines / seeds.length, survival: 1 - deaths / seeds.length };
}

/** A tiny worker pool. Falls back to inline evaluation when size <= 1. */
export function createEvaluator(size) {
  if (size <= 1) {
    return {
      async evaluate(candidates, seeds, maxPieces) {
        return candidates.map((c) => fitnessOf(c.weights, seeds, maxPieces));
      },
      async close() {},
    };
  }

  const url = new URL('./ga-worker.js', import.meta.url);
  let rejectCurrent = null; // rejects the evaluate() call in progress
  const workers = Array.from({ length: size }, () => {
    const w = new Worker(url);
    w.unref();
    // One permanent error listener per worker. Attaching a fresh one inside
    // evaluate() would leak a listener per generation and trip node's
    // MaxListenersExceededWarning after ten of them.
    w.on('error', (err) => rejectCurrent?.(err));
    return w;
  });

  return {
    evaluate(candidates, seeds, maxPieces) {
      return new Promise((resolve, reject) => {
        const results = new Array(candidates.length);
        let nextTask = 0;
        let done = 0;

        const dispatch = (worker) => {
          if (nextTask >= candidates.length) return;
          const index = nextTask++;
          worker.once('message', (msg) => {
            results[index] = msg;
            if (++done === candidates.length) resolve(results);
            else dispatch(worker);
          });
          worker.postMessage({ weights: candidates[index].weights, seeds, maxPieces });
        };

        rejectCurrent = reject;
        for (const w of workers) dispatch(w);
      });
    },
    async close() {
      await Promise.all(workers.map((w) => w.terminate()));
    },
  };
}

/* ---------------------------------------------------------------- the loop */

export const DEFAULTS = {
  populationSize: 50,
  generations: 40,
  seeds: [1, 2, 3, 4, 5, 6, 7, 8],
  maxPieces: 500,
  maxPiecesCeiling: 20000,
  replaceFraction: 0.3, // share of the population replaced each generation
  tournamentSize: 10,
  mutationRate: 0.25,
  mutationStrength: 0.2,
  randomSeed: 1234,
  workers: 1,
};

export async function runGA(config = {}, onGeneration = () => {}) {
  const cfg = { ...DEFAULTS, ...config };
  const rand = mulberry32(cfg.randomSeed);
  const evaluator = createEvaluator(cfg.workers);
  let maxPieces = cfg.maxPieces;

  try {
    // generation 0: a population of random guesses
    let population = Array.from({ length: cfg.populationSize }, () => ({
      weights: randomVector(rand),
      fitness: 0,
      survival: 0,
    }));

    let scores = await evaluator.evaluate(population, cfg.seeds, maxPieces);
    population.forEach((c, i) => Object.assign(c, scores[i]));
    population.sort((a, b) => b.fitness - a.fitness);

    const childCount = Math.max(1, Math.round(cfg.populationSize * cfg.replaceFraction));
    const history = [];

    for (let gen = 1; gen <= cfg.generations; gen++) {
      // breed
      const children = [];
      for (let i = 0; i < childCount; i++) {
        const [p1, p2] = selectParents(population, rand, cfg.tournamentSize);
        const weights = mutate(crossover(p1, p2), rand, cfg.mutationRate, cfg.mutationStrength);
        children.push({ weights, fitness: 0, survival: 0 });
      }

      scores = await evaluator.evaluate(children, cfg.seeds, maxPieces);
      children.forEach((c, i) => Object.assign(c, scores[i]));

      // the worst `childCount` of the old population make way
      population = population.slice(0, cfg.populationSize - childCount).concat(children);
      population.sort((a, b) => b.fitness - a.fitness);

      const median = population[population.length >> 1].fitness;

      // Saturation check: 0.4 lines per piece is the physical ceiling (four
      // cells per piece, ten per row). Once the middle of the population is
      // near it, fitness has stopped telling candidates apart - so make the
      // test harder and re-score everyone against the new bar.
      const ceiling = maxPieces * 0.4;
      let raised = false;
      if (median > ceiling * 0.9 && maxPieces * 2 <= cfg.maxPiecesCeiling) {
        maxPieces *= 2;
        raised = true;
        scores = await evaluator.evaluate(population, cfg.seeds, maxPieces);
        population.forEach((c, i) => Object.assign(c, scores[i]));
        population.sort((a, b) => b.fitness - a.fitness);
      }

      const row = {
        generation: gen,
        best: population[0].fitness,
        median: population[population.length >> 1].fitness,
        mean: population.reduce((a, c) => a + c.fitness, 0) / population.length,
        survival: population[0].survival,
        maxPieces,
        raised,
        weights: population[0].weights.slice(),
      };
      history.push(row);
      onGeneration(row);
    }

    return { best: population[0], population, history, maxPieces, config: cfg };
  } finally {
    await evaluator.close();
  }
}

