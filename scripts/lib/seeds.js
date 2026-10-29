/**
 * seeds.js - which games each job plays. A game is fixed by its seed, so
 * every player gets exactly the same pieces. Each job has its own range,
 * and the ranges never overlap: a network is never chosen and then scored
 * on the same games (the train / validation / test split, done with seeds).
 *
 *   training           train-dqn.js training games        1,000,000+
 *   checkpoint         greedy test inside training           500,000+   picks each run's best checkpoint
 *   checkpointGarbage  the same, garbage games (--garbage)   600,000+
 *   selection          train-many.js                          70,000+   picks the best run
 *   benchmark          hardmode.js --seed 90000                90,000+   the numbers we report
 *   quickBenchmark     hardmode.js default                     20,000+
 *   gaHeldOut          legacy-ga/compare.js                    10,000+
 *
 * (The genetic algorithm trains on seeds 1-8: see DEFAULTS in legacy-ga/ga.js.)
 */
export const SEED_BASE = Object.freeze({
  training: 1_000_000,
  checkpoint: 500_000,
  checkpointGarbage: 600_000,
  selection: 70_000,
  benchmark: 90_000,
  quickBenchmark: 20_000,
  gaHeldOut: 10_000,
});
