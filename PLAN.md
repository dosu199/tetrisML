# Plan — Tetris ML

Last updated: 20 October 2026 (v0.8).

**Goal:** a neural network that plays Tetris well and learns by itself.

## Where we are (v0.8)

- Network: `dqn-model.json` (4 features, random pieces, seed 8), picked from
  8 seeds with `train_many.js`. 50 fresh games:
  - random pieces, no preview: DQN 12/50 lost, bot 9/50 (old network 16/50)
  - 7-bag, no preview, cap 10,000: DQN **0/50**, bot 8/50
  - random pieces with preview + lookahead: DQN 2/50 lost, bot 0/50
  - garbage every 5 pieces, with lookahead: DQN **319** pieces, bot 254
- Stability experiments (8 seeds each): learning-rate decay, weight
  averaging, more steps per game. **None** made a typical run better. Seed
  spread (14–48 lost) is the real limit; train many seeds and select.

## Possible next steps

1. **Train with garbage rows**, the only mode where everyone still loses.
2. **Speed up the 2-ply search** (about 30x slower than 1-ply).
3. **A different learning signal:** a reward that also penalises holes or
   height directly (reward shaping), or n-step targets. These change *what*
   it learns, not just how steadily.
4. **Later:** the raw board (240 cells) as input.

## Rules we work by

- Decide on at least 50 games with fresh seeds (90000+). 10 games is not
  enough. When everyone reaches the 2,000-piece cap, raise it (`--cap 10000`).
- Train more than one seed before concluding anything.
- Do not change direction without agreeing first.
