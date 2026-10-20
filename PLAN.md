# Plan — Tetris ML

Last updated: 20 October 2026 (v0.9).

**Goal:** a neural network that plays Tetris well and learns by itself.

## Where we are (v0.9)

- Network: `dqn-model.json` (4 features, random pieces, seed 8), picked from
  8 seeds with `train_many.js`. 50 fresh games:
  - random pieces, no preview: DQN 12/50 lost, bot 9/50 (old network 16/50)
  - 7-bag, no preview, cap 10,000: DQN **0/50**, bot 8/50
  - random pieces with preview + lookahead: DQN 2/50 lost, bot 0/50
  - garbage every 5 pieces, with lookahead: DQN **319** pieces, bot 254
- Stability experiments (8 seeds each): learning-rate decay, weight
  averaging, more steps per game. **None** made a typical run better. Seed
  spread (14–48 lost) is the real limit; train many seeds and select.
- Garbage training (`--garbage 5`, 16 seeds): typical run a bit better (30 vs
  34 lost) and fewer very bad runs, but no network beat the shipped one. The
  model is unchanged.
- **We have hit the ceiling of "4 numbers + 1-ply".** Further gains need a
  network that sees more of the board, or a different learning signal.

## Possible next steps

1. **Raw board as input (a CNN).** The network sees all 240 cells instead of
   4 numbers and learns its own features: wells, overhangs, where an I piece
   fits. This is the real "deep" in deep Q-learning, and the biggest
   remaining step. Training is much slower in plain JavaScript (hours, not
   minutes), and it needs a careful plan first.
2. **n-step targets:** learn from the reward of the next few moves instead
   of just one. This often makes DQN learn faster and steadier. It is a small
   change.
3. **Speed up the 2-ply search** (about 30x slower than 1-ply).

## Rules we work by

- Decide on at least 50 games with fresh seeds (90000+). 10 games is not
  enough. When everyone reaches the 2,000-piece cap, raise it (`--cap 10000`).
- Train more than one seed before concluding anything.
- Do not change direction without agreeing first.
