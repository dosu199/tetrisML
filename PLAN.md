# Plan — Tetris ML

Last updated: 14 October 2026 (v0.7).

**Goal:** a neural network that plays Tetris well and learns by itself.

## Where we are (v0.7)

- Network: `dqn-model.json` (4 features, trained on random pieces, seed 124).
- New: **2-ply lookahead** (`chooseDqnMove(..., next, { lookahead: true })`).
  50 fresh games:
  - random pieces + preview: DQN 2-ply **0/50** lost, bot 0/50, DQN 1-ply 16/50
  - garbage every 5 pieces: DQN 2-ply **321** pieces, bot 254, DQN 1-ply 148
  - 7-bag, no preview, cap 10,000: DQN 1/50 lost, bot 8/50
- Tried without success: 8 features, a stricter checkpoint check (it picked
  the same checkpoints). A typical
  training run lands around 30/50 lost on random pieces without preview; our
  network (16–22/50) is a lucky seed.

## Possible next steps

1. **Speed up the 2-ply search** (reuse memory in `predict`, skip duplicate
   placements). It is about 30× slower than 1-ply.
2. **More stable training**, so that a typical run is as good as our lucky
   one: lower the learning rate toward the end, or average the weights of the
   last checkpoints. Measure with `train_many.js`.
3. **Train with garbage rows**, the only mode where everyone still loses.
4. **Later:** the raw board (240 cells) as input.

## Rules we work by

- Decide on at least 50 games with fresh seeds (90000+). 10 games is not
  enough. When everyone reaches the 2,000-piece cap, raise it (`--cap 10000`).
- Train more than one seed before concluding anything.
- Do not change direction without agreeing first.
