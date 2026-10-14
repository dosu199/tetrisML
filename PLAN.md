# Plan — Tetris ML

Last updated: 14 October 2026 (v0.6).

**Goal:** a neural network that plays Tetris well and learns by itself.

## Where we are (v0.6)

- One DQN in plain JavaScript: `src/tetris-dqn.js` + `train_dqn.js`.
- `dqn-model.json`: 4 features, trained on random pieces, seed 124. 50 fresh
  games, no preview:
  - 7-bag, up to 10,000 pieces: DQN **1/50** topped out, bot 8/50.
  - Fully random pieces, up to 2,000: DQN **16/50**, bot 9/50 (was 25/50).
- Done: 8 features + scaling (did not help); training on random pieces
  (helped, 25 → 16).
- Lesson: the difference between seeds (16/50 vs 33/50) is still bigger than
  any setting. Checkpoint selection is the weakest part.

## Possible next steps (in order)

1. **Better checkpoint selection.** The check inside training uses 5 games
   capped at 1,000 pieces; good networks all hit 1,000, so it cannot tell
   them apart. Use 20 games with a cap of 3,000, and train 3+ seeds per setting
   automatically (one command), keeping the best on a separate test.
2. **Look one piece ahead** (2-ply search with the network as the
   evaluator). No training needed.
3. Retry `--features full` with longer training, once 1 is in place.
4. **Later:** the raw board (240 cells) as input.

## Rules we work by

- Decide on at least 50 games with fresh seeds (90000+). 10 games is not
  enough. When everyone reaches the 2,000-piece cap, raise it (`--cap 10000`).
- Train more than one seed before concluding anything.
- Do not change direction without agreeing first.
