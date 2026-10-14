# Plan — Tetris ML

Last updated: 14 October 2026.

**Goal:** a neural network that plays Tetris well and learns by itself.

## Where we are (v0.5)

- One DQN in plain JavaScript: `src/tetris-dqn.js` + `train_dqn.js`.
- `dqn-model.json` (4 features, seed 125), 50 fresh games, no preview:
  - 7-bag, up to 10,000 pieces: DQN **0/50** topped out, bot 8/50. The
    network beats the bot.
  - Fully random pieces, up to 2,000: DQN 25/50, bot 9/50.
- Step 1 tried (8 features + scaling): it **did not help**. The seed matters
  more than the features. See README, "What we learned".

## Possible next steps (in order)

1. **Train on random pieces too.** Add a `--memoryless` option to
   `train_dqn.js` (use `MemorylessGen` instead of `Bag`). Target: below 25/50
   in the "random" mode.
2. **Look one piece ahead** (2-ply search with the network as the evaluator).
   No training needed.
3. **Better checkpoint selection.** Use more test games inside training
   (e.g. 20 instead of 5, with a higher cap), and train 3+ seeds per setting.
   After that, retry `--features full` with longer training (`--epochs 6000`).
4. **Later:** the raw board (240 cells) as input instead of hand-picked
   features.

## Rules we work by

- Decide on at least 50 games with fresh seeds (90000+). 10 games is not
  enough. When everyone reaches the 2,000-piece cap, raise it (`--cap 10000`).
- Train more than one seed before concluding anything.
- Do not change direction without agreeing first.
