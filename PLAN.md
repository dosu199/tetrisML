# Plan — Tetris ML

Last updated: 13 October 2026.

**Goal:** a neural network that plays Tetris well and learns by itself.

## Where we are (v0.4)

- The old imitation pipeline (CSV data, Python training, CNN, DAgger) has
  been removed. In its place is a single DQN in plain JavaScript:
  `src/tetris-dqn.js` + `train_dqn.js`, about 450 lines in total.
- Training takes about 5 minutes. The best network is saved in `dqn-model.json`.
- Results, 50 fresh games:
  - 7-bag, no preview: DQN **5/50** topped out, bot 1/50.
  - Fully random pieces: DQN 43/50, bot 9/50. It never trained on random
    pieces.

## Possible next steps (in order, each one small)

1. **Watch it in the browser.** Use the snippet in the README.
2. **Train on random pieces too.** Add a `--memoryless` option to
   `train_dqn.js` that uses `MemorylessGen` instead of `Bag`. Expectation:
   much better in the "random" mode. About 15 lines of code and 5 minutes of
   training.
3. **Stability.** We tried input normalization plus a target network. The
   best result was about the same (865 vs 1000 in the test), and the network
   still collapsed at the end, so we did not keep it. Best-checkpointing is
   enough for now.
4. **Next piece in the state.** Add the next piece (7 numbers, one-hot) to the
   input, so the network can plan two pieces ahead the way the bot does.

## Rules we work by

- Decide on at least 50 games with fresh seeds (90000+). 10 games is not
  enough.
- Do not change direction without agreeing first.
