# Plan — Tetris ML

Last updated: 22 October 2026 (v0.15).

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

## Done on 21 October (v0.11–v0.13), no change to the network

- `presentation.html` ("How it works"): the live network with every option
  coloured, the 4 numbers it sees, its top choices, a status line saying
  what is happening, the learning steps (reward, Bellman check, exploration,
  training curve), and the scoreboard against the bot.
- `learning.html` ("Watch it learn"): six training snapshots
  (`snapshots/game-0 … 2900.json`) playing side by side, plus the same six
  measured on 50 games: 21 → 61 → 217 → 376 → 2,000 (never lost) → 91.
- New look for the game (`style.css`), a shared top bar (`nav.css`), and a
  fix in `index.html` for the duplicate dropdown that stopped the game.
- `train_dqn.js --snapshots …` saves copies during training.

## Done on 22 October (v0.14): hold box

Hold in the game, the bot and the network (no retraining). 50 games, cap
2,000, classic → with hold:
- random pieces, no preview, lost: network 12 → 1, bot 9 → 0
- random pieces, next visible, lost: network + lookahead 2 → 0, bot 0 → 0
- rising garbage, pieces survived: network + lookahead 319 → 2,000 (never
  lost), bot 254 → 1,946 (4 lost)

Next: with hold, the 2,000-piece cap is too easy to separate the players.
Raise it (`--cap 10000`) before comparing further. Then decide between
stage B (retrain with the held piece as an input) and proposal 1 (5-step).

## Done on 22 October (v0.15)

- Faster: 2–2.5x on lookahead tests (no allocations in predict / 2-ply,
  `hardmode.js --workers`, results identical).
- Harder tests with hold: `garbage3`, `garbage4`, cap 10,000; tetrises/100.
  Baseline (50 games): random no-preview cap 10k lost bot 0 / net 3;
  garbage3 pieces bot 202 / net 114 / net+2ply 210; garbage4 bot 1,000 /
  net 215 / net+2ply 992.
- 5-step: NOT better over 16 runs (35.5 vs 33.6 lost). Dropped.
- Stage B (`train_dqn.js --hold`, 11 inputs): NOT better. Best of 8 runs:
  6 vs 3 lost (random, cap 10k), 97 vs 114 (garbage3). The hold-trained
  networks hold less and make almost no tetrises.
- `dqn-model.json` unchanged.

## ▶ NEXT: options

1. Stage B again, but trained where it matters: hold + garbage in training
   (`--hold --garbage 3`), since ordinary hold games almost never end and
   give little to learn from.
2. Deeper lookahead with hold (piece after next, averaged over 7).
3. Presentation polish: network diagram, hold in the demo, score label.
4. The CNN (raw board), as a separate learning project.

## Proposal 1 (closed: 5-step did not help)

Finish the 5-step check, then train more networks and keep the best.
Expected: 15–30% fewer lost games on random pieces (12 → about 8–10/50),
enough to beat the bot (9/50). Then try proposal 2 (a penalty for losing).

## Where we stopped on 20 October

**Task:** combine the repo's approach with our additions into one best recipe.
We keep the repo's core (4 features, 4→64→64→1, same reward) and add
**n-step targets** (`--nstep N` in `train_dqn.js`, off by default, so the
defaults still rebuild the shipped `dqn-model.json`).

Results so far: 8 seeds each (1–8), selection games (random pieces, no
preview, seeds 70000+), average games lost out of 50:

| recipe | games lost per seed | average |
|---|---|---|
| plain (the repo's) | 38, 36, 37, 32, 48, 44, 22, 14 | 34 |
| 3-step | 50, 50, 42, 23, 22, 34, 17, 33 | 34 |
| **5-step** | 36, 37, 18, 40, 24, 28, 19, 23 | **28** |
| garbage (16 seeds) | see README v0.9 | 30 |
| 5-step + garbage | 19, 43, 29, 49, 30, 42, 19, 50 | 35 (worse) |

5-step looks best, but 8 seeds fooled us before (learning-rate decay). A
confirmation run was interrupted: **5-step vs plain on seeds 9–16.**
No network has beaten the shipped one yet (14/50 on the selection games).

**To continue tomorrow:**

1. Rerun the confirmation (about 1 hour; both commands together):
   `node train_many.js --seeds 9,10,11,12,13,14,15,16 --out n5.json --nstep 5`
   `node train_many.js --seeds 9,10,11,12,13,14,15,16 --out plain.json`
2. If 5-step is still clearly better over 16 seeds, it becomes the
   recommended recipe: `node train_many.js --nstep 5`, with more seeds to
   find a network that beats the shipped one.
3. If not, we have reached the limit of the 4-feature approach. The next big
   step is the raw-board CNN.

## Later

- **Raw board as input (a CNN):** the biggest remaining step, but hours of
  training.
- **Speed up the 2-ply search** (about 30x slower than 1-ply).

## Rules we work by

- Decide on at least 50 games with fresh seeds (90000+). 10 games is not
  enough. When everyone reaches the 2,000-piece cap, raise it (`--cap 10000`).
- Train more than one seed before concluding anything.
- Do not change direction without agreeing first.
