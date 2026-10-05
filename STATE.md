# Project state — resume from here

Last updated: 2026-10-05

## CURRENT STATUS (October 2026)

The game runs in the browser. A convolutional value network
(`cnn-model.json`, 2,129 parameters, 2-ply) plays exactly as well as the
search bot: 0 losses in 20 games of 2,000 pieces and in 10 games of 5,000
pieces, same mean lines. See "Phase 2E". Imitation has reached its teacher's
ceiling; the next step is learning from its own play (RL). `PLAN.md` has the step-by-step version in Bosnian.

The section below is the September record of the repo mix-up and is kept for
history; it is resolved.

## WHERE WE STOPPED (September — resolved)

All three phases are written, run and measured — but **only in node**. The one
thing still not done is getting the refactored code running in the browser on
the real page, and there is a specific mess in the way.

**What happened.** The Phase 1+2 zip was extracted over the repo instead of
merged into it, and commit `74cdee8` ("implemented phase 2") deleted the
original game files: `index.html`, `style.css`, `main.js`, `images/`,
`TetrisAiFinal.rar`. That commit is already pushed. Nothing is lost — they live
in the parent commit.

**Recovery:**

```bash
git checkout 74cdee8^ -- index.html style.css images/
git add -A && git commit -m "Restore original game files" && git push
```

**Then, the actual integration.** Do NOT write a fresh `index.html`. We tried
that and the styling did not match, because `style.css` was written for the
original markup — the old code referenced wrappers like `#content` and `#users`
that a generic page does not have. Restore the original page and change exactly
one line in it: replace

```html
<script src="main.js"></script>
```

with

```html
<script type="module">
  import { startVisualGame } from './src/tetris-visual.js';
  const { weights } = await fetch('./weights.json').then((r) => r.json());
  startVisualGame({ weights, msPerRow: 45, lookahead: true, imagePath: 'images/' });
</script>
```

and move it to just before `</body>` if it sits in `<head>`.

This works without touching the CSS because `src/tetris-visual.js` builds the
same DOM the old `createBlocks()` did: `#gameContainer` holding
`div.row.row_0` … `div.row.row_26`, each with 16 `div.block` children. Same
class names, same structure.

Serve it with `npx serve .` and open the forwarded port — `type="module"` will
not load over `file://` or the VS Code preview.

If the styling still fights after that, the next step is to read `style.css`
and adapt, not to regenerate the page.

**After that is resolved:** delete `main.js`, and the project is fully
integrated.

**Agent switcher (done).** `startVisualGame` now takes a `chooser` and exposes
`setChooser()`, `setSpeed()` and an `onStats` callback, so the page can switch
between the search bot, the regression network and the ranking network live and
show lines / pieces / games / best. Switching wipes the board, because carrying
one agent's stack into another's run would make the comparison meaningless.
The search bot stays the default — the networks are there to be watched, not
because they play better.

## The goal

Build a machine-learning bot that plays the Tetris game in `tetrisML`. The
original `main.js` already contained a competent bot, but it was a hand-written
heuristic using four weights copied from a blog post. The project is to replace
progressively more of that human-authored judgement with things learned from
data.

## Where we are

**Phase 1 of 4 complete.** The evaluation weights are now learned by a genetic
algorithm rather than copied.

| phase | what it is | status |
|---|---|---|
| 0 | Headless simulator — game logic off the DOM, bugs fixed, tests | done |
| 1 | Genetic algorithm learns the 8 evaluation weights | done |
| 2 | Supervised learning — neural nets trained to imitate the bot | done |
| 3 | Reinforcement learning — DQN over afterstates, no teacher | not started |

Phase 2 is complete in the sense that matters: both architectures are built,
trained and measured, and the measurements explain themselves. Neither network
plays as well as the search bot, and the reasons are quantified below — that is
the finding, not a failure to finish.

## The ladder, conceptually

This is the thing to keep in mind, because it is what the phases mean:

- Original code: human picks the features, human picks the numbers. Not ML.
- **Now: human picks the features, machine learns the numbers.**
- Phase 2: machine learns a function mapping board to move, by imitation.
- Phase 3: machine learns everything from trial and error.

Each step removes more human judgement from the loop.

## Files

```
tetrisML/
├── src/
│   ├── tetris-core.js      board, pieces, seeded 7-bag, collision, drop, line clears
│   ├── tetris-ai.js        8 features, evaluation, 1-ply and 2-ply search
│   ├── tetris-headless.js  playGame(weights, seed) — the fast environment
│   ├── tetris-visual.js    the ONLY file that touches the DOM
│   ├── ga.js               the genetic algorithm
│   └── ga-worker.js        worker-thread body for parallel fitness evaluation
│   └── tetris-policy.js    Phase 2: input encoding, move labels, JS inference
├── bench.js                26 sanity tests + speed benchmark
├── train.js                runs the GA, writes weights.json + training-log.csv
├── compare.js              learned vs published weights on held-out seeds
├── collect.js              plays the bot, logs decisions -> data/moves.csv
├── collect_values.js       logs afterstates + expert scores -> data/values.csv
├── train_policy.py         behaviour cloning -> model.json
├── train_value.py          value regression -> value-model.json
├── play_policy.js          lets a network play; compares it with the expert
├── demo.html               standalone demo page
├── weights.json            the learned weights (committed on purpose)
├── index.html              yours — one script tag changed
├── style.css               yours — unchanged
├── images/                 yours — unchanged
└── main.js                 SUPERSEDED, delete once the browser version is confirmed
```

`tetris-core`, `tetris-ai`, `tetris-headless` and `ga` never reference
`document` or `window`. That separation is what makes training possible.

## Verified results

Sanity: **26 tests pass** (`node bench.js --test`).

Speed, published weights, 500-piece cap:

| | median lines | throughput |
|---|---|---|
| 1-ply (training) | 196 of a possible 200 | 11.7 games/s |
| 2-ply (playing) | 199 | 0.5 games/s |

Training run — 50 candidates, 8 generations, 2 cores, `--ceiling 2000`, 211s,
starting from random weights:

```
gen    best   median     mean    cap
  1   164.8      0.3     16.1    500
  3   197.0     99.8    108.1    500
  4   396.8    365.9    287.4   1000   <- cap raised
  5   797.5    794.1    711.4   2000   <- cap raised
  8   797.8    796.5    796.0   2000
```

Held out — 10 games, seeds 10000+, 10,000-piece cap, never trained on:

| | median | mean | worst game | topped out |
|---|---|---|---|---|
| published | 3994.5 | 3330.0 | 558 | 2 |
| learned | 3995.0 | **3914.5** | **3420** | 2 |

Interpretation: the medians are identical because both sets of weights reach
the cap on a typical game — the median has saturated and stopped measuring
anything. The gain is in **consistency**: 17.6% higher mean, and a worst game
of 3,420 lines against 558.

Learned weights (in `weights.json`):

| feature | learned | published |
|---|---|---|
| aggregateHeight | -0.72861 | -0.51007 |
| linesCleared | 0.24479 | 0.76067 |
| holes | -0.24099 | -0.35663 |
| bumpiness | 0.01141 | -0.18448 |
| maxHeight | -0.10832 | 0 |
| rowTransitions | -0.36670 | 0 |
| colTransitions | -0.37006 | 0 |
| wells | -0.26047 | 0 |

The GA disagreed with the published weights about what matters: far more weight
on total height, far less on lines cleared, and real weight on three features
the published vector ignores entirely. Surface smoothness measured as
transitions carries more signal than the original four features could express.

**Reproducibility is verified.** The same `--seed` produces bit-identical
weights regardless of `--workers`, because the RNG only runs on the main thread
and fitness evaluation is deterministic.

## Phase 2 results

Two architectures, same teacher (the Phase 1 bot), same course toolchain
(pandas, train/test split, sklearn).

**A. Behaviour cloning** — 240 board cells + two 7-wide piece one-hots in, one
of 40 (rotation, column) classes out. 60,000 decisions collected.

| model | test accuracy |
|---|---|
| most-frequent baseline | 9.1% |
| logistic regression | 33.9% |
| MLP 256×128 | **59.3%** (top-3: 84.7%) |

Then it played: **3.4 lines per game against the expert's 198.3 — 1.7%**, and
it topped out in all 20 games.

The diagnostic that explains it: agreement with the expert is 59.3% on boards
the *expert* reached, but only 36.1% on boards the *policy* reaches. That is
covariate shift. The expert keeps the stack flat, so the training set is
nothing but flat boards; one mistake puts the model somewhere it has never
been, where it is worse, so it errs again.

A DAgger round (`--policy`, expert labels every state but the model sometimes
drives, β=0.8) added 29,807 rows from exactly those recovery states. Retrained
on 89,807 rows: test accuracy went *down* to 57.1% — the new data is harder —
while play improved to 5.1 lines (2.6%). Real but nowhere near enough.

**B. Value network** — score an afterstate instead of naming a move. 240 cells
+ lines cleared in, the expert's own evaluation score out; at play time we
enumerate legal placements ourselves and argmax the network's judgement. This
is knowledge distillation, and it is the same shape Phase 3 needs.

| model | test R² |
|---|---|
| linear regression | 0.980 |
| MLP 128×64 | **0.998** |

Played: **24.2 lines per game, 12.2% of the expert** — seven times better than
behaviour cloning, still far from the teacher.

Why R² = 0.998 is not enough, measured over 20,813 afterstates:

| | |
|---|---|
| network RMSE | 0.969 score units |
| median gap, best vs second-best candidate | 1.049 score units |
| decisions where the gap is smaller than the RMSE | **46.4%** |

That is the whole story in three numbers. R² is measured against the target's
total spread (std ≈ 20.9), but the decision only ever needs to resolve
differences of about 1.0 between the top candidates. On nearly half of all
moves the network's noise exceeds the margin it is being asked to judge, so it
picks the wrong placement — and in a game where every move constrains the next,
those mistakes accumulate.

The lesson generalises well beyond Tetris: **when a model's output feeds an
argmax, the metric that matters is error relative to the decision margin, not
error relative to the target's variance.** A headline metric can look
outstanding while the model is useless for the decision it was built for.

**C. Pairwise ranking (RankNet)** — same network, same inputs, same inference
code; only the training objective changes. Instead of "what score does this
board deserve", the loss asks "of these two boards the bot was choosing
between, which is better": `L = log(1 + exp(-(f(better) - f(worse))))`.
Implemented by hand in numpy (`train_rank.py`) — sklearn has no ranking loss
and PyTorch is a 529 MB dependency for a two-layer network.

The result is the most interesting thing in the whole project.

| | all pairs | close pairs | picks bot's best | **lines per game** |
|---|---|---|---|---|
| regression (MSE) | 93.9% | 72.2% | 93.8% | 24.2 |
| ranking (pairwise) | 92.0% | 69.6% | 91.5% | **51.3** |

**Every offline metric prefers regression. The only metric that matters prefers
ranking, by more than 2×.** Ranking plays at 25.8% of the expert against
regression's 12.2%.

The explanation is in the agreement numbers. On boards the *expert* reaches —
the training distribution — regression wins (93.8% vs 91.5%). On boards the
*network itself* reaches during play, it is the other way round: ranking agrees
with the expert 50.6% of the time, regression only 36.5%.

So the ranking objective did not produce a more accurate model. It produced one
that **degrades more gracefully off-distribution**. Freed from having to
reproduce absolute magnitudes, it learned relative structure that still holds
on boards it was never trained on — and in a game where every move determines
what you see next, robustness outside the training distribution beats accuracy
inside it.

### What this settles, and what it does not

It settles the diagnostic question: the training objective genuinely mattered,
worth a 2× improvement, so this was not purely a capacity problem.

It does not close the gap — 25.8% is still a long way from the expert. Two
suspects remain, and they are testable in this order:

**Covariate shift, again.** The value dataset is collected while the *expert*
drives, so it is all tidy boards. The DAgger fix that helped behaviour cloning
applies here unchanged: collect value data while the *network* drives, with the
expert still supplying every label. This is the obvious next experiment and it
reuses code that already exists.

**Architecture.** A fully-connected network sees 240 unrelated numbers; it has
no idea the board is a 2D grid. A small convolutional network sees
neighbourhoods, which is how holes, overhangs and wells are actually defined.

## Phase 2D — 2-ply search with the network, and DAgger (October 2026)

Goal confirmed by the user: a neural network that plays well, not the
search bot.

**2-ply.** `chooseValueMove(board, piece, net, { next, lookahead, beam })` now
looks one piece ahead: rank the current piece's moves by 1-ply score, expand
the top `beam` (default 5), judge each by the best board the next piece can
reach. ~6 ms per decision in node.

The first version passed the network only the SECOND move's line clears.
That played at 33.4 lines — a third of 1-ply. Screening variants showed the
cause: with first-move clears uncredited, the search postponed clearing and
the stack grew. Crediting both moves (`min(4, lines1 + lines2)`, exactly what
the Phase 1 bot does) took it to 184.9 lines, 93.2% of the expert, 2 deaths
in 20. Agreement on its own boards rose from 50.6% to 74.5%.

**DAgger.** `collect_values.js --policy <model> --lookahead --beta 0.5` lets
the network drive half the pieces (with 2-ply) while the expert labels every
candidate board. `merge_csv.js` concatenates datasets and shifts
`decision_id`s so pairs never cross files. One round: 141,702 new rows, merged
to 213,702, retrained as `rank-dagger.json`.

At a 2,000-piece cap (10 games, seeds 20000+):

| | median | mean | % expert | died | own-board agreement |
|---|---|---|---|---|---|
| rank, 2-ply | 679.5 | 527.8 | 66.1% | 6/10 | 75.5% |
| rank + DAgger, 2-ply | 797.5 | 676.7 | 84.8% | 2/10 | 80.9% |
| expert | 798 | 798.2 | 100% | 0/10 | — |

The DAgger model at 1-ply barely moved (51.3 → 54.6, noise): the data was
collected under the 2-ply policy, so it covers the boards the 2-ply player
reaches. DAgger fixes the distribution of the policy you collected with.

10 games is a small sample; the median matching the expert is the more
reliable signal than the mean or the death count.

**DAgger round 2.** `rank-dagger.json` drove 70% of pieces (beta 0.3, 2-ply,
cap 600, seeds 60000+): 208,302 rows, merged to 422,004 / 70,334 decisions,
retrained 25,000 steps as `rank-dagger2.json`. Re-measured both rounds on 20
games at a 2,000-piece cap:

| | median | mean | worst | % expert | died | own-board agreement |
|---|---|---|---|---|---|---|
| round 1 | 798 | 703.6 | 23 | 88.1% | 4/20 | 80.9% |
| round 2 | 798 | 762.0 | 339 | 95.4% | 3/20 | 83.0% |
| expert | 798.5 | 798.4 | 797 | 100% | 0/20 | — |

Gains are shrinking (round 1: +22 points, round 2: +7). The next lever is
architecture: a small convolutional network.

## Phase 2E — convolutional value network (October 2026)

`train_cnn.py`, numpy only (download.pytorch.org is blocked from the build
sandbox and the PyPI torch wheel needs multi-GB CUDA libs; numpy also means the
user's Codespace needs no new dependency).

Architecture: board padded to 26x12 with walls/floor = 1 and sky = 0, plus a
constant height channel `(25 - paddedRow) / 24`; conv 3x3 2->16 ReLU;
depthwise 3x3 + pointwise 1x1 16->32 ReLU; average pool over all 240
positions; [32 pooled, lines] -> 32 ReLU -> 1. 2,129 parameters, no
per-position weights. Pairwise RankNet loss, Adam, cosine LR decay, 10,000
steps of 256 pairs on `data/combined2.csv` (~32 min on 2 cores).

`--gradcheck` compares all analytic gradients with central differences in
float64: worst relative error 6.1e-8.

The test split reproduces train_rank.py's exactly (fresh `default_rng(42)`,
same shuffle), so the MLP comparison is on decisions neither model trained on.
An earlier draft used a different split and graded the MLP partly on its own
training data, flattering it by ~2 points.

| | params | all pairs | close pairs | top-1 |
|---|---|---|---|---|
| MLP rank-dagger2 | 94,977 | 95.6% | 78.9% | 96.6% |
| CNN | 2,129 | 98.9% | 93.7% | 99.1% |

Play, 2-ply beam 5, seeds 20000+:

| | cap | games | mean | worst | died | own-board agreement |
|---|---|---|---|---|---|---|
| MLP rank-dagger2 | 2000 | 20 | 762.0 | 339 | 3 | 83.0% |
| CNN | 2000 | 20 | 798.3 | 797 | 0 | 91.2% |
| expert | 2000 | 20 | 798.4 | 797 | 0 | — |
| CNN | 5000 | 10 | 1998.3 | 1997 | 0 | — |
| expert | 5000 | 10 | 1998.3 | 1998 | 0 | — |

JS inference (`loadCnnNet` in tetris-policy.js): conv1's constant parts (bias,
height channel, walls, floor) are precomputed into a base map; per board only
filled cells are scattered in. Rows more than two above the stack are
identical to an empty board (5x5 receptive field), so their pooled
contribution is precomputed as a running total by row and skipped. ~87 us per
evaluation, ~10-17 ms per 2-ply decision. `cnn-model.json` carries five boards
with Python scores; the JS loader verifies them and throws on mismatch.

Both caps are now saturated for both players. Further comparisons need longer
games or harder conditions.

## Decisions already made (don't relitigate)

Kept the original 16×27 board with the 3-cell frame. The frame is why collision
checks need no bounds tests, and keeping it made the refactor a much smaller
delta.

Board is a flat `Uint8Array`, cell `(r, c)` at `r * WIDTH + c`. Copying the
board is the hottest operation in the search; `slice()` beats nested loops by a
wide margin.

Piece randomiser is a standard 7-bag, not the original five-array sack. The old
one could repeat pieces heavily, which adds fitness noise that makes two weight
vectors hard to tell apart.

The piece takes its chosen rotation and column immediately, then falls. The
original slid one column per tick and switched to `gameSpeed = 10` past row 4,
so a piece could land somewhere the AI never evaluated. The move you score has
to be the move you made.

Eight features, not four. The four extras default to weight 0, so the published
vector reproduces the original behaviour exactly while leaving the GA room.

Training uses 1-ply, playing uses 2-ply. Lookahead costs ~24x and barely
changes the *ranking* of weight vectors.

Fitness is "lines cleared within N pieces", not "lines until death", because
with good weights this bot does not lose and training would never terminate.

Train on seeds 1–8, evaluate on seeds 10000+. Same discipline as a train/test
split.

## Bugs fixed in Phase 0 (for the record)

The AI's simulation was passing the **live** `gameBoard` to `breakeRowsForAi`,
which mutates — so every candidate placement was deleting rows from the real
game. There is now a regression test for this.

`aiCurrentTetrominoPosition.x` was never reset between candidate columns, so
most placements were silently skipped.

Bumpiness compared filled *cell counts* per column instead of column *heights*.

`checkHorizontalWallCollision` never added `i`/`j`, so its nested loops checked
one cell sixteen times.

`gameSpeed = 10` past row 4 meant pieces could land where the AI never looked.

Plus: dead `tetromi` object with wrong shapes, `oneTetrominoSack[34]` magic
numbers, a duplicated re-clearing block in `breakeRows`, and `querySelectorAll`
for all 27 rows on every frame.

## Open items

**The browser version still does not run.** See "Where we stopped" at the top —
this is the live blocker. The code passes a fake-DOM smoke test in node (builds
432 blocks, plays, clears lines, updates the score element), so the renderer
logic is sound; what is unresolved is the page and CSS integration.

`main.js` is superseded by `src/`. Delete it once the browser version works.

Datasets (`data/*.csv`, ~123 MB) are gitignored and not in the repo. Regenerate
with `collect.js` and `collect_values.js` when needed — the commands are under
"How to reproduce Phase 2".

The demo training run used `--ceiling 2000` to finish in three minutes. The
default ceiling is 20,000, which keeps selection pressure on far longer and
should produce better weights, at the cost of several minutes per late
generation. Worth running properly.

The action space excludes tuck and spin moves — placements are "rotate, slide,
drop" only. This is standard for Tetris AI and probably fine, but it is a
ceiling on how good any agent here can get.

## How to resume

```bash
cd tetrisML
node bench.js --test      # expect: 26 passed, 0 failed
node bench.js             # expect: ~196 median lines, ~11 games/s
node compare.js --games 20 --cap 10000
```

To start the learning over from scratch:

```bash
rm -f weights.json training-log.csv
node train.js --workers 4          # full default run, 40 generations
```

To watch it play with learned weights, in `index.html`:

```html
<script type="module">
  import { startVisualGame } from './src/tetris-visual.js';
  const { weights } = await fetch('./weights.json').then((r) => r.json());
  startVisualGame({ weights, msPerRow: 45, lookahead: true });
</script>
```

`type="module"` will not load over `file://`. In a Codespace run `npx serve .`
and open the forwarded port.

## How to reproduce Phase 2

```bash
node collect.js --games 120 --cap 500          # ~150s -> data/moves.csv (31MB)
python3 train_policy.py --data data/moves.csv  # ~60s  -> model.json
node play_policy.js --games 20                 # the reality check

node collect_values.js --games 40 --cap 300    # -> data/values.csv
python3 train_value.py --data data/values.csv  # -> value-model.json
node play_policy.js --value value-model.json
```

## Next step

Phase 3, reinforcement learning. The value network from Phase 2B is already the
right architecture — a DQN over afterstates is that same network with a
different training signal. Instead of regressing onto the expert's opinion, the
target becomes the reward plus the network's own estimate of the next state,
and the teacher disappears entirely.

Before that, the cheap win is a pairwise ranking loss on the Phase 2B data. It
targets the quantity the argmax actually consumes, and it will show whether the
gap is a training-objective problem or a capacity problem — worth knowing
before investing months in RL.
