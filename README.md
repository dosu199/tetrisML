# Tetris ML

A neural network that learns to play Tetris by playing it. Nobody shows it
good moves; it only gets the game's reward. There is also a classic search bot
to measure it against.

The approach follows
[vietnh1009/Tetris-deep-Q-learning-pytorch](https://github.com/vietnh1009/Tetris-deep-Q-learning-pytorch),
rewritten in plain JavaScript. It needs no PyTorch or Python, and the same
code trains in node and plays in the browser.

## How the network plays

For each new piece:

1. Try every rotation in every column.
2. For each resulting board, compute 4 numbers: **lines cleared, holes,
   bumpiness, total height**.
3. The network (4 → 64 → 64 → 1, about 4,500 weights) turns those 4 numbers
   into one score: how good this board is for the rest of the game.
4. Play the move with the highest score.

**With lookahead (v0.7):** when the next piece is visible, it also tries every
placement of the next piece for each candidate move, and scores the pair with
the same formula it learned from: `reward now + 0.99 × (reward next +
0.99 × score of the final board)`. This uses the same network with no
retraining; only the way it picks moves changes. It is about 30× slower,
which is still fast enough to watch.

## How it learns (deep Q-learning)

- **Reward** per piece: `1 + 10 × lines²`. Surviving earns a little and clearing
  lines earns a lot. A tetris (4 lines) is worth 16× a single.
- **Target** (the Bellman equation):
  `score(previous board) ≈ reward + 0.99 × score(new board)`, or just the
  reward if the game ended. Repeat this over thousands of moves and value flows
  backwards from line clears and deaths to the moves that caused them.
- **Exploration**: at first it plays at random (epsilon 1). Epsilon falls to
  0.001 over 2,000 games, so by the end it almost always picks its best move.
- **Replay memory**: the last 30,000 moves. After every game it takes one
  training step on a random batch of 512 of them.
- **Best checkpoint**: DQN does not improve steadily. It climbs, then often
  collapses. Every 100 games, `train_dqn.js` plays 5 test games with no
  exploration and saves the network only when it beats the best so far. In our
  run the final network survived 144 pieces on average while the saved best
  survived about 1,900. So always use the saved best network, never the final
  one.

## Results

The shipped `dqn-model.json` (v0.8: random pieces, seed 8). 50 games per row on
seeds 90000+ (never seen in training). "1-ply" means it decides on the current
piece only. "2-ply" means it also looks at the next piece.

| pieces | preview | cap | bot (search) | DQN 1-ply | DQN 2-ply |
|---|---|---|---|---|---|
| 7-bag | no | 10,000 | 8/50 lost | **0/50 lost** | (needs preview) |
| fully random | no | 2,000 | **9/50 lost** | 12/50 lost | (needs preview) |
| fully random | yes | 2,000 | **0/50 lost** | 12/50 lost | 2/50 lost |
| 7-bag + garbage row every 5 pieces | yes | — | 254 pieces | 136 pieces | **319 pieces** |

(In the garbage mode everyone eventually tops out, so the number is mean
pieces survived; higher is better.)

- **Without preview:** the network beats the bot on normal pieces. On fully
  random pieces the bot is still a little better (9 vs 12 lost).
- **With preview:** the network with lookahead nearly matches the bot on
  random pieces (2 vs 0 lost) and beats it clearly in the hardest mode, 319
  vs 254 pieces with rising garbage. It never trained on garbage at all.

### Deep lookahead (v0.18)

The network can now look **two pieces ahead** when it plays (`{ deep: true }`
in `chooseDqnAction`, `--deep` in `hardmode.js`). For each spot for the
current piece, it tries the spots for the next piece and then averages over
all 7 possible pieces after that (the "expectimax" idea). Checking every
combination would be about 200× slower, so it uses **beam search**: every
option is first scored the quick 2-ply way, and only the best 6 first moves
(and 4 follow-ups each) get the deeper look. With hold, it plays about 9
pieces per second on one core; the 2-ply version plays about 66.

50 games each, seeds 90000+, all with hold:

| test | bot | network + lookahead (2-ply) | network + deep lookahead (3-ply) |
|---|---|---|---|
| garbage every 3 pieces, pieces survived | 202 | 210 | **403** |
| garbage every 4 pieces (cap 2,000), pieces survived | 1,000 (47 lost) | 992 (47 lost) | **2,000 (0 lost)** |

Looking one piece further almost doubles survival in the hardest test, and
the network now beats the bot clearly in every garbage test. The network
itself is unchanged; only the search around it is deeper. Deep lookahead is
the default on the game page and the strongest opponent in Play vs Bot.

### Faster tests, harder tests, and two experiments that did not help (v0.15)

- **Faster.** `predict` and the 2-ply search no longer allocate memory on every
  call, and `hardmode.js` runs games on all CPU cores (`--workers N`, results
  are identical). Lookahead tests are 2–2.5× faster overall.
- **Harder tests.** With hold, almost every game reached the 2,000-piece cap.
  New modes `garbage4` and `garbage3` (a garbage row every 4 / 3 pieces), plus
  a tetrises-per-100-pieces column. Baseline with hold, 50 games:

  | test | bot | network | network + lookahead |
  |---|---|---|---|
  | random pieces, no preview, cap 10,000, games lost | **0** | 3 | — |
  | garbage every 3 pieces, pieces survived | 202 | 114 | **210** |
  | garbage every 4 pieces, pieces survived | **1,000** | 215 | 992 |

- **5-step learning: no.** Over 16 training runs, 5-step lost 35.5 of 50 games
  on average and normal 1-step lost 33.6. Its lead after 8 runs was luck.
- **Training with hold (stage B): no, so far.** `train_dqn.js --hold` gives
  the network 7 extra inputs (which piece is in the hold box) and lets it hold
  during training. Of 8 runs, the best was worse than the current network
  using hold without training: 6 vs 3 games lost on random pieces, 97 vs 114
  pieces with garbage every 3. The hold-trained networks hold less and make
  almost no tetrises (0.04 vs 1.9 per 100 pieces); some barely hold at all.
  The option stays in `train_dqn.js` for future experiments.
- **Training with hold + garbage: closer, still no.** The suspicion was that
  with hold, training games almost never end, so the network rarely sees a
  loss. `--hold --garbage 3` puts a garbage row every 3 pieces into half of
  the training games. Of 8 runs, two tied the current network on the
  selection games (98 and 96 vs 97 pieces). On the final test, both play with
  hold, 50 games, up to 10,000 pieces:

  | test | current | hold + garbage, best | second |
  |---|---|---|---|
  | random pieces, no preview, games lost | **3** | 9 | 8 |
  | garbage every 3 pieces, pieces survived | **114** | 106 | 96 |
  | garbage every 4 pieces, pieces survived | **215** | 203 | 161 |

  Garbage closed most of the gap (106 vs 97 last time), but the current
  network still wins every test. The hold-trained networks hold less (13–22%
  of pieces vs 24–30%) and make fewer tetrises.

`dqn-model.json` is unchanged. The best setup so far is the network trained
**without** hold, using hold when it plays.

### Hold (v0.14)

The game now has a **hold box**, as in modern Tetris. Instead of placing the
current piece, a player may put it in the box and take the held piece (or the
next one, if the box is empty), once per piece. The bot and the network both
use it, so comparisons stay fair. The network was **not retrained**: it scores
boards, and hold simply gives it more boards to choose from. The rule they
share is `decideWithHold` in `src/tetris-core.js`. It compares "place this
piece" with "hold, then place what you get", averaging over all 7 pieces when
that piece is still unknown.

50 games per row, seeds 90000+, games stopped at 2,000 pieces. Classic rules
first, then with hold:

| test | bot | network | network + lookahead |
|---|---|---|---|
| random pieces, no preview, games lost | 9 → **0** | 12 → **1** | — |
| random pieces, next visible, games lost | 0 → 0 | 12 → 2 | 2 → **0** |
| rising garbage, average pieces survived | 254 → 1,946 (4 lost) | 136 → 486 | 319 → **2,000 (0 lost)** |

Hold removes almost all losses on random pieces: 92% fewer for the network,
100% fewer for the bot. With rising garbage, the network with lookahead now
never loses in 2,000 pieces; it used to survive about 320. Both players hold
about one piece in four. Turn it on in a test with `node hardmode.js --hold`.
The game page has hold switched on (`startVisualGame({ hold: true })`).

### Making training stable (v0.8)

Our network came from a lucky run, so we tried to make every run good. Each
setting was trained on 8 seeds with `train_many.js` and scored on the same 50
selection games (random pieces, no preview, seeds 70000+). The numbers are
games lost out of 50; lower is better.

| setting | seeds 1–8, games lost | average |
|---|---|---|
| plain DQN (default) | 38, 36, 37, 32, 48, 44, 22, **14** | 34 |
| learning rate 1e-3 → 1e-4 after exploration (`--lr-end 1e-4`) | 20, 42, 31, 23, 42, 45, 22, 48 | 34 |
| 4 training steps per game (`--updates 4`) | 50, 50, 43, 29, 50, 50, 45, 45 | 45 |
| weight averaging (`--avg 0.99`), seeds 1–4 only | 50, 50, 50, 34 | 46 |
| learning-rate decay from the start, seeds 1–4 only | 50, 50, 50, 50 | 50 |

None of them made a typical run better.

- **Learning-rate decay** looked like a clear win after 4 seeds (average 29
  vs 36). With 4 more seeds it was exactly even. Four runs were not enough to
  tell.
- **Decay from the start** stopped learning entirely. The network learns most
  near the end of exploration, and by then the rate was already too low.
- **More steps per game and weight averaging** both made things worse. Extra
  steps let the network chase its own moving targets faster. Averaging mixes
  networks that are too different from each other.

So for this setup, the honest answer to "how do we make every run good?" is
that we can't yet. The spread between seeds (14 to 48 lost) is part of how DQN
behaves here. What works is to **train many seeds and keep the best on a
separate test**, which is what `train_many.js` does. Seed 8 came out of these
runs. On the final test it beats the old network without preview (12 vs 16
lost on random pieces, 0 vs 1 on normal pieces up to 10,000). With lookahead
it is level (2 vs 0 lost on random pieces, 319 vs 321 pieces with garbage).
Those differences are within luck, so overall it is a modest step forward.

All three options stay in `train_dqn.js` (off by default), so you can
experiment with them.

### Training with garbage rows (v0.9)

`--garbage 5` adds a rising garbage row every 5 pieces in half of the training
games (`--garbage-share` changes the half). The check inside training then also
plays garbage games. We trained 16 seeds with it. Same selection games as in
v0.8: random pieces, no preview, seeds 70000+, games lost out of 50.

| training | seeds | games lost | average |
|---|---|---|---|
| plain (v0.8 table) | 1–8 | 38, 36, 37, 32, 48, 44, 22, 14 | 34 |
| with garbage | 1–8 | 31, 37, 19, 30, 16, 27, 14, 40 | 27 |
| with garbage | 9–16 | 40, 32, 29, 26, 24, 39, 37, 31 | 32 |

Training with garbage makes a typical run a bit better, 30 lost on average
over 16 seeds vs 34. Its worst runs were also less bad (40 vs 48). It is the
same lesson as v0.6: practising on harder games helps on easier ones. The gain
is small, though, and the seed spread is still large.

The best garbage-trained network (seed 7) did **not** beat the shipped one on
the final games:

| final test, 50 games | shipped (seed 8) | garbage seed 7 |
|---|---|---|
| random pieces, no preview, lost | **12/50** | 15/50 |
| 7-bag, cap 10,000, lost | 0/50 | 0/50 |
| garbage, with lookahead, mean pieces | 319 | **349** |

It is better at garbage and worse at random pieces, and both differences are
within luck. So `dqn-model.json` is unchanged in v0.9. Garbage survival itself
barely moved without lookahead (about 130–150 pieces for every network).
Digging out of garbage needs planning that 4 board numbers cannot describe.

### Picking the best of several runs (v0.7)

`train_many.js` trains several seeds and picks the winner on a separate set of
games (70000+), so the final test (90000+) stays fair. It is the train /
validation / test split.

It never makes your model worse: the current `dqn-model.json` takes part in
the selection test and is replaced only if a new run beats it.

We also tried a stricter check inside training: 20 games capped at 3,000
pieces instead of 5 capped at 1,000. It made **no difference**: on seeds 1–4
it picked exactly the same checkpoints as the default check. Those four
networks lost 32–38 of 50 selection games, while our network lost 22. So a
typical training run is weaker than that network; seed 124 was lucky. Making a typical
run as good is a job for more stable training, not for a stricter check.

### Training on random pieces (v0.6)

`--pieces bag|random|mix` picks what the network trains on. The check inside
training then uses the same kind of pieces. Same 50-game test as above, random
pieces:

| trained on | seed | topped out (random) | topped out (7-bag, 10k cap) |
|---|---|---|---|
| 7-bag (v0.5 model) | 125 | 25/50 | 0/50 |
| random | 123 | 30/50 | |
| **random** | **124** | **16/50** | **1/50** |
| mix | 123 | 19/50 | 1/50 |
| mix | 124 | 33/50 | |
| mix | 125 | 33/50 | |

Training on harder pieces made the best network better on random pieces
(25 → 16) without losing its strength on normal ones. A network trained on
hard cases handles easy ones too; the reverse is not true. But the gap between
seeds (16 vs 33) is still bigger than the gain itself. The weak point is now
how we pick the checkpoint, not what we train on.

### What we learned getting here (v0.5)

- **One training run proves little.** We trained the same code with three
  seeds. In the 10,000-piece test, the three networks lost 20/50, 12/50 and
  0/50 games. The random seed mattered more than any setting we tried. So
  train several seeds and keep the best on a 50-game test.
- **More inputs did not help (yet).** Giving the network all 8 of the bot's
  features (`--features full`) made it no better, and two of three runs were
  much worse. Possible reasons: about 3,000 training steps are too few for
  the extra inputs, and the 5-game check inside training is too coarse to
  pick the best checkpoint.
- **Feature scaling is required for 8 inputs.** Without it, the 8-feature
  network did not learn at all; it stayed at about 50 pieces. The inputs
  differ by up to 60× (lines 0–4, total height up to 240), so a few large
  ones swamp the rest. `FEATURE_SCALE` in `tetris-dqn.js` divides each input
  by roughly its largest value.

## Files

```
src/tetris-core.js       board, pieces, 7-bag, collision, line clears, playEpisode()
src/tetris-dqn.js        the network: features, MLP with backprop + Adam, chooseDqnMove()
src/tetris-ai.js         8 features + search bot (the opponent)
src/tetris-visual.js     the only file that touches the DOM
src/tetris-headless.js   fast games for the genetic algorithm
src/ga.js, ga-worker.js  genetic algorithm for the bot's weights
train_dqn.js             trains the network        -> dqn-model.json
train_many.js            trains several seeds, keeps the best on a validation set
hardmode.js              bot vs network, 5 rule modes (--lookahead adds 2-ply)
test_dqn.js              gradient check + smoke tests for the network
versus.html/.css/.js     play against the network, side by side
versus-ai-worker.js      the opponent's thinking, in a background thread
presentation.html/.css/.js   visual explanation page, network playing live
learning.html/.css/.js   six training snapshots playing side by side
snapshots/               the network saved at 6 moments of training
nav.css                  the top bar shared by all pages
bench.js                 26 tests for the game core
train.js / compare.js    genetic algorithm for the bot -> weights.json
```

## Commands

```bash
npm test                 # 26 core tests + the DQN checks
node train_dqn.js        # train a new network, about 5 minutes -> dqn-model.json
node hardmode.js         # bot vs network in all 5 modes, 10 games each
node hardmode.js --model dqn-model.json --games 50 --seed 90000 --mode no-preview
node hardmode.js --model dqn-model.json --lookahead --games 50 --seed 90000 --mode garbage5
node train_many.js --seeds 1,2,3,4   # 4 networks, 2 at a time, about 15 min
npx serve .              # then open index.html in the browser
```

Training options (all have defaults): `--epochs 3000 --decay 2000 --batch 512
--lr 1e-3 --gamma 0.99 --memory 30000 --seed 8 --pieces random --features basic --out dqn-model.json`.
Experimental, off by default: `--lr-end`, `--avg`, `--updates` (see v0.8 above),
`--garbage N` and `--garbage-share` (see v0.9), and `--nstep N`: learn from
the next N moves' rewards instead of one (in testing, see PLAN.md).
`--features full` uses all 8 bot features (scaled automatically); `--scale`
turns scaling on for the 4 basic ones. Training is deterministic: the
defaults reproduce the shipped `dqn-model.json` exactly.

Training writes over `dqn-model.json`, so use `--out` to keep the old one
(for example `--out dqn-try2.json`). You can then compare the two networks in
one run:
`node hardmode.js --model dqn-model.json,dqn-try2.json --games 50 --seed 90000`.

## Play vs Bot page

`versus.html` lets you play against the network (or the search bot) on two
boards side by side, with exactly the same pieces. Both of you see the next
piece and can hold.

- **Battle:** clearing 2 / 3 / 4 lines at once sends 1 / 2 / 4 garbage rows to
  the other board, and your own clears cancel garbage coming at you. Whoever
  tops out first loses.
- **Race:** first to 40 lines wins; topping out loses.
- **Opponent:** network + deep lookahead (strongest), network + lookahead,
  network, or search bot, at a speed you choose (from **Super easy**, 1 piece
  every 4 seconds with about 30% of its pieces dropped in a random spot, up to
  5 per second). Opponent and speed can be changed at any time, also during a
  match; a new mode starts when you press Restart. The opponent thinks in a background thread
  (`versus-ai-worker.js`), so its slower searches never make your own piece
  stutter. On a slow computer the deep lookahead may not keep up with the
  fastest setting.
- **Keys:** ← → move, ↑ or X rotate, Z rotate the other way, ↓ soft drop,
  Space hard drop, C or Shift hold, P or Esc pause. On phones and tablets,
  touch buttons appear under the boards.

## Presentation page

`presentation.html` is a visual explanation of the project, with the real
network playing live:

1. **Watch it think:** every option for the current piece is drawn on the
   board, coloured from red (bad) to green (good), with the 4 numbers the
   network sees and its top choices. Pause and step through one move at a
   time. With the Hold box on (as in the game), the options holding would
   give are drawn dashed, and the status line says when it chooses to hold.
2. **How it learned:** what it sees, a diagram of the network
   (4 → 64 → 64 → 1), the reward, the Bellman rule (checked live against the
   game), exploration (a mini-game with a slider for random moves), and the
   real training curve.
3. **How good is it:** the 50-game results against the bot, with classic
   rules and with the Hold box.

Open it like the game: `npx serve .`, then click **How it works** in the top
bar (or go to `/presentation.html`). All pages share the bar from `nav.css`.

## Watch it learn page

`learning.html` replays the training: six copies of the network, saved after
0, 1,000, 1,700, 2,200, 2,800 and 2,900 training games, play side by side with
the same pieces. Below them are the same six measured on 50 test games
(average pieces survived: 21, 61, 217, 376, 2,000 = never lost, and 91 after
training knocked it over).

The copies are in `snapshots/`, made with
`node train_dqn.js --snapshots 0,1000,1700,2200,2800,2900`. Training is
repeatable, so `snapshots/game-2800.json` is exactly `dqn-model.json`. It only
reads `src/` and `dqn-model.json`, and has its own `presentation.css`. The
game page (`index.html`, `style.css`) is not touched.

## Play page (you, or the network)

`index.html` is the game itself. Since v0.19 you can play it yourself:

- **Turn on ML** switch: off = you play, on = the player chosen under
  **Who plays** (network + deep lookahead, network + lookahead, network, or
  the search bot) plays and you watch. Picking a player turns ML on. The page
  remembers your choice.
- **Keys:** ← → move, ↑ or X rotate, Z rotate back, ↓ soft drop, Space hard
  drop, C or Shift hold, P or Esc pause, Enter (or Space) after game over to
  play again. On phones and tablets, touch buttons appear at the bottom.
- Gravity speeds up every 10 lines, as in the Play vs Bot page.

Under the hood this is `startVisualGame({ human: true, onState })` in
`src/tetris-visual.js`, with `setHuman(on)`, `press(action)`, `softDrop(on)`
and `pause(on)` for the page. `type="module"` scripts do not load over
`file://`; in a Codespace, run `npx serve .` and open the forwarded port.

## History

The project began with a genetic algorithm that tuned the bot's 8 weights.
That work is still used: it produced `weights.json`. Next came imitation
learning, where networks copied the bot (MLP, ranking net, CNN, DAgger). That
works, but it can never beat its teacher, and it needed a large Python
pipeline. Version 0.4 replaced it with this much smaller DQN, which learns from
the reward alone. Version 0.5 added feature sets and scaling, and picked the
best of several training seeds. Version 0.6 trains on random pieces.
Version 0.7 adds the 2-ply lookahead and `train_many.js`. Version 0.8
tests three ways to stabilise training (none helped) and ships seed 8.
Version 0.14 adds the hold box. Version 0.15 makes tests faster and harder,
and tests 5-step learning and training with hold (neither helped). Version
0.17 adds Play vs Bot, 0.18 the deep (3-ply) lookahead, and 0.19 lets you
play the Play page yourself.
Version 0.9 adds training with garbage rows. The old code is in git history if you ever want it.
