# Plan — Tetris ML

Last updated: 27 October 2026 (v0.31).

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

- Stage B with hold + garbage (`--hold --garbage 3`, 8 runs): closer but
  NOT better. Final test, both with hold, cap 10,000: random lost 3 (current)
  vs 9 / 8; garbage3 114 vs 106 / 96; garbage4 215 vs 203 / 161.
  Conclusion: best setup = network trained WITHOUT hold, using hold to play.

## Done on 22 October (v0.17): Play vs Bot

`versus.html`: you vs the network / bot on the same pieces, Battle (garbage)
and Race (40 lines) modes, adjustable opponent speed, keyboard and touch
controls, hold and next for both. New top-bar item on every page.

## Done on 23 October (v0.18): deep (3-ply) lookahead

Beam search: next piece known, the piece after it averaged over all 7; only
the top 6 first moves (4 follow-ups each) get the deep look. 50 games, hold:
- garbage every 3: deep 403 vs 2-ply 210 vs bot 202
- garbage every 4 (cap 2,000): deep 2,000 (0 lost) vs 2-ply 992 vs bot 1,000
Default on the game page; strongest Play vs Bot opponent (runs in a Web
Worker so your piece never stutters). Network unchanged.

## Done on 23 October (v0.19): pages

- Play page: you can play it yourself (keyboard and touch); a "Turn on ML"
  switch hands the game to the player chosen under "Who plays".
- Play vs Bot: Super easy speed (1 piece every 4 s, ~30% random drops), new
  intro screen, opponent and speed can be changed at any time.
- Network unchanged.

## Done on 23 October (v0.20): one look for all pages

- `theme.css`, loaded by every page: one dark palette, one background, one
  Start button, the intro screen with falling blocks.
- Play page always opens on "Ready to play?" (you play); ML only when switched on.
- Play vs Bot: the opponent's board gets its own intro screen; calmer colours.

## Done on 23 October (v0.21)

- Falling tetrominoes (all 7 kinds) as a faint background on every page, and
  on the intro screens (`tetro-bg.js`). Off for reduced-motion users.
- v0.22 fix: the script now carries its own styles (a cached old
  `theme.css` showed the pieces as an unstyled row), and the shared files
  are linked as `theme.css?v=22` / `tetro-bg.js?v=22` so browsers reload them.
- v0.23: pieces only in the page background (none on the intro screens),
  more visible (opacity 0.16) and denser (about one per 45px of width).
- v0.24: the ML switch hands over the same game (no reset) both ways.
- v0.25: How it works / Watch it learn: the hero and shaded sections are
  see-through, so the pieces fall from the top there too.
- v0.26: intro screen class renamed `.intro-screen` (it clashed with the
  How it works paragraphs); Play panel: stat cards and a who-plays badge
  instead of the text line and key list; every file linked with `?v=26`.
- v0.27: Play vs Bot, "You" sits above your board on wide screens.

## Done on 27 October (v0.28): folder structure

Pages stay in the root; `assets/{css,js,images}`, `src/`, `models/`
(+ `snapshots/`), `scripts/` (+ `legacy-ga/`), `tests/`. Renamed
`presentation.*` to `how-it-works.*` (`presentation.css` is now
`assets/css/base.css`, since three pages use it); `presentation.html` forwards
old links. Underscore names became dashes (`train-dqn.js`, `test-dqn.js`).
Removed: `test.js` (old copy of hardmode), `tetris-ml-dqn.zip`, `runs/`
(now ignored), 3 unused images. Older entries below use the old file names.

v0.29 (showcase polish): README top with live link and preview image, tab
icon (`assets/meta/favicon.svg` + PNGs), link preview card on every page
(`assets/meta/preview.png`, 1200x630), "Made by Suad" footer on every page.

v0.30: removed mentions of the original repo. v0.31: Play page hint (H),
Rematch button / Enter on Play vs Bot, Lighthouse fixes: accessibility 100 on
every page (button contrast, heading order, logo link name), no layout shift
on load (board sizes reserved in CSS). Left as is: compression and caching
(GitHub Pages handles them), minifying (needs a build step).

## ▶ NEXT: options

1. ~~Deeper lookahead with hold~~ done in v0.18. Possible follow-ups: a wider
   beam, or 4-ply, but each costs speed; harder tests are needed first
   (garbage every 4 is now never lost).
2. ~~Presentation polish~~ done in v0.16: network diagram, hold in the demo,
   classic vs hold scoreboard, "Score" label on the game page.
3. The CNN (raw board), as a separate learning project.
4. (Optional) More hold-training ideas: longer training, a penalty for
   losing, or starting from the current network instead of from scratch.

## Proposal 1 (closed: 5-step did not help)

Finish the 5-step check, then train more networks and keep the best.
Expected: 15–30% fewer lost games on random pieces (12 → about 8–10/50),
enough to beat the bot (9/50). Then try proposal 2 (a penalty for losing).

## Where we stopped on 20 October

**Task:** combine the basic recipe with our additions into one best recipe.
We keep the basic core (4 features, 4→64→64→1, same reward) and add
**n-step targets** (`--nstep N` in `train_dqn.js`, off by default, so the
defaults still rebuild the shipped `dqn-model.json`).

Results so far: 8 seeds each (1–8), selection games (random pieces, no
preview, seeds 70000+), average games lost out of 50:

| recipe | games lost per seed | average |
|---|---|---|
| plain (basic recipe) | 38, 36, 37, 32, 48, 44, 22, 14 | 34 |
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
