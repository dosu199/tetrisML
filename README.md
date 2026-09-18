# Phase 0 — headless Tetris core

Your game logic, pulled out of `main()` and off the DOM, so a learning
algorithm can play millions of pieces against it. Everything here is a
refactor of code you already wrote; no new game features, no framework, no
change to your board layout (still 16×27 with the 3-cell frame).

Verified: 26 sanity tests pass, plus a fake-DOM smoke test of the renderer.

## Baseline numbers

Measured on this machine with your published weights and a 500-piece cap:

| | 1-ply (training) | 2-ply (playing) |
|---|---|---|
| median lines / game | 196 | 199 |
| topped out | 0 of 100 | 0 of 20 |
| throughput | **11.7 games/s** (5,841 pieces/s) | 0.5 games/s |

The theoretical maximum is 200 lines in 500 pieces (four cells per piece, ten
per row), so the search is running at about 98% efficiency — that is the
evidence that the placement enumeration is now correct.

A typical GA run of 50 candidates × 5 seeds × 40 generations is about 10,000
games, which is **roughly 14 minutes single-threaded**. Rendered at your
current speed that same run would have taken months. That factor is the entire
reason for this phase.

## Bugs fixed

**The simulation was mutating the live game.** `calculateBestTetrominoPostion`
called `breakeRowsForAi(gameBoard, ...)` — the real board from the closure,
not `aiGameBoard`. Since that function deletes rows in place, every candidate
placement the AI considered was clearing rows out of the game you were
watching, and reporting the line count of the wrong board. There is now a
regression test (`chooseMove leaves the real board untouched`) that fails if
this ever comes back.

**Most placements were never evaluated.** `aiCurrentTetrominoPosition.x` was
never reset between columns, so after the first drop it held the landing row
and every later column started its drop buried inside the stack, hitting the
`if (collision) { y -= 1; continue; }` branch and being skipped. Columns are
now enumerated directly from the piece's width.

**Bumpiness measured the wrong quantity.** `numberOfBlocksInOneColumn` counts
filled *cells*; bumpiness needs column *heights*. The two are equal only on a
board with no holes — that is, only on boards where bumpiness matters least.

**`checkHorizontalWallCollision` did nothing.** It tested
`aIGameBoard[pos.x][pos.y + 1]` without adding `i` and `j`, so the nested
loops checked the same single cell sixteen times. It happened not to break
because of the 3-wide frame. Replaced by direct column bounds.

**The piece didn't always land where the AI decided.** `gameSpeed = 10` once
`x >= 4` started the fast drop whether or not the piece had finished rotating
and sliding, so a piece with far to travel could land somewhere that was never
evaluated. For a learning agent this is fatal: the move you score has to be
the move you made. The piece now takes its chosen rotation and column
immediately and then falls.

Also: the unused `tetromi` object (whose `S` was an O piece), the
`oneTetrominoSack[34]` magic numbers, the duplicated `while (rowsToBeBroken > 0)`
block in `breakeRows` that re-cleared rows and double-counted score, and the
`querySelectorAll` call for all 27 rows on every frame.

## Two deliberate behaviour changes

The piece randomiser is now a standard **7-bag** (shuffle all seven, deal them
out, repeat) instead of the five-array sack. The old one could deal the same
piece many times in a row, which adds variance that makes two weight vectors
hard to tell apart.

The board is a flat `Uint8Array` instead of an array of arrays. Cell `(r, c)`
is `board[r * WIDTH + c]`. Copying the board is the hottest operation in the
search — thousands of copies per piece — and `board.slice()` is far faster
than the nested loops in `copyBoard`.

## Files

```
src/tetris-core.js       board, pieces, seeded 7-bag, collision, drop, line clears
src/tetris-ai.js         8 features, evaluation, 1-ply and 2-ply search
src/tetris-headless.js   playGame(weights, seed) and evaluateWeights()  ← Phase 1 calls this
src/tetris-visual.js     the only file that touches the DOM
bench.js                 sanity tests + benchmark
demo.html                standalone demo page (your own index.html is untouched)
```

`tetris-core`, `tetris-ai` and `tetris-headless` never mention `document` or
`window`, which is why the same code runs in node and in the browser.

## Running it

```bash
node bench.js --test     # 26 tests
node bench.js            # tests, then the benchmark
npx serve .              # then open demo.html or index.html
```

`type="module"` scripts will not load over `file://`. In a Codespace, run
`npx serve .` and open the forwarded port — the VS Code file preview is not
enough.

## Wiring it into your existing page

Drop `src/`, `bench.js`, `package.json` and `demo.html` into your project root
next to `index.html`, `style.css` and `images/`. Then in your `index.html`,
replace the line that loads the old script:

```html
<!-- remove -->
<script src="main.js"></script>

<!-- add -->
<script type="module">
  import { startVisualGame } from './src/tetris-visual.js';
  startVisualGame({ msPerRow: 45, lookahead: true });
</script>
```

That is the only change to your HTML, and `style.css` needs none at all: the
renderer builds the same `#gameContainer > .row.row_N > .block` structure your
`createBlocks()` did, and writes to `#scoreBoard` and `#nextTetromino` the
same way. Sprite filenames are unchanged too (`red.png` … `purple.png`,
`nextRed.png` … `nextPurple.png`), so `images/` is untouched. If your sprites
live elsewhere, pass `imagePath`.

`main.js` is fully replaced — rename it to `main.old.js` until you have the
new version running in the browser, then delete it.

## Phase 1 — the genetic algorithm (done)

```bash
node train.js                 # 50 candidates, 40 generations
node train.js --population 50 --generations 8 --workers 2 --ceiling 2000
node compare.js --games 20 --cap 10000
```

`train.js` writes `weights.json` and `training-log.csv`. `compare.js` scores
the learned weights against the published ones on seeds starting at 10,000 —
games that were never used in training, which is the same discipline as the
train/test split from your course.

### What a real run looked like

50 candidates, 8 generations, two cores, 211 seconds, starting from completely
random weights:

| gen | best | median | mean | cap |
|---|---|---|---|---|
| 1 | 164.8 | 0.3 | 16.1 | 500 |
| 2 | 196.5 | 11.0 | 54.2 | 500 |
| 3 | 197.0 | 99.8 | 108.1 | 500 |
| 4 | 396.8 | 365.9 | 287.4 | 1000 ← raised |
| 5 | 797.5 | 794.1 | 711.4 | 2000 ← raised |
| 8 | 797.8 | 796.5 | 796.0 | 2000 |

Held out, 10 games at a 10,000-piece cap:

| | median | mean | worst game | topped out |
|---|---|---|---|---|
| published | 3994.5 | 3330.0 | 558 | 2 |
| learned | 3995.0 | **3914.5** | **3420** | 2 |

Read that carefully, because it is the interesting part. The medians are
identical — both sets of weights reach the cap on a typical game, so the median
has saturated and stopped measuring anything. The difference is in
**consistency**: the learned weights average 17.6% more lines and their worst
game was 3,420 lines against 558. The GA did not find a higher ceiling; it
found weights that fall apart far less often.

It also disagreed with the published weights about *why*. It weights
`aggregateHeight` much more heavily (-0.73 vs -0.51), cares far less about
`linesCleared` (0.24 vs 0.76), and put real weight on three features the
published vector ignores entirely — `rowTransitions`, `colTransitions` and
`wells` all came out around -0.3. Surface smoothness, measured as transitions,
turns out to matter more than the original four features could express.

### Things that will bite you

**Fitness saturates.** Once most of the population reaches the cap, every
candidate scores the same and selection pressure vanishes — learning silently
stops. The cap doubles automatically when the population median passes 90% of
the physical ceiling (0.4 lines per piece), which is what the "cap raised"
rows above are. Each raise re-scores the whole population, so those generations
take much longer.

**Use the same seeds for every candidate.** Different seeds means you are
ranking luck, and the population converges on whoever drew friendly pieces.

**The default ceiling is 20,000 pieces.** The run above used `--ceiling 2000`
to finish in three minutes. Letting it go to the default keeps selection
pressure on for far longer and produces better weights, but expect the last
generations to take several minutes each.

## Playing with the learned weights

```html
<script type="module">
  import { startVisualGame } from './src/tetris-visual.js';
  const { weights } = await fetch('./weights.json').then((r) => r.json());
  startVisualGame({ weights, msPerRow: 45, lookahead: true });
</script>
```

## Phase 2 starts here

The GA learned eight numbers. Phase 2 learns the *function* instead, and it is
the phase that uses your Udemy course directly.

The idea is behaviour cloning: let the Phase 1 bot play thousands of games and
log every decision — the board state before the move, and the placement it
chose. That is a labelled dataset, and everything after that is ordinary
supervised learning of exactly the kind the course covers. Load it with pandas,
split train and test, fit a model in scikit-learn, then a small network in
Keras, and measure both how closely it reproduces the bot's choices and how
well the resulting agent actually plays.

Two things make this a genuinely good exercise rather than busywork. The
dataset is one you generated, so you understand every column. And there is a
real question to answer at the end: a network that predicts the bot's move 95%
of the time may still play far worse than the bot, because the 5% it gets
wrong are the hard positions. Finding that out is the lesson.

The concrete next step is a `collect.js` that plays N games and writes a CSV of
`(features, chosen placement)` rows.
