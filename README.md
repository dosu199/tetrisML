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

The shipped `dqn-model.json`. 50 games per row on seeds 90000+ (never seen in
training). "1-ply" means it decides on the current piece only. "2-ply" means
it also looks at the next piece.

| pieces | preview | cap | bot (search) | DQN 1-ply | DQN 2-ply |
|---|---|---|---|---|---|
| 7-bag | no | 10,000 | 8/50 lost | **1/50 lost** | (needs preview) |
| fully random | no | 2,000 | **9/50 lost** | 16/50 lost | (needs preview) |
| fully random | yes | 2,000 | **0/50 lost** | 16/50 lost | **0/50 lost** |
| 7-bag + garbage row every 5 pieces | yes | — | 254 pieces | 148 pieces | **321 pieces** |

(In the garbage mode everyone eventually tops out, so the number is mean
pieces survived; higher is better.)

- **Without preview:** the network beats the bot on normal pieces. The bot is
  still better on fully random pieces.
- **With preview:** the network with lookahead matches the bot on random
  pieces (neither lost a game) and beats it clearly in the hardest mode, 321
  vs 254 pieces with rising garbage. It never trained on garbage at all.

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
typical training run is weaker than ours; seed 124 was lucky. Making a typical
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
--lr 1e-3 --gamma 0.99 --memory 30000 --seed 124 --pieces random --features basic --out dqn-model.json`.
`--features full` uses all 8 bot features (scaled automatically); `--scale`
turns scaling on for the 4 basic ones. Training is deterministic: the
defaults reproduce the shipped `dqn-model.json` exactly.

Training writes over `dqn-model.json`, so use `--out` to keep the old one
(for example `--out dqn-try2.json`). You can then compare the two networks in
one run:
`node hardmode.js --model dqn-model.json,dqn-try2.json --games 50 --seed 90000`.

## Watching it play in the browser

In `index.html`, replace your current game `<script type="module">` with:

```html
<select id="agent">
  <option value="dqn2">Neural network + lookahead</option>
  <option value="dqn">Neural network (DQN)</option>
  <option value="bot">Bot (search)</option>
</select>
<div id="agentStats"></div>

<script type="module">
  import { startVisualGame } from './src/tetris-visual.js';
  import { chooseMove } from './src/tetris-ai.js';
  import { loadDqn, chooseDqnMove } from './src/tetris-dqn.js';

  const json = (f) => fetch(f).then((r) => r.json());
  const weights = (await json('./weights.json')).weights;
  const net = loadDqn(await json('./dqn-model.json'));

  const choosers = {
    dqn2: (board, cur, next) => chooseDqnMove(board, cur, net, next, { lookahead: true }),
    dqn: (board, cur) => chooseDqnMove(board, cur, net),
    bot: (board, cur, next) => chooseMove(board, cur, next, weights, { lookahead: true }),
  };

  const statsEl = document.getElementById('agentStats');
  const game = startVisualGame({
    chooser: choosers.dqn2,
    msPerRow: 45,
    imagePath: 'images/',
    onStats: (s) => {
      statsEl.textContent = `game ${s.games} · lines ${s.lines} · pieces ${s.pieces} · best ${s.bestLines} lines`;
    },
  });

  document.getElementById('agent').addEventListener('change', (e) => {
    game.setChooser(choosers[e.target.value]);
  });
</script>
```

`type="module"` scripts do not load over `file://`. In a Codespace, run
`npx serve .` and open the forwarded port.

## History

The project began with a genetic algorithm that tuned the bot's 8 weights.
That work is still used: it produced `weights.json`. Next came imitation
learning, where networks copied the bot (MLP, ranking net, CNN, DAgger). That
works, but it can never beat its teacher, and it needed a large Python
pipeline. Version 0.4 replaced it with this much smaller DQN, which learns from
the reward alone. Version 0.5 added feature sets and scaling, and picked the
best of several training seeds. Version 0.6 trains on random pieces.
Version 0.7 adds the 2-ply lookahead and `train_many.js`. The old code is in git history if you ever want it.
