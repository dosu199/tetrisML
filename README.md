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

50 games each on seeds 90000+ (never seen in training), capped at 2,000 pieces:

| mode | player | mean pieces | mean lines | topped out |
|---|---|---|---|---|
| no preview, 7-bag | bot (search) | 1970.6 | 783.0 | 1/50 |
| no preview, 7-bag | **DQN** | **1902.4** | **753.0** | **5/50** |
| no preview, random pieces | bot (search) | 1877.5 | 741.8 | 9/50 |
| no preview, random pieces | **DQN** | **783.1** | **294.9** | **43/50** |

With the normal 7-bag generator the network is close to the hand-tuned bot. It
learned this in about 5 minutes, from nothing but the reward. With fully
random pieces it is much weaker, because it never saw that generator in
training (it trained on 7-bag only). The DQN decides from the current piece
alone, so whether you show it the next piece makes no difference.

## Files

```
src/tetris-core.js       board, pieces, 7-bag, collision, line clears, playEpisode()
src/tetris-dqn.js        the network: 4 features, MLP with backprop + Adam, chooseDqnMove()
src/tetris-ai.js         8 features + search bot (the opponent)
src/tetris-visual.js     the only file that touches the DOM
src/tetris-headless.js   fast games for the genetic algorithm
src/ga.js, ga-worker.js  genetic algorithm for the bot's weights
train_dqn.js             trains the network        -> dqn-model.json
hardmode.js              bot vs network, 5 rule modes
test_dqn.js              gradient check + smoke test for the network
bench.js                 26 tests for the game core
train.js / compare.js    genetic algorithm for the bot -> weights.json
```

## Commands

```bash
npm test                 # 26 core tests + the DQN checks
node train_dqn.js        # train a new network, about 5 minutes -> dqn-model.json
node hardmode.js         # bot vs network in all 5 modes, 10 games each
node hardmode.js --model dqn-model.json --games 50 --seed 90000 --mode no-preview
npx serve .              # then open index.html in the browser
```

Training options (all have defaults): `--epochs 3000 --decay 2000 --batch 512
--lr 1e-3 --gamma 0.99 --memory 30000 --seed 123 --out dqn-model.json`.

Training writes over `dqn-model.json`, so use `--out` to keep the old one
(for example `--out dqn-try2.json`). You can then compare the two networks in
one run:
`node hardmode.js --model dqn-model.json,dqn-try2.json --games 50 --seed 90000`.

## Watching it play in the browser

In `index.html`, replace your current game `<script type="module">` with:

```html
<select id="agent">
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
    dqn: (board, cur) => chooseDqnMove(board, cur, net),
    bot: (board, cur, next) => chooseMove(board, cur, next, weights, { lookahead: true }),
  };

  const statsEl = document.getElementById('agentStats');
  const game = startVisualGame({
    chooser: choosers.dqn,
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
the reward alone. The old code is in git history if you ever want it.
