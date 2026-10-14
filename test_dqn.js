/**
 * test_dqn.js - checks for the DQN code.   node test_dqn.js
 *
 * 1. Gradient check: the backward pass in MLP.trainStep must agree with a
 *    numerical derivative (nudge each weight a tiny bit, see how the loss
 *    moves). If backprop had a bug, training would quietly learn nonsense.
 * 2. Smoke test: dqn-model.json loads and plays a legal game.
 */

import { readFileSync } from 'node:fs';
import { mulberry32, playEpisode } from './src/tetris-core.js';
import { MLP, FEATURE_SETS, loadDqn, chooseDqnMove } from './src/tetris-dqn.js';

let failed = 0;
const check = (name, ok, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  if (!ok) failed++;
};

// 1. gradient check on a tiny network
const r = mulberry32(7);
const net = new MLP([4, 5, 3, 1], r, FEATURE_SETS.basic);
const B = 6;
const X = new Float64Array(B * 4).map(() => r() * 4);
const y = new Float64Array(B).map(() => r() * 3);
const loss = (n) => {
  let s = 0;
  for (let i = 0; i < B; i++) s += (n.predict(X.subarray(i * 4, i * 4 + 4)) - y[i]) ** 2;
  return s / B;
};
// On Adam's first step its first moment is m = 0.1 x gradient, so a copy of the
// network that takes one step (with a negligible learning rate) exposes the
// analytic gradient as m / 0.1.
const probe = MLP.fromJSON(net.toJSON());
probe.trainStep(X, y, 1e-12);
let worst = 0;
for (let l = 0; l < net.W.length; l++) {
  for (const [arr, m] of [[net.W[l], probe.mW[l]], [net.b[l], probe.mb[l]]]) {
    for (let k = 0; k < arr.length; k++) {
      const old = arr[k];
      arr[k] = old + 1e-6;
      const lp = loss(net);
      arr[k] = old - 1e-6;
      const lm = loss(net);
      arr[k] = old;
      const numeric = (lp - lm) / 2e-6;
      const analytic = m[k] / 0.1;
      worst = Math.max(worst, Math.abs(numeric - analytic) / Math.max(1e-9, Math.abs(numeric) + Math.abs(analytic)));
    }
  }
}
check('backprop matches the numerical gradient', worst < 1e-4, `worst relative error ${worst.toExponential(1)}`);

// 2. the trained model loads and plays
let model = null;
try {
  model = loadDqn(JSON.parse(readFileSync('dqn-model.json', 'utf8')));
} catch (e) {
  check('dqn-model.json loads', false, e.message);
}
if (model) {
  const res = playEpisode((b, c) => chooseDqnMove(b, c, model), 1, { cap: 300 });
  check('dqn-model.json plays a game', res.pieces > 0, `${res.pieces} pieces, ${res.lines} lines`);
}

console.log(failed ? `\n${failed} check(s) failed\n` : '\nall DQN checks passed\n');
process.exit(failed ? 1 : 0);
