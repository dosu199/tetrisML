/**
 * test_dqn.js - checks for the DQN code.   node test_dqn.js
 *
 * 1. Gradient check: the backward pass in MLP.trainStep must agree with a
 *    numerical derivative (nudge each weight a tiny bit, see how the loss
 *    moves). If backprop had a bug, training would quietly learn nonsense.
 * 2. Smoke test: dqn-model.json loads and plays a legal game, with and
 *    without the 2-ply and 3-ply lookahead, and with the hold box.
 * 3. A network that sees the hold box (11 inputs) saves, loads and plays.
 */

import { readFileSync } from 'node:fs';
import { mulberry32, playEpisode, HOLD } from './src/tetris-core.js';
import { MLP, FEATURE_SETS, loadDqn, chooseDqnMove, chooseDqnAction } from './src/tetris-dqn.js';

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
  const res2 = playEpisode((b, c, n) => chooseDqnMove(b, c, model, n, { lookahead: true }), 1, { cap: 60 });
  check('2-ply lookahead plays a game', res2.pieces > 0, `${res2.pieces} pieces, ${res2.lines} lines`);

  // hold rules: a chooser that tries to hold every time may hold only once
  // per piece, so every turn is exactly one hold and then one placement
  const always = (b, c, n, h) => (h && h.canHold ? HOLD : chooseDqnMove(b, c, model));
  const r3 = playEpisode(always, 3, { cap: 100, hold: true });
  check('hold: once per piece, game stays legal', r3.holds === r3.pieces, `${r3.pieces} pieces, ${r3.holds} holds`);
  const r4 = playEpisode((b, c, n, h) => chooseDqnAction(b, c, model, n, h), 3, { cap: 300, hold: true });
  check('hold-aware network plays a game', r4.pieces > 0, `${r4.pieces} pieces, ${r4.holds} holds`);
  const r5 = playEpisode((b, c, n, h) => chooseDqnAction(b, c, model, n, h), 3, { cap: 300 });
  const r6 = playEpisode((b, c) => chooseDqnMove(b, c, model), 3, { cap: 300 });
  check('without hold, the new chooser plays exactly like before', r5.lines === r6.lines && r5.pieces === r6.pieces, `${r5.lines} = ${r6.lines} lines`);
  const r8 = playEpisode((b, c, n, h) => chooseDqnAction(b, c, model, n, h, { deep: true }), 5, { cap: 25, hold: true, garbageEvery: 4 });
  check('deep (3-ply) lookahead plays a game', r8.pieces > 0, `${r8.pieces} pieces, ${r8.lines} lines`);
}

// 3. a network that sees the hold box (11 inputs) saves, loads and plays
{
  const hnet = new MLP([FEATURE_SETS.hold.length, 8, 1], mulberry32(3), FEATURE_SETS.hold, false);
  const back = loadDqn({ ...hnet.toJSON(), kind: 'dqn' });
  const r7 = playEpisode((b, c, n, h) => chooseDqnAction(b, c, back, n, h), 4, { cap: 40, hold: true });
  check('hold-aware network (11 inputs) loads and plays', back.input.held.length === 7 && r7.pieces > 0, `${r7.pieces} pieces, ${r7.holds} holds`);
}

console.log(failed ? `\n${failed} check(s) failed\n` : '\nall DQN checks passed\n');
process.exit(failed ? 1 : 0);
