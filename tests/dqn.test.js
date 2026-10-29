/**
 * dqn.test.js - the network (src/tetris-dqn.js).
 *
 * 1. Gradient check: the backward pass in MLP.trainStep must agree with a
 *    numerical derivative (nudge each weight a tiny bit, see how the loss
 *    moves). If backprop had a bug, training would quietly learn nonsense.
 * 2. The shipped model loads and plays legal games: 1, 2 and 3-ply, with and
 *    without the hold box.
 * 3. A network that sees the hold box (11 inputs) saves, loads and plays.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createBoard, mulberry32, playEpisode, HOLD, PIECE_INDEX } from '../src/tetris-core.js';
import { MLP, FEATURE_SETS, loadDqn, chooseDqnMove, chooseDqnAction, bestFollow } from '../src/tetris-dqn.js';

const model = loadDqn(JSON.parse(readFileSync(new URL('../models/dqn-model.json', import.meta.url), 'utf8')));

test('backprop matches the numerical gradient', (t) => {
  const rand = mulberry32(7);
  const net = new MLP([4, 5, 3, 1], { rand });
  const B = 6;
  const X = new Float64Array(B * 4).map(() => rand() * 4);
  const y = new Float64Array(B).map(() => rand() * 3);
  const loss = (n) => {
    let s = 0;
    for (let i = 0; i < B; i++) s += (n.predict(X.subarray(i * 4, i * 4 + 4)) - y[i]) ** 2;
    return s / B;
  };
  // Adam's first moment after its first step is m = (1 - beta1) x gradient =
  // 0.1 x gradient, so a copy that takes one (negligible) step exposes the
  // analytic gradient as m / 0.1.
  const probe = MLP.fromJSON(net.toJSON());
  probe.trainStep(X, y, 1e-12);
  let worst = 0;
  for (let l = 0; l < net.W.length; l++) {
    for (const [params, m] of [[net.W[l], probe.mW[l]], [net.b[l], probe.mb[l]]]) {
      for (let k = 0; k < params.length; k++) {
        const old = params[k];
        params[k] = old + 1e-6;
        const lossUp = loss(net);
        params[k] = old - 1e-6;
        const lossDown = loss(net);
        params[k] = old;
        const numeric = (lossUp - lossDown) / 2e-6;
        const analytic = m[k] / 0.1;
        worst = Math.max(worst, Math.abs(numeric - analytic) / Math.max(1e-9, Math.abs(numeric) + Math.abs(analytic)));
      }
    }
  }
  t.diagnostic(`worst relative error ${worst.toExponential(1)}`);
  assert.ok(worst < 1e-4);
});

test('a model survives a save and load unchanged', () => {
  const back = loadDqn(JSON.parse(JSON.stringify(model.toJSON())));
  const x = Float64Array.from([1, 3, 12, 40]);
  assert.equal(back.predict(x), model.predict(x));
  assert.throws(() => loadDqn({ ...model.toJSON(), kind: 'other' }), /not a DQN model/);
});

test('the shipped model plays legal games', async (t) => {
  await t.test('1-ply', () => {
    const r = playEpisode((b, c) => chooseDqnMove(b, c, null, model), 1, { cap: 300 });
    assert.ok(r.pieces > 0);
  });
  await t.test('2-ply lookahead', () => {
    const r = playEpisode((b, c, n) => chooseDqnMove(b, c, n, model, { lookahead: true }), 1, { cap: 60 });
    assert.ok(r.pieces > 0);
  });
  await t.test('3-ply lookahead, with hold and garbage', () => {
    const r = playEpisode((b, c, n, h) => chooseDqnAction(b, c, n, model, h, { deep: true }), 5, { cap: 25, hold: true, garbageEvery: 4 });
    assert.ok(r.pieces > 0);
  });
  await t.test('without a hold context, chooseDqnAction plays exactly like chooseDqnMove', () => {
    const action = playEpisode((b, c, n, h) => chooseDqnAction(b, c, n, model, h), 3, { cap: 300 });
    const move = playEpisode((b, c) => chooseDqnMove(b, c, null, model), 3, { cap: 300 });
    assert.deepEqual(action, move);
  });
});

test('hold rules', async (t) => {
  await t.test('a player may hold only once per piece', () => {
    const alwaysHold = (b, c, n, h) => (h && h.canHold ? HOLD : chooseDqnMove(b, c, null, model));
    const r = playEpisode(alwaysHold, 3, { cap: 100, hold: true });
    assert.equal(r.holds, r.pieces);
  });
  await t.test('the network decides when to hold', () => {
    const r = playEpisode((b, c, n, h) => chooseDqnAction(b, c, n, model, h), 3, { cap: 300, hold: true });
    assert.ok(r.pieces > 0);
  });
});

test('a network that sees the hold box (11 inputs)', async (t) => {
  const holdNet = new MLP([FEATURE_SETS.hold.length, 8, 1], { rand: mulberry32(3), features: FEATURE_SETS.hold });
  const back = loadDqn({ ...holdNet.toJSON(), kind: 'dqn' });
  await t.test('saves, loads and plays with hold', () => {
    assert.equal(back.inputSpec.held.length, 7);
    const r = playEpisode((b, c, n, h) => chooseDqnAction(b, c, n, back, h), 4, { cap: 40, hold: true });
    assert.ok(r.pieces > 0);
  });
  await t.test('scores real numbers in a search without hold', () => {
    // regression: the search's scratch input used to have 8 slots, so inputs 9-11 read as NaN
    assert.ok(Number.isFinite(bestFollow(createBoard(), PIECE_INDEX.I, back)));
  });
});
