/**
 * regression.test.js - pins down exactly how every player plays.
 *
 * Each case plays one fixed game and compares the result and a fingerprint
 * of every move (a hash of the move list) with the values recorded from the
 * current code. A refactor that changes a single move, or a single bit of a
 * training step, fails here. If a change is MEANT to change play, re-record:
 * run with RECORD=1 and paste the printed values below.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { playEpisode, mulberry32 } from '../src/tetris-core.js';
import { chooseMoveAction } from '../src/tetris-ai.js';
import { loadDqn, chooseDqnAction, MLP } from '../src/tetris-dqn.js';

const readJson = (path) => JSON.parse(readFileSync(new URL(path, import.meta.url), 'utf8'));
const net = loadDqn(readJson('../models/dqn-model.json'));
const { weights } = readJson('../models/weights.json');

const bot = (opts) => (b, c, n, h) => chooseMoveAction(b, c, n, weights, h, opts);
const network = (opts) => (b, c, n, h) => chooseDqnAction(b, c, n, net, h, opts);

const CASES = [
  { name: 'bot 2-ply', chooser: bot(), seed: 1, rules: { cap: 150 },
    expected: { lines: 59, pieces: 150, died: false, holds: 0, tetrises: 0, moves: '55f54feee3246c3d' } },
  { name: 'bot 1-ply, memoryless pieces', chooser: bot({ lookahead: false }), seed: 2, rules: { cap: 150, randomizer: 'memoryless' },
    expected: { lines: 58, pieces: 150, died: false, holds: 0, tetrises: 0, moves: '1bb8b3b4f9e678f5' } },
  { name: 'bot 2-ply, loses to garbage', chooser: bot(), seed: 6, rules: { cap: 400, garbageEvery: 2 },
    expected: { lines: 21, pieces: 50, died: true, holds: 0, tetrises: 0, moves: '6ff0b1c7c43a3f71' } },
  { name: 'network 1-ply, no preview', chooser: network(), seed: 3, rules: { cap: 150, preview: false },
    expected: { lines: 56, pieces: 150, died: false, holds: 0, tetrises: 1, moves: 'e426bd14f105d89d' } },
  { name: 'network 1-ply, loses to garbage', chooser: network(), seed: 7, rules: { cap: 400, garbageEvery: 2 },
    expected: { lines: 19, pieces: 48, died: true, holds: 0, tetrises: 0, moves: '02459844da2faaef' } },
  { name: 'network 2-ply, hold', chooser: network({ lookahead: true }), seed: 4, rules: { cap: 80, hold: true },
    expected: { lines: 29, pieces: 80, died: false, holds: 23, tetrises: 1, moves: '2af1dbb76d32d95b' } },
  { name: 'network 3-ply, garbage', chooser: network({ deep: true }), seed: 5, rules: { cap: 30, garbageEvery: 3 },
    expected: { lines: 12, pieces: 30, died: false, holds: 0, tetrises: 0, moves: '915df910d653baa2' } },
];

/** playEpisode's result plus a 16-hex-digit hash of every action taken. */
function playRecorded(chooser, seed, rules) {
  const hash = createHash('sha256');
  const recording = (board, current, next, holdCtx) => {
    const m = chooser(board, current, next, holdCtx);
    hash.update(JSON.stringify(m && (m.hold ? 'H' : [m.rotation, m.col, m.row])));
    return m;
  };
  return { ...playEpisode(recording, seed, rules), moves: hash.digest('hex').slice(0, 16) };
}

for (const { name, chooser, seed, rules, expected } of CASES) {
  test(`plays exactly as recorded: ${name}`, () => {
    const actual = playRecorded(chooser, seed, rules);
    if (process.env.RECORD) console.log(name, JSON.stringify(actual));
    assert.deepEqual(actual, expected);
  });
}

test('training steps are bit-for-bit reproducible', () => {
  const m = new MLP([4, 8, 8, 1], { rand: mulberry32(9) });
  const X = new Float64Array(16).map((_, i) => (i * 7) % 5);
  const y = new Float64Array(4).map((_, i) => i);
  assert.deepEqual([m.trainStep(X, y, 1e-3), m.trainStep(X, y, 1e-3)], [14.160004833728228, 13.863373956674995]);
});
