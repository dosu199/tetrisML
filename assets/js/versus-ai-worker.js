/**
 * versus-ai-worker.js - the opponent's thinking for versus.html, in a
 * background thread (a Web Worker), so a slow decision (the 3-ply search
 * takes about 0.1 s) never makes your own piece stutter.
 *
 * Message in:  { id, kind, board, cur, next, held, canHold }
 * Message out: { id, hold: true }                    -> hold, then ask again
 *              { id, move: { rotation, row, col } }  -> place here
 *              { id, move: null }                    -> no legal move (topped out)
 */

import { loadDqn, chooseDqnAction } from '../../src/tetris-dqn.js';
import { chooseMoveAction, PUBLISHED_WEIGHTS } from '../../src/tetris-ai.js';

let net = null;
let weights = PUBLISHED_WEIGHTS;

const ready = (async () => {
  net = loadDqn(await fetch(new URL('../../models/dqn-model.json', import.meta.url)).then((r) => r.json()));
  try {
    const w = await fetch(new URL('../../models/weights.json', import.meta.url)).then((r) => (r.ok ? r.json() : null));
    if (w?.weights) weights = w.weights;
  } catch {}
})();

function decide({ kind, board, cur, next, held, canHold }) {
  const hold = { held, canHold };
  if (kind === 'bot') return chooseMoveAction(board, cur, next, weights, hold, { lookahead: true });
  if (kind === 'dqn') return chooseDqnAction(board, cur, net, next, hold);
  if (kind === 'dqn2') return chooseDqnAction(board, cur, net, next, hold, { lookahead: true });
  return chooseDqnAction(board, cur, net, next, hold, { deep: true }); // dqn3
}

self.onmessage = async (e) => {
  await ready;
  const { id } = e.data;
  const m = decide(e.data);
  if (m && m.hold) self.postMessage({ id, hold: true });
  else if (!m || m.score === -Infinity) self.postMessage({ id, move: null });
  else self.postMessage({ id, move: { rotation: m.rotation, row: m.row, col: m.col } });
};
