/**
 * ai-worker.js - an AI player's thinking in a background thread (a Web
 * Worker), so even the slow 3-ply search (about 0.1 s) never makes your own
 * piece stutter. Used through assets/js/lib/ai-client.js, by Play vs Bot
 * (the opponent) and by Play (the hint).
 *
 * Message in:  { id, player, board, cur, next, held, canHold }   player: a key of PLAYERS
 * Message out: { id, reply }   reply: see toReply() in lib/ai-client.js
 *              { id, error }   the models could not be loaded, or the player failed
 */

import { loadModels } from './lib/models.js';
import { chooserFor, toReply } from './lib/ai-client.js';

const models = loadModels();
models.catch(() => {}); // reported per request below

self.onmessage = async (e) => {
  const { id, player, board, cur, next, held, canHold } = e.data;
  try {
    const choose = chooserFor(player, await models);
    self.postMessage({ id, reply: toReply(choose(board, cur, next, { held, canHold })) });
  } catch (err) {
    self.postMessage({ id, error: String(err?.message ?? err) });
  }
};
