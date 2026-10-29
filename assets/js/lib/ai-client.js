/**
 * ai-client.js - asks an AI player for its move, in the background worker
 * (assets/js/ai-worker.js) when possible, here on the page otherwise.
 *
 * If the worker cannot start or cannot load the models, every question
 * (including those already waiting) is answered on the page instead, so a
 * player never freezes waiting for an answer that will not come.
 */

import { makeChooser } from '../../../src/tetris-players.js';

/** A position to decide on: { board, cur, next, held, canHold }. */

/**
 * What a player answers, in a form that can be posted between threads:
 *   { hold: true }                       hold, then ask again (with canHold false)
 *   { move: { rotation, row, col } }     place the piece there
 *   { move: null }                       no legal move: topped out
 */
export function toReply(action) {
  if (action?.hold) return { hold: true };
  if (!action) return { move: null };
  return { move: { rotation: action.rotation, row: action.row, col: action.col } };
}

const choosers = new WeakMap(); // models -> { playerId -> chooser }

/** The chooser of player `id` for these models (made once, then reused). */
export function chooserFor(id, models) {
  if (!choosers.has(models)) choosers.set(models, {});
  const byId = choosers.get(models);
  byId[id] ??= makeChooser(id, models);
  return byId[id];
}

export class AiClient {
  /** @param {{ net, weights }} models  the page's own copy, for answering here when the worker fails */
  constructor(models) {
    this.models = models;
    this.waiting = new Map(); // id -> { resolve, player, position }
    this.nextId = 0;
    this.worker = null;
    try {
      this.worker = new Worker(new URL('../ai-worker.js', import.meta.url), { type: 'module' });
      this.worker.onmessage = (e) => this.onAnswer(e.data);
      this.worker.onerror = () => this.useMainThread();
      this.worker.onmessageerror = () => this.useMainThread();
    } catch {
      this.worker = null;
    }
  }

  /**
   * Player `player`'s decision for `position`.
   * @returns {Promise<{ hold: true } | { move: object | null }>}
   */
  decide(player, position) {
    if (!this.worker) return Promise.resolve(this.decideHere(player, position));
    return new Promise((resolve) => {
      const id = ++this.nextId;
      this.waiting.set(id, { resolve, player, position });
      this.worker.postMessage({ id, player, ...position });
    });
  }

  decideHere(player, { board, cur, next, held, canHold }) {
    return toReply(chooserFor(player, this.models)(board, cur, next, { held, canHold }));
  }

  onAnswer({ id, reply, error }) {
    const request = this.waiting.get(id);
    if (!request) return;
    if (error) {
      this.useMainThread(); // answers this request too
      return;
    }
    this.waiting.delete(id);
    request.resolve(reply);
  }

  /** The worker is unusable: stop it and answer everything here from now on. */
  useMainThread() {
    this.worker?.terminate();
    this.worker = null;
    for (const { resolve, player, position } of this.waiting.values()) resolve(this.decideHere(player, position));
    this.waiting.clear();
  }
}
