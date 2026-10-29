/**
 * tetris-players.js - the AI players the website offers, in one place: their
 * names and how each one decides. The Play page, Play vs Bot and its worker
 * all use this list, so a player is named and behaves the same everywhere.
 */

import { chooseMoveAction } from './tetris-ai.js';
import { chooseDqnAction } from './tetris-dqn.js';

/**
 * id -> { label, detail, name, short, search, bot }
 *   label, detail  what the menus say: how strong it is, and how it gets there
 *   name, short    the full and the short name of the player (match headers)
 *   search         how it looks ahead (passed to chooseDqnAction / chooseMoveAction)
 *   bot            true for the search bot (weights instead of the network)
 */
export const PLAYERS = Object.freeze({
  dqn3: Object.freeze({ label: 'Strongest', detail: '3 ahead', name: 'Network, 3 ahead', short: 'Network 3', search: Object.freeze({ deep: true }) }),
  dqn2: Object.freeze({ label: 'Strong', detail: '2 ahead', name: 'Network, 2 ahead', short: 'Network 2', search: Object.freeze({ lookahead: true }) }),
  dqn: Object.freeze({ label: 'Quick', detail: '1 piece', name: 'Network, 1 piece', short: 'Network 1', search: Object.freeze({}) }),
  bot: Object.freeze({ label: 'Search bot', detail: 'no network', name: 'Search bot', short: 'Bot', bot: true, search: Object.freeze({ lookahead: true }) }),
});

export const DEFAULT_PLAYER = 'dqn3';

/**
 * A chooser for player `id`:
 *   (board, current, next, holdCtx) -> placement | HOLD | null (no move: game over)
 * @param {string} id  a key of PLAYERS
 * @param {{ net: import('./tetris-dqn.js').MLP, weights: number[] }} models
 */
export function makeChooser(id, { net, weights }) {
  const player = PLAYERS[id];
  if (!player) throw new Error(`unknown player: ${id}`);
  if (player.bot) return (board, current, next, holdCtx) => chooseMoveAction(board, current, next, weights, holdCtx, player.search);
  return (board, current, next, holdCtx) => chooseDqnAction(board, current, next, net, holdCtx, player.search);
}
