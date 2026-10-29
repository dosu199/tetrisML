/**
 * tetris-controls.js - how a human moves a falling piece: move, rotate with
 * wall kicks, soft and hard drop, hold, key repeat (DAS / ARR), gravity and
 * lock delay. Used by both pages you can play on (Play and Play vs Bot), so
 * they feel exactly the same. No DOM: the pages turn keys and touches into
 * actions and call this.
 */

import { PIECES, PLAY_LEFT, PLAY_COLS, collides } from './tetris-core.js';

/** All times in milliseconds. */
export const TIMING = Object.freeze({
  lockDelay: 500, // a landed piece can still slide or turn for this long
  maxLockResets: 15, // ...and each slide or turn restarts that, this many times
  das: 160, // a held left/right starts repeating after this long
  arr: 45, // ...then repeats this often
  softDrop: 35, // per row while down is held
  maxFrame: 100, // longest frame step, so a background tab does not jump ahead
});

/** Where a turned piece may go if it does not fit as is: [rows, cols] to try, in order. */
const KICKS = Object.freeze([[0, 0], [0, -1], [0, 1], [0, -2], [0, 2], [-1, 0], [-1, -1], [-1, 1]]);

/** Keyboard (KeyboardEvent.code) to action. */
export const KEYMAP = Object.freeze({
  ArrowLeft: 'left',
  ArrowRight: 'right',
  ArrowDown: 'down',
  ArrowUp: 'cw',
  KeyX: 'cw',
  KeyZ: 'ccw',
  Space: 'drop',
  KeyC: 'hold',
  ShiftLeft: 'hold',
  ShiftRight: 'hold',
});

/** Classic scoring: points for clearing 0..4 lines with one piece. */
export const LINE_POINTS = Object.freeze([0, 40, 100, 300, 1200]);

/** Milliseconds per row of gravity: 800 at the start, 15% faster every 10 lines, never under 70. */
export const gravityMs = (lines) => Math.max(70, 800 * 0.85 ** Math.floor(lines / 10));

/** The game action for a key press, or null. Keys pressed with Ctrl, Cmd or Alt are the browser's. */
export function actionForKey(event) {
  if (event.ctrlKey || event.metaKey || event.altKey) return null;
  return KEYMAP[event.code] ?? null;
}

/** Column where a new piece appears: centred (rounded left). */
export const spawnColumn = (rotation) => PLAY_LEFT + Math.floor((PLAY_COLS - rotation.width) / 2);

/** Row where a piece at (row, col) would land if dropped: where the ghost is drawn. */
export function landingRow(board, rotation, row, col) {
  let r = row;
  while (!collides(board, rotation, r + 1, col)) r++;
  return r;
}

/**
 * The falling piece of one human player.
 *
 * The page owns the game (board, piece queue, score) and tells the
 * controller about it through callbacks; the controller owns the falling
 * piece, { r, row, col } (r = rotation index), and its timers.
 */
export class PieceController {
  /**
   * @param {object} game
   * @param {() => Uint8Array} game.board  the board the piece falls on
   * @param {() => number} game.piece      the current piece (index into PIECES)
   * @param {(r: number, row: number, col: number) => void} game.lock
   *        place the piece there for good; the page scores it, moves on to
   *        the next piece and calls spawn()
   * @param {() => void} [game.hold]       the hold key was pressed (the page decides whether that is allowed)
   * @param {() => void} [game.onSoftDropRow]  the piece fell one row because down is held
   */
  constructor(game) {
    this.game = game;
    this.active = null;
    this.keys = { left: null, right: null, down: false }; // left/right: { since, last } while held
    this.resetTimers();
  }

  resetTimers() {
    this.gravityAcc = 0;
    this.lockStart = null; // when the piece touched down (null = in the air)
    this.lockResets = 0;
  }

  /** The rotation (cells, width, ...) of the falling piece. */
  get rotation() {
    return PIECES[this.game.piece()].rotations[this.active.r];
  }

  fits(rotation, row, col) {
    return !collides(this.game.board(), rotation, row, col);
  }

  /** The current piece appears at the top, in the middle. False (and no piece) if it does not fit: topped out. */
  spawn() {
    const rotation = PIECES[this.game.piece()].rotations[0];
    const col = spawnColumn(rotation);
    this.resetTimers();
    this.active = this.fits(rotation, 0, col) ? { r: 0, row: 0, col } : null;
    return this.active !== null;
  }

  /** Takes over a piece that is already falling (an agent's) where it is. */
  adopt(r, row, col) {
    this.active = { r, row, col };
    this.resetTimers();
  }

  clear() {
    this.active = null;
  }

  /** Sliding or turning a landed piece gives it a little more time. */
  touchedOnGround(now) {
    if (this.lockStart != null && this.lockResets < TIMING.maxLockResets) {
      this.lockStart = now;
      this.lockResets++;
    }
  }

  move(dc, dr, now = performance.now()) {
    const a = this.active;
    if (!a || !this.fits(this.rotation, a.row + dr, a.col + dc)) return false;
    a.row += dr;
    a.col += dc;
    if (dc) this.touchedOnGround(now);
    return true;
  }

  rotate(dir, now = performance.now()) {
    const a = this.active;
    if (!a) return;
    const rotations = PIECES[this.game.piece()].rotations;
    if (rotations.length === 1) return;
    const r = (a.r + dir + rotations.length) % rotations.length;
    for (const [dr, dc] of KICKS) {
      if (this.fits(rotations[r], a.row + dr, a.col + dc)) {
        a.r = r;
        a.row += dr;
        a.col += dc;
        this.touchedOnGround(now);
        return;
      }
    }
  }

  hardDrop() {
    if (!this.active) return;
    this.active.row = landingRow(this.game.board(), this.rotation, this.active.row, this.active.col);
    this.lock();
  }

  lock() {
    const { r, row, col } = this.active;
    this.active = null;
    this.resetTimers();
    this.game.lock(r, row, col);
  }

  /** One action: 'left' 'right' 'down' 'cw' 'ccw' 'drop' 'hold'. Unknown actions are ignored. */
  run(action, now = performance.now()) {
    if (!this.active) return;
    if (action === 'left') this.move(-1, 0, now);
    else if (action === 'right') this.move(1, 0, now);
    else if (action === 'down') this.move(0, 1, now);
    else if (action === 'cw') this.rotate(1, now);
    else if (action === 'ccw') this.rotate(-1, now);
    else if (action === 'drop') this.hardDrop();
    else if (action === 'hold') this.game.hold?.();
  }

  /** A key went down: left/right move now and repeat while held; down soft-drops while held. */
  keyDown(action, now = performance.now()) {
    if (action === 'left' || action === 'right') {
      this.run(action, now);
      this.keys[action] = { since: now, last: now };
      this.keys[action === 'left' ? 'right' : 'left'] = null;
    } else if (action === 'down') {
      this.keys.down = true;
    } else {
      this.run(action, now);
    }
  }

  keyUp(action) {
    if (action === 'left' || action === 'right') this.keys[action] = null;
    if (action === 'down') this.keys.down = false;
  }

  setSoftDrop(on) {
    this.keys.down = !!on;
  }

  /** Forget held keys (the window lost focus, the game paused or restarted). */
  releaseAll() {
    this.keys.left = this.keys.right = null;
    this.keys.down = false;
  }

  /**
   * One frame: key repeat, gravity and lock delay.
   * @param {number} now  frame time (ms, performance.now() clock)
   * @param {number} dt   ms since the last frame
   * @param {number} gravity  ms per row (gravityMs(lines))
   */
  step(now, dt, gravity) {
    for (const dir of ['left', 'right']) {
      const k = this.keys[dir];
      if (k && now - k.since >= TIMING.das && now - k.last >= TIMING.arr) {
        this.run(dir, now);
        k.last = now;
      }
    }
    if (!this.active) return;

    this.gravityAcc += dt;
    const msPerRow = this.keys.down ? TIMING.softDrop : gravity;
    while (this.gravityAcc >= msPerRow) {
      this.gravityAcc -= msPerRow;
      if (!this.move(0, 1, now)) {
        this.gravityAcc = 0; // resting on something: gravity starts over when it can fall again
        break;
      }
      this.lockStart = null; // fell a row: a fresh lock delay when it lands
      if (this.keys.down) this.game.onSoftDropRow?.();
    }

    const a = this.active;
    const grounded = !this.fits(this.rotation, a.row + 1, a.col);
    if (!grounded) this.lockStart = null;
    else if (this.lockStart == null) this.lockStart = now;
    else if (now - this.lockStart >= TIMING.lockDelay) this.lock();
  }
}
