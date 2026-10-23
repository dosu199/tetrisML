/**
 * tetris-visual.js - everything that touches the DOM, and nothing else.
 *
 * This is the only file in src/ that knows a browser exists. It targets the
 * same element ids your current page uses (#gameContainer, #scoreBoard,
 * #nextTetromino) and the same .row_N / .block class names, so your CSS should
 * carry over untouched.
 *
 * Two behavioural changes worth knowing about:
 *
 *  - Block elements are looked up once and cached. The original ran
 *    querySelectorAll(".row_" + i) for all 27 rows on every single frame.
 *  - The piece is put into its chosen rotation and column immediately, then
 *    falls. The original rotated one step and slid one column per tick while
 *    also switching to gameSpeed = 10 once it passed row 4, so a piece with a
 *    long way to travel could land somewhere the AI never evaluated. For a
 *    learning agent that is fatal: the move it is scored on has to be the move
 *    it actually made.
 *
 * Human mode (v0.19): with { human: true } (or setHuman(true)) you play it
 * yourself with the keyboard or the touch buttons; the agent is switched off.
 */

import {
  WIDTH,
  HEIGHT,
  PLAY_LEFT,
  PLAY_COLS,
  PLAY_BOTTOM,
  idx,
  createBoard,
  Bag,
  PIECES,
  placePiece,
  clearLines,
  collides,
} from './tetris-core.js';
import { chooseMove, PUBLISHED_WEIGHTS } from './tetris-ai.js';

const IMAGES = {
  1: 'red',
  2: 'skyBlue',
  3: 'yellow',
  4: 'green',
  5: 'orange',
  6: 'blue',
  7: 'purple',
};

// Fallback colours, so the board is still readable if the images are missing.
const COLORS = {
  1: '#e6483d',
  2: '#4ec3e0',
  3: '#f2c94c',
  4: '#5fc27e',
  5: '#f2994a',
  6: '#4a7ef2',
  7: '#9b51e0',
  8: '#e6eaf4',
};

const NEXT_IMAGES = {
  0: 'nextRed', // Z
  1: 'nextSkyBlue', // I
  2: 'nextYellow', // O
  3: 'nextGreen', // S
  4: 'nextOrange', // L
  5: 'nextBlue', // J
  6: 'nextPurple', // T
};

/* ---------------------------------------------------------------- renderer */

export class Renderer {
  constructor(container, { imagePath = 'images/' } = {}) {
    this.imagePath = imagePath;
    this.blocks = new Array(WIDTH * HEIGHT);

    container.innerHTML = '';
    for (let r = 0; r < HEIGHT; r++) {
      const row = document.createElement('div');
      row.classList.add('row', 'row_' + r);
      for (let c = 0; c < WIDTH; c++) {
        const block = document.createElement('div');
        block.classList.add('block');
        row.appendChild(block);
        this.blocks[idx(r, c)] = block;
      }
      container.appendChild(row);
    }
  }

  paint(el, value, opacity = '1') {
    el.style.opacity = opacity;
    if (value === 0) {
      el.style.backgroundImage = 'none';
      el.style.backgroundColor = '';
      return;
    }
    if (value === 8) {
      el.style.backgroundImage = 'none';
      el.style.backgroundColor = COLORS[8];
      return;
    }
    el.style.backgroundImage = `url(${this.imagePath}${IMAGES[value]}.png)`;
    el.style.backgroundColor = COLORS[value];
    el.style.opacity = opacity;
  }

  /** Draws the settled board plus the falling piece and its ghost. */
  draw(board, piece) {
    for (let i = 0; i < board.length; i++) this.paint(this.blocks[i], board[i]);

    if (!piece) return;
    const { rotation, row, col } = piece;

    // ghost: where it would land from here
    let ghostRow = row;
    while (!collides(board, rotation, ghostRow + 1, col)) ghostRow++;
    for (const [dr, dc] of rotation.cells) {
      this.paint(this.blocks[idx(ghostRow + dr, col + dc)], rotation.value, '0.3');
    }
    for (const [dr, dc] of rotation.cells) {
      this.paint(this.blocks[idx(row + dr, col + dc)], rotation.value, '1');
    }
  }
}

/* ------------------------------------------------------------------- game */

// Human play: same timings as the Play vs Bot page
const LOCK_DELAY = 500; // ms a landed piece can still slide or turn
const MAX_LOCK_RESETS = 15;
const DAS = 160; // ms before a held left/right starts repeating
const ARR = 45; // ms between repeats
const SOFT_DROP = 35; // ms per row while holding down
const KICKS = [[0, 0], [0, -1], [0, 1], [0, -2], [0, 2], [-1, 0], [-1, -1], [-1, 1]];
const KEYMAP = {
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
};

export function startVisualGame(options = {}) {
  let weights = options.weights || PUBLISHED_WEIGHTS;
  let msPerRow = options.msPerRow ?? 40;
  const {
    seed = Date.now() & 0xffff,
    lookahead = true,
    imagePath = 'images/',
    containerId = 'gameContainer',
    scoreId = 'scoreBoard',
    nextId = 'nextTetromino',
    holdId = 'holdTetromino',
    hold = false, // modern Tetris hold box; choosers that ignore it never hold
  } = options;

  const container = document.getElementById(containerId);
  if (!container) throw new Error(`No #${containerId} element on the page`);
  const scoreBoard = document.getElementById(scoreId);
  const nextBoard = document.getElementById(nextId);
  const holdBoard = document.getElementById(holdId);

  const renderer = new Renderer(container, { imagePath });

  let board = createBoard();
  let bag = new Bag(seed);
  let current = bag.next();
  let next = bag.next();
  let score = 0;
  let lines = 0;
  let held = null; // the piece in the hold box, or null
  let holds = 0;

  // Stats that survive a game ending, so the page can show what an agent is
  // actually worth over several games rather than one lucky or unlucky run.
  let games = 1;
  let pieces = 0;
  let bestLines = 0;
  let totalLines = 0;

  /**
   * Who decides where the piece goes. Swappable at runtime, which is the whole
   * point: the search bot, the regression network and the ranking network all
   * satisfy the same signature, so you can watch them play the same game.
   *
   *   (board, currentPiece, nextPiece, holdCtx) -> { rotation, col, row } | HOLD | null
   *
   * holdCtx = { held, canHold } is passed only when the game has hold on.
   */
  let chooser =
    options.chooser || ((b, c, n) => chooseMove(b, c, n, weights, { lookahead }));

  // human mode state
  let human = !!options.human;
  let phase = 'playing'; // ready | playing | paused | over (human mode only)
  let canHold = true;
  let gravityAcc = 0;
  let lockStart = null;
  let lockResets = 0;
  let overAt = 0;
  const keys = { left: null, right: null, down: false };

  let active = null; // { rotation, row, col, target } (agent) or { r, rotation, row, col } (human)
  let accumulator = 0;
  let lastTime = 0;
  let running = true;

  const setPhase = (p, info = {}) => {
    phase = p;
    options.onState?.(p, { lines, score, pieces, ...info });
  };

  function showNext() {
    if (nextBoard) nextBoard.style.backgroundImage = `url(${imagePath}${NEXT_IMAGES[next]}.png)`;
  }

  function showHold() {
    if (holdBoard) holdBoard.style.backgroundImage = held == null ? 'none' : `url(${imagePath}${NEXT_IMAGES[held]}.png)`;
  }

  function showScore() {
    if (scoreBoard) scoreBoard.innerHTML = String(score);
    options.onStats?.({
      score,
      lines,
      pieces,
      games,
      holds,
      bestLines,
      avgLines: totalLines / Math.max(games - 1, 1),
      finishedGames: games - 1,
      human,
    });
  }

  function swapHold() {
    holds++;
    if (held == null) {
      held = current;
      current = next;
      next = bag.next();
    } else {
      [held, current] = [current, held];
    }
    showHold();
    showNext();
  }

  function spawn() {
    // A network chooser returns no score field; only the search bot does, and
    // -Infinity is its way of saying every remaining move loses.
    let move = chooser(board, current, next, hold ? { held, canHold: true } : undefined);
    if (hold && move && move.hold) {
      // put the current piece in the hold box and take the held one (or the
      // next one, if the box was empty); then the piece we got must be placed
      swapHold();
      move = chooser(board, current, next, { held, canHold: false });
    }
    if (move === null || move.hold || move.score === -Infinity) return false;
    active = {
      rotation: PIECES[current].rotations[move.rotation],
      row: 0,
      col: move.col,
      target: move.row,
    };
    return true;
  }

  /** Human mode: the new piece appears at the top in the middle. */
  function spawnHuman() {
    const rotation = PIECES[current].rotations[0];
    const col = PLAY_LEFT + Math.floor((PLAY_COLS - rotation.width) / 2);
    active = { r: 0, rotation, row: 0, col };
    gravityAcc = 0;
    lockStart = null;
    lockResets = 0;
    showNext();
    if (collides(board, rotation, 0, col)) {
      active = null;
      totalLines += lines;
      if (lines > bestLines) bestLines = lines;
      overAt = performance.now();
      setPhase('over');
      return false;
    }
    return true;
  }

  function reset(hard = false, ready = false) {
    if (hard) {
      games = 1;
      bestLines = 0;
      totalLines = 0;
    } else {
      if (!human) {
        // (in human mode these were already counted when the game ended)
        totalLines += lines;
        if (lines > bestLines) bestLines = lines;
      }
      games++;
    }
    board = createBoard();
    bag = new Bag((Date.now() & 0xffff) + games);
    current = bag.next();
    next = bag.next();
    score = 0;
    lines = 0;
    pieces = 0;
    held = null;
    holds = 0;
    canHold = true;
    active = null;
    keys.left = keys.right = null;
    keys.down = false;
    showHold();
    showScore();
    if (human) {
      spawnHuman();
      setPhase(ready ? 'ready' : 'playing'); // 'ready' waits for a key before gravity starts
    }
  }

  /** Puts the active piece into the board, clears lines and scores. */
  function lockActive() {
    placePiece(board, active.rotation, active.row, active.col);
    pieces++;
    const cleared = clearLines(board);
    if (cleared > 0) {
      lines += cleared;
      // classic scoring: a quadruple is worth far more than four singles
      score += [0, 40, 100, 300, 1200][cleared];
    }
    showScore();
    active = null;
    current = next;
    next = bag.next();
    canHold = true;
  }

  function tick() {
    if (!active) {
      if (!spawn()) {
        reset();
        return;
      }
      showNext();
      return;
    }

    if (active.row < active.target) {
      active.row++;
      return;
    }

    lockActive(); // landed
  }

  /* ---- human controls ---- */

  const fits = (rotation, row, col) => !collides(board, rotation, row, col);

  function afterMoveOnGround() {
    // sliding or turning a landed piece gives it a little more time
    if (lockStart != null && lockResets < MAX_LOCK_RESETS) {
      lockStart = performance.now();
      lockResets++;
    }
  }

  function move(dc, dr) {
    if (!active || !fits(active.rotation, active.row + dr, active.col + dc)) return false;
    active.row += dr;
    active.col += dc;
    if (dc) afterMoveOnGround();
    return true;
  }

  function rotate(dir) {
    if (!active) return;
    const rots = PIECES[current].rotations;
    if (rots.length === 1) return;
    const r = (active.r + dir + rots.length) % rots.length;
    for (const [dr, dc] of KICKS) {
      if (fits(rots[r], active.row + dr, active.col + dc)) {
        active.r = r;
        active.rotation = rots[r];
        active.row += dr;
        active.col += dc;
        afterMoveOnGround();
        return;
      }
    }
  }

  function lockHuman() {
    lockActive();
    spawnHuman();
  }

  function hardDrop() {
    if (!active) return;
    while (fits(active.rotation, active.row + 1, active.col)) active.row++;
    lockHuman();
  }

  function holdHuman() {
    if (!hold || !canHold || !active) return;
    swapHold();
    canHold = false;
    spawnHuman();
  }

  const ACTIONS = {
    left: () => move(-1, 0),
    right: () => move(1, 0),
    down: () => move(0, 1),
    cw: () => rotate(1),
    ccw: () => rotate(-1),
    drop: hardDrop,
    hold: holdHuman,
  };

  function go() {
    lastTime = 0;
    setPhase('playing');
  }

  function setPaused(on) {
    if (!human) return;
    if (on && phase === 'playing') setPhase('paused');
    else if (!on && phase === 'paused') {
      lastTime = 0;
      setPhase('playing');
    }
  }

  /** One input from the keyboard or a touch button. */
  function press(action) {
    if (!human || !running) return;
    if (phase === 'ready') {
      if (action !== 'pause') go();
      return;
    }
    if (action === 'pause') return setPaused(phase === 'playing');
    if (phase === 'over') {
      // Space restarts too, but not in the first second, so a hard drop
      // that was already on its way does not skip the result
      if (action === 'restart' || (action === 'drop' && performance.now() - overAt > 1000)) reset();
      return;
    }
    if (phase !== 'playing') return;
    ACTIONS[action]?.();
  }

  function onKeyDown(e) {
    if (!human || !running) return;
    if (e.target.closest?.('select, input, textarea, button')) return;
    if (e.code === 'KeyP' || e.code === 'Escape') {
      setPaused(phase === 'playing');
      return;
    }
    if (e.code === 'Enter' && phase === 'over') {
      reset();
      return;
    }
    if (phase === 'ready' && (KEYMAP[e.code] || e.code === 'Enter')) {
      e.preventDefault();
      go();
      return;
    }
    const action = KEYMAP[e.code];
    if (!action) return;
    e.preventDefault(); // no page scrolling while playing
    if (e.repeat) return; // we do our own repeat for left/right/down
    if (phase !== 'playing') {
      press(action);
      return;
    }
    if (action === 'left' || action === 'right') {
      ACTIONS[action]();
      const t = performance.now();
      keys[action] = { since: t, last: t };
      keys[action === 'left' ? 'right' : 'left'] = null;
    } else if (action === 'down') {
      keys.down = true;
    } else {
      ACTIONS[action]();
    }
  }

  function onKeyUp(e) {
    const action = KEYMAP[e.code];
    if (action === 'left' || action === 'right') keys[action] = null;
    if (action === 'down') keys.down = false;
  }

  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);

  function humanStep(t, dt) {
    if (phase !== 'playing') return;
    for (const dir of ['left', 'right']) {
      const k = keys[dir];
      if (k && t - k.since >= DAS && t - k.last >= ARR) {
        ACTIONS[dir]();
        k.last = t;
      }
    }
    if (!active) return;
    gravityAcc += dt;
    const level = Math.floor(lines / 10);
    const g = keys.down ? SOFT_DROP : Math.max(70, 800 * 0.85 ** level);
    while (gravityAcc >= g) {
      gravityAcc -= g;
      if (!move(0, 1)) {
        gravityAcc = 0;
        break;
      }
      if (keys.down) score += 1; // a point per row of soft drop
    }
    const grounded = !fits(active.rotation, active.row + 1, active.col);
    if (!grounded) lockStart = null;
    else if (lockStart == null) lockStart = t;
    else if (t - lockStart >= LOCK_DELAY) lockHuman();
  }

  function frame(timestamp) {
    if (!running) return;
    const dt = lastTime === 0 ? 0 : Math.min(100, timestamp - lastTime);
    lastTime = timestamp;

    if (human) {
      humanStep(timestamp, dt);
    } else {
      accumulator += dt;
      while (accumulator >= msPerRow) {
        accumulator -= msPerRow;
        tick();
      }
    }

    renderer.draw(board, active);
    window.requestAnimationFrame(frame);
  }

  showScore();
  showNext();
  showHold();
  if (human) reset(true, true);
  window.requestAnimationFrame(frame);

  return {
    stop() {
      running = false;
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    },
    setWeights(w) {
      weights = w;
    },
    /**
     * Swap the agent mid-session. The board is wiped and the counters reset,
     * because carrying a stack built by one agent into another's run would
     * make the comparison meaningless.
     */
    setChooser(fn) {
      chooser = fn;
      if (!human) reset(true);
    },
    /** Switch between you playing (true) and the agent playing (false). New game. */
    setHuman(on) {
      human = !!on;
      phase = 'playing';
      accumulator = 0;
      reset(true, true);
      if (!human) options.onState?.('agent', {});
    },
    isHuman: () => human,
    /** For touch buttons: 'left' 'right' 'down' 'cw' 'ccw' 'drop' 'hold' 'pause' 'restart' 'start'. */
    press,
    /** Touch soft drop: true while the button is held down. */
    softDrop(on) {
      keys.down = !!on && human;
    },
    pause: setPaused,
    setSpeed(ms) {
      msPerRow = ms;
    },
    stats: () => ({ score, lines, pieces, games, bestLines }),
  };
}
