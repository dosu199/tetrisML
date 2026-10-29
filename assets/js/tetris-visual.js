/**
 * tetris-visual.js - the Play page's game: a DOM board that either you or
 * an AI player plays, with hand-over between the two in the same game.
 *
 *   you    the keyboard or the touch buttons move the piece
 *          (src/tetris-controls.js: the same controls as Play vs Bot)
 *   agent  the chooser picks a placement when a piece appears; the piece is
 *          put into that rotation and column at once and then falls, so the
 *          move it plays is exactly the move it chose
 *
 * The board is a grid of <div class="block"> elements (.row_N rows) inside
 * #gameContainer, created once. Each frame every cell gets a class: v1-v7
 * (a piece's colour), ghost (where your piece will land) or hint (where the
 * agent would put it); the look is all CSS (assets/css/style.css).
 */

import { WIDTH, HEIGHT, idx, createBoard, Bag, PIECES, placePiece, clearLines } from '../../src/tetris-core.js';
import { PieceController, TIMING, KEYMAP, LINE_POINTS, gravityMs, landingRow } from '../../src/tetris-controls.js';

const RESTART_GUARD_MS = 1000; // after a game over, Space restarts only after this long
const IGNORE_KEYS_IN = 'select, input, textarea, button'; // keys typed there are not game keys

/** Draws piece `pieceIdx` (or nothing, for null) into a .mini element as small cells. */
export function drawMiniPiece(el, pieceIdx) {
  if (!el) return;
  el.replaceChildren();
  if (pieceIdx == null) return;
  const { cells, width, height, value } = PIECES[pieceIdx].rotations[0];
  el.style.setProperty('--w', width);
  const filled = new Set(cells.map(([r, c]) => r * width + c));
  for (let i = 0; i < width * height; i++) {
    const cell = document.createElement('i');
    if (filled.has(i)) cell.className = `v${value}`;
    el.appendChild(cell);
  }
}

/* ---------------------------------------------------------------- renderer */

class Renderer {
  constructor(container) {
    this.blocks = new Array(WIDTH * HEIGHT);
    this.classes = new Array(WIDTH * HEIGHT).fill('');

    container.innerHTML = '';
    for (let r = 0; r < HEIGHT; r++) {
      const row = document.createElement('div');
      row.classList.add('row', `row_${r}`);
      for (let c = 0; c < WIDTH; c++) {
        const block = document.createElement('div');
        block.classList.add('block');
        row.appendChild(block);
        this.blocks[idx(r, c)] = block;
      }
      container.appendChild(row);
    }
  }

  /**
   * Draws the settled board, the falling piece ({ rotation, row, col }) with
   * its ghost, and the hint (cell indexes) when there is one. Layers, from
   * the bottom: board, ghost, hint, piece. Only cells that changed are touched.
   */
  draw(board, piece, hint = null) {
    const next = Array.from(board, (v) => (v ? `block v${v}` : 'block'));
    if (piece) {
      const { rotation, row, col } = piece;
      const ghostRow = landingRow(board, rotation, row, col);
      for (const [dr, dc] of rotation.cells) next[idx(ghostRow + dr, col + dc)] = 'block ghost';
    }
    for (const i of hint ?? []) next[i] = 'block hint';
    if (piece) {
      const { rotation, row, col } = piece;
      for (const [dr, dc] of rotation.cells) next[idx(row + dr, col + dc)] = `block v${rotation.value}`;
    }
    for (let i = 0; i < next.length; i++) {
      if (next[i] !== this.classes[i]) {
        this.blocks[i].className = next[i];
        this.classes[i] = next[i];
      }
    }
  }
}

/* ------------------------------------------------------------------- game */

/**
 * Starts the game loop and returns its controls.
 *
 * @param {object} options
 * @param {Function} options.chooser  the agent: (board, current, next, holdCtx) -> placement | HOLD | null
 * @param {(position: object) => Promise<object|null>} [options.askHint]
 *        where the agent would put your piece. position: { board, cur, next,
 *        held, canHold }; the answer: a placement, HOLD or null. Default: ask
 *        `chooser` here on the page.
 * @param {boolean} [options.human=false]  you play (true) or the agent (false)
 * @param {boolean} [options.hint=false]   show where the agent would put your piece
 * @param {boolean} [options.hold=false]   the hold box is in the game
 * @param {number} [options.msPerRow=40]   how fast the agent's pieces fall
 * @param {(phase: string, info: object) => void} [options.onState]
 *        'ready' (waiting for you to start), 'playing', 'paused', 'over' (you
 *        topped out; info has lines and score) or 'agent' (the agent took over)
 * @param {(hint: { hold: boolean } | null) => void} [options.onHint]
 * @param {(stats: object) => void} [options.onStats]
 *        { score, lines, pieces, games, bestLines, lastClear: { lines, points } | null }
 */
export function startVisualGame(options) {
  const {
    hold = false,
    msPerRow = 40,
    containerId = 'gameContainer',
    scoreId = 'scoreBoard',
    nextId = 'nextTetromino',
    holdId = 'holdTetromino',
  } = options;

  const container = document.getElementById(containerId);
  if (!container) throw new Error(`No #${containerId} element on the page`);
  const scoreBoard = document.getElementById(scoreId);
  const nextBoard = document.getElementById(nextId);
  const holdBoard = document.getElementById(holdId);
  const renderer = new Renderer(container);

  /* -- state -- */
  let chooser = options.chooser;
  const askHint =
    options.askHint ?? ((p) => Promise.resolve(chooser(p.board, p.cur, p.next, { held: p.held, canHold: p.canHold })));
  let human = !!options.human;
  let phase = human ? 'ready' : 'agent'; // ready | playing | paused | over (you) | agent
  let running = true;

  // over several games, so the page can show more than one lucky or unlucky run
  let games = 1;
  let bestLines = 0;

  // the current game
  let board = createBoard();
  let gen = new Bag(newSeed());
  let current = gen.next();
  let next = gen.next();
  let held = null; // the piece in the hold box
  let canHold = true; // once per piece
  let score = 0;
  let lines = 0;
  let pieces = 0;
  let lastClear = null; // { lines, points } of the latest piece that cleared lines

  // the agent's falling piece: { rotation, row, col, target }
  let agentPiece = null;
  let agentClock = 0; // ms not yet spent on falling rows

  // your falling piece
  const controller = new PieceController({
    board: () => board,
    piece: () => current,
    lock: (r, row, col) => {
      lockPiece(PIECES[current].rotations[r], row, col);
      spawnYourPiece();
    },
    hold: holdYourPiece,
    onSoftDropRow: () => {
      score += 1; // a point per row of soft drop
      showScore();
    },
  });

  // hint: where the agent would put your current piece
  let hintOn = !!options.hint;
  let hintCells = null; // cell indexes to outline, or null
  let hintTimer = null;
  let pieceSeq = 0; // bumps on every new piece, so a hint that arrives late is dropped
  let overAt = 0;
  let lastTime = 0;

  function newSeed() {
    return (Date.now() & 0xffff) + games;
  }

  function setPhase(p, info = {}) {
    phase = p;
    options.onState?.(p, { lines, score, pieces, ...info });
  }

  /* -- display -- */

  let shownNext;
  let shownHeld;
  function showNext() {
    if (next === shownNext) return;
    shownNext = next;
    drawMiniPiece(nextBoard?.querySelector('.mini'), next);
  }

  function showHold() {
    if (held === shownHeld) return;
    shownHeld = held;
    drawMiniPiece(holdBoard?.querySelector('.mini'), held);
  }

  function showScore() {
    if (scoreBoard) scoreBoard.textContent = score.toLocaleString('en-US');
    options.onStats?.({ score, lines, pieces, games, bestLines, lastClear });
  }

  /** The falling piece to draw: yours or the agent's. */
  function fallingPiece() {
    if (!human) return agentPiece;
    const a = controller.active;
    return a && { rotation: controller.rotation, row: a.row, col: a.col };
  }

  /* -- rules shared by you and the agent -- */

  function swapHold() {
    if (held == null) {
      held = current;
      current = next;
      next = gen.next();
    } else {
      [held, current] = [current, held];
    }
    canHold = false;
    showHold();
    showNext();
  }

  /** Puts a piece into the board for good, clears lines, scores, and moves on to the next piece. */
  function lockPiece(rotation, row, col) {
    placePiece(board, rotation, row, col);
    pieces++;
    const cleared = clearLines(board);
    if (cleared > 0) {
      lines += cleared;
      score += LINE_POINTS[cleared];
      lastClear = { lines: cleared, points: LINE_POINTS[cleared] };
    }
    showScore();
    current = next;
    next = gen.next();
    canHold = true;
  }

  function recordBest() {
    if (lines > bestLines) bestLines = lines;
  }

  /** A new game. keepStats: count the game that ended; waitForStart: you press a key before it begins. */
  function reset({ keepStats = true, waitForStart = false } = {}) {
    if (keepStats) {
      if (phase !== 'over') recordBest(); // a game you lost was already counted
      games++;
    } else {
      games = 1;
      bestLines = 0;
    }
    board = createBoard();
    gen = new Bag(newSeed());
    current = gen.next();
    next = gen.next();
    score = 0;
    lines = 0;
    pieces = 0;
    lastClear = null;
    held = null;
    canHold = true;
    agentPiece = null;
    controller.clear();
    controller.releaseAll();
    showHold();
    showScore();
    if (human) {
      spawnYourPiece();
      setPhase(waitForStart ? 'ready' : 'playing');
    }
  }

  /* -- the agent -- */

  /** The agent decides where the current piece goes. False if it has no move: game over. */
  function spawnAgentPiece() {
    let move = chooser(board, current, next, hold ? { held, canHold } : undefined);
    if (hold && canHold && move?.hold) {
      // the current piece goes into the hold box; the piece it gets must be placed
      swapHold();
      move = chooser(board, current, next, { held, canHold: false });
    }
    if (!move || move.hold) return false;
    agentPiece = { rotation: PIECES[current].rotations[move.rotation], row: 0, col: move.col, target: move.row };
    showNext();
    return true;
  }

  /** One row of the agent's game: a new piece, one row of falling, or landing. */
  function agentTick() {
    if (!agentPiece) {
      if (!spawnAgentPiece()) reset();
    } else if (agentPiece.row < agentPiece.target) {
      agentPiece.row++;
    } else {
      lockPiece(agentPiece.rotation, agentPiece.row, agentPiece.col);
      agentPiece = null;
    }
  }

  /* -- you -- */

  /** Your next piece appears at the top. False if it does not fit: game over. */
  function spawnYourPiece() {
    showNext();
    pieceSeq++;
    const fits = controller.spawn();
    updateHint();
    if (!fits) {
      recordBest();
      overAt = performance.now();
      setPhase('over');
    }
    return fits;
  }

  function holdYourPiece() {
    if (!hold || !canHold) return;
    swapHold();
    spawnYourPiece();
  }

  function setPaused(on) {
    if (!human) return;
    if (on && phase === 'playing') {
      controller.releaseAll();
      setPhase('paused');
    } else if (!on && phase === 'paused') {
      lastTime = 0;
      setPhase('playing');
    }
  }

  /** One input from the keyboard or a touch button. */
  function press(action) {
    if (!human || !running) return;
    if (phase === 'ready') {
      if (action !== 'pause') {
        // 'start', or any game key
        lastTime = 0;
        setPhase('playing');
      }
      return;
    }
    if (action === 'pause') return setPaused(phase === 'playing');
    if (phase === 'over') {
      // Space restarts too, but not at once, so a hard drop that was already
      // on its way does not skip the result
      if (action === 'restart' || (action === 'drop' && performance.now() - overAt > RESTART_GUARD_MS)) reset();
      return;
    }
    if (phase === 'playing') controller.run(action);
  }

  /**
   * Asks the agent where it would put your current piece, just after the
   * piece appears; only while you play with the hint on.
   */
  function updateHint() {
    hintCells = null;
    clearTimeout(hintTimer);
    if (!human || !hintOn || !controller.active) {
      options.onHint?.(null);
      return;
    }
    const seq = pieceSeq;
    const position = { board, cur: current, next, held, canHold: hold && canHold };
    const stillWanted = () => running && seq === pieceSeq && human && hintOn;
    // on the next turn of the event loop, so the new piece is drawn first
    hintTimer = setTimeout(async () => {
      if (!stillWanted()) return;
      const m = await askHint(position);
      if (!stillWanted()) return;
      if (m?.hold) {
        options.onHint?.({ hold: true });
      } else if (m) {
        const rotation = PIECES[current].rotations[m.rotation];
        hintCells = rotation.cells.map(([dr, dc]) => idx(m.row + dr, m.col + dc));
        options.onHint?.({ hold: false });
      } else {
        options.onHint?.(null);
      }
    }, 0);
  }

  /* -- input -- */

  function onKeyDown(e) {
    if (!human || !running) return;
    if (e.target.closest?.(IGNORE_KEYS_IN)) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return; // browser shortcuts: Ctrl+C is not hold
    if (e.code === 'KeyP' || e.code === 'Escape') {
      setPaused(phase === 'playing');
      return;
    }
    if (e.code === 'Enter' && phase === 'over') {
      reset();
      return;
    }
    const action = KEYMAP[e.code];
    if (phase === 'ready' && (action || e.code === 'Enter')) {
      e.preventDefault();
      press('start');
      return;
    }
    if (!action) return;
    e.preventDefault(); // no page scrolling while playing
    if (e.repeat) return; // the controller repeats left/right itself
    if (phase === 'playing') controller.keyDown(action);
    else press(action);
  }

  function onKeyUp(e) {
    const action = KEYMAP[e.code];
    if (action) controller.keyUp(action);
  }

  // a key released while the window is in the background never sends keyup
  const releaseKeys = () => controller.releaseAll();

  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', releaseKeys);

  /* -- the loop -- */

  function frame(now) {
    if (!running) return;
    const dt = lastTime === 0 ? 0 : Math.min(TIMING.maxFrame, now - lastTime);
    lastTime = now;

    if (human) {
      if (phase === 'playing') controller.step(now, dt, gravityMs(lines));
    } else {
      agentClock += dt;
      while (agentClock >= msPerRow) {
        agentClock -= msPerRow;
        agentTick();
      }
    }

    renderer.draw(board, fallingPiece(), human && hintOn ? hintCells : null);
    window.requestAnimationFrame(frame);
  }

  showScore();
  showNext();
  showHold();
  if (human) reset({ keepStats: false, waitForStart: true });
  window.requestAnimationFrame(frame);

  return {
    stop() {
      running = false;
      clearTimeout(hintTimer);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', releaseKeys);
    },

    /** Swap the agent. The game goes on: the new agent takes over the board as it is, from the next piece. */
    setChooser(fn) {
      chooser = fn;
      if (human) {
        pieceSeq++;
        updateHint();
      }
    },

    /** Show (true) or hide (false) where the agent would put your piece. */
    setHint(on) {
      hintOn = !!on;
      pieceSeq++;
      updateHint();
    },

    /**
     * Hand the same game over between you (true) and the agent (false):
     * board, score, lines, hold and the falling piece all carry over.
     */
    setHuman(on) {
      on = !!on;
      if (on === human) return;
      agentClock = 0;
      controller.releaseAll();
      human = on;
      if (on) {
        lastTime = 0;
        if (agentPiece) {
          // take over the agent's falling piece where it is
          const r = Math.max(0, PIECES[current].rotations.indexOf(agentPiece.rotation));
          controller.adopt(r, agentPiece.row, agentPiece.col);
          agentPiece = null;
          pieceSeq++;
          updateHint();
          setPhase('playing');
        } else if (spawnYourPiece()) {
          setPhase('playing');
        }
      } else {
        if (phase === 'over') {
          reset(); // your stack reached the top: the agent starts a new game
        } else {
          controller.clear(); // the agent places the current piece again, from the top
        }
        hintCells = null;
        clearTimeout(hintTimer);
        options.onHint?.(null);
        setPhase('agent');
      }
      showScore();
    },

    isHuman: () => human,

    /** A new game (the one being played is counted), for you or the agent. */
    restart() {
      if (running) reset();
    },

    /** For touch buttons and the overlay: 'left' 'right' 'cw' 'ccw' 'drop' 'hold' 'pause' 'restart' 'start'. */
    press,

    /** Touch soft drop: true while the button is held down. */
    softDrop(on) {
      controller.setSoftDrop(on && human);
    },

    pause: setPaused,
  };
}
