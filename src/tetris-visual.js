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
 */

import {
  WIDTH,
  HEIGHT,
  PLAY_LEFT,
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

export function startVisualGame(options = {}) {
  let weights = options.weights || PUBLISHED_WEIGHTS;
  const {
    seed = Date.now() & 0xffff,
    msPerRow = 40,
    lookahead = true,
    imagePath = 'images/',
    containerId = 'gameContainer',
    scoreId = 'scoreBoard',
    nextId = 'nextTetromino',
  } = options;

  const container = document.getElementById(containerId);
  if (!container) throw new Error(`No #${containerId} element on the page`);
  const scoreBoard = document.getElementById(scoreId);
  const nextBoard = document.getElementById(nextId);

  const renderer = new Renderer(container, { imagePath });

  let board = createBoard();
  let bag = new Bag(seed);
  let current = bag.next();
  let next = bag.next();
  let score = 0;
  let lines = 0;

  let active = null; // { rotation, row, col }
  let accumulator = 0;
  let lastTime = 0;
  let running = true;

  function showNext() {
    if (nextBoard) nextBoard.style.backgroundImage = `url(${imagePath}${NEXT_IMAGES[next]}.png)`;
  }

  function showScore() {
    if (scoreBoard) scoreBoard.innerHTML = String(score);
  }

  function spawn() {
    const move = chooseMove(board, current, next, weights, { lookahead });
    if (move === null || move.score === -Infinity) return false;
    active = {
      rotation: PIECES[current].rotations[move.rotation],
      row: 0,
      col: move.col,
      target: move.row,
    };
    return true;
  }

  function reset() {
    board = createBoard();
    bag = new Bag((Date.now() & 0xffff) + 1);
    current = bag.next();
    next = bag.next();
    score = 0;
    lines = 0;
    active = null;
    showScore();
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

    // landed
    placePiece(board, active.rotation, active.row, active.col);
    const cleared = clearLines(board);
    if (cleared > 0) {
      lines += cleared;
      // classic scoring: a quadruple is worth far more than four singles
      score += [0, 40, 100, 300, 1200][cleared];
      showScore();
    }
    active = null;
    current = next;
    next = bag.next();
  }

  function frame(timestamp) {
    if (!running) return;
    if (lastTime === 0) lastTime = timestamp;
    accumulator += timestamp - lastTime;
    lastTime = timestamp;

    while (accumulator >= msPerRow) {
      accumulator -= msPerRow;
      tick();
    }

    renderer.draw(board, active);
    window.requestAnimationFrame(frame);
  }

  showScore();
  showNext();
  window.requestAnimationFrame(frame);

  return {
    stop() {
      running = false;
    },
    setWeights(w) {
      weights = w;
    },
    stats: () => ({ score, lines }),
  };
}
