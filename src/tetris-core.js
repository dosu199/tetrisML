/**
 * tetris-core.js - the game rules. No DOM, no timers, no global state, so the
 * same code runs in Node (training, benchmarks, tests) and in the browser.
 *
 * The board is a 16 x 27 grid: a 10 x 24 playfield inside a 3-cell frame down
 * the left, down the right and across the bottom. Because the frame counts as
 * filled, collision checks never need bounds tests.
 *
 *        col 0 1 2 | 3 ......... 12 | 13 14 15
 *   row 0        # # #  playfield    #  #  #
 *   ...
 *   row 23       # # #  playfield    #  #  #
 *   row 24-26    # # # # # # # # # # #  #  #   <- floor
 *
 * The board is one flat Uint8Array; cell (r, c) lives at index r * WIDTH + c.
 * Copying the board is the hottest operation in the search (thousands of
 * copies per piece), and `slice()` on a typed array is a single memcpy.
 *
 * Contents, top to bottom: board, pieces, randomness, mechanics, garbage,
 * hold, and playEpisode (one full game with any chooser).
 */

/* ------------------------------------------------------------------ board */

export const WIDTH = 16;
export const HEIGHT = 27;

/** Cell value of the frame. Garbage rows use it too, so they render grey. */
export const FRAME = 8;

export const PLAY_LEFT = 3; // first playfield column
export const PLAY_RIGHT = 12; // last playfield column, inclusive
export const PLAY_COLS = PLAY_RIGHT - PLAY_LEFT + 1; // 10
export const PLAY_BOTTOM = HEIGHT - 4; // 23, last playfield row, inclusive
export const PLAY_ROWS = PLAY_BOTTOM + 1; // 24

/** Index of cell (row, col) in the flat board. */
export const idx = (r, c) => r * WIDTH + c;

/** An empty playfield inside its frame. */
export function createBoard() {
  const board = new Uint8Array(WIDTH * HEIGHT);
  for (let r = 0; r < HEIGHT; r++) {
    for (let c = 0; c < PLAY_LEFT; c++) board[idx(r, c)] = FRAME; // left wall
    for (let c = PLAY_RIGHT + 1; c < WIDTH; c++) board[idx(r, c)] = FRAME; // right wall
  }
  for (let r = PLAY_BOTTOM + 1; r < HEIGHT; r++) {
    for (let c = 0; c < WIDTH; c++) board[idx(r, c)] = FRAME; // floor
  }
  return board;
}

export const copyBoard = (board) => board.slice();

/* ----------------------------------------------------------------- pieces */

// Cell values double as colour/sprite ids: Z=1, I=2, O=3, S=4, L=5, J=6, T=7.
const SHAPES = {
  Z: {
    value: 1,
    grids: [
      [[1, 1, 0, 0],
       [0, 1, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[0, 0, 1, 0],
       [0, 1, 1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 0]],
    ],
  },
  I: {
    value: 2,
    grids: [
      [[1, 1, 1, 1],
       [0, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[1, 0, 0, 0],
       [1, 0, 0, 0],
       [1, 0, 0, 0],
       [1, 0, 0, 0]],
    ],
  },
  O: {
    value: 3,
    grids: [
      [[1, 1, 0, 0],
       [1, 1, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
    ],
  },
  S: {
    value: 4,
    grids: [
      [[0, 1, 1, 0],
       [1, 1, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 0]],
    ],
  },
  L: {
    value: 5,
    grids: [
      [[0, 0, 1, 0],
       [1, 1, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[1, 0, 0, 0],
       [1, 0, 0, 0],
       [1, 1, 0, 0],
       [0, 0, 0, 0]],
      [[1, 1, 1, 0],
       [1, 0, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[1, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 0]],
    ],
  },
  J: {
    value: 6,
    grids: [
      [[1, 0, 0, 0],
       [1, 1, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[1, 1, 0, 0],
       [1, 0, 0, 0],
       [1, 0, 0, 0],
       [0, 0, 0, 0]],
      [[1, 1, 1, 0],
       [0, 0, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[0, 1, 0, 0],
       [0, 1, 0, 0],
       [1, 1, 0, 0],
       [0, 0, 0, 0]],
    ],
  },
  T: {
    value: 7,
    grids: [
      [[0, 1, 0, 0],
       [1, 1, 1, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [1, 0, 0, 0],
       [0, 0, 0, 0]],
      [[1, 1, 1, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 0],
       [0, 0, 0, 0]],
      [[0, 1, 0, 0],
       [1, 1, 0, 0],
       [0, 1, 0, 0],
       [0, 0, 0, 0]],
    ],
  },
};

/**
 * Turns a 4x4 grid into a normalised rotation: the four occupied cells,
 * shifted so the topmost row and the leftmost column are 0. An anchor of
 * (row 0, col PLAY_LEFT) then always sits flush in the top-left corner of the
 * playfield, so the legal columns are simply PLAY_LEFT .. PLAY_RIGHT - width + 1.
 */
function makeRotation(grid, value) {
  const raw = [];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) if (grid[r][c]) raw.push([r, c]);
  }
  const minRow = Math.min(...raw.map((p) => p[0]));
  const minCol = Math.min(...raw.map((p) => p[1]));
  const cells = raw.map(([r, c]) => [r - minRow, c - minCol]);
  const height = Math.max(...cells.map((p) => p[0])) + 1;
  const width = Math.max(...cells.map((p) => p[1])) + 1;
  return { cells, width, height, value };
}

/** The 7 pieces, in the order Z I O S L J T; a piece is referred to by its index. */
export const PIECES = Object.entries(SHAPES).map(([name, { value, grids }]) => ({
  name,
  value,
  rotations: grids.map((g) => makeRotation(g, value)),
}));

export const PIECE_INDEX = Object.fromEntries(PIECES.map((p, i) => [p.name, i]));

/* ------------------------------------------------------------- randomness */

/** Small, fast, seedable PRNG (returns floats in [0, 1)). Same seed, same game. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The 7-bag randomiser of modern Tetris: shuffle all seven pieces, deal them
 * out, repeat. Never more than 12 pieces between two of the same kind.
 */
export class Bag {
  constructor(seed) {
    this.rand = mulberry32(seed);
    this.queue = [];
  }
  refill() {
    const bag = [0, 1, 2, 3, 4, 5, 6];
    for (let i = bag.length - 1; i > 0; i--) {
      const j = Math.floor(this.rand() * (i + 1));
      [bag[i], bag[j]] = [bag[j], bag[i]];
    }
    this.queue.push(...bag);
  }
  next() {
    if (this.queue.length === 0) this.refill();
    return this.queue.shift();
  }
}

/**
 * The classic (NES-style) randomiser: every piece drawn uniformly and
 * independently. No guarantees: four S pieces in a row, or 30 pieces without
 * an I, are both possible. Harder than the 7-bag.
 */
export class MemorylessGen {
  constructor(seed) {
    this.rand = mulberry32(seed);
  }
  next() {
    return Math.floor(this.rand() * PIECES.length);
  }
}

/** 'bag' (7-bag) or 'memoryless': a piece generator with .next(). */
export function createPieceGenerator(randomizer, seed) {
  if (randomizer === 'bag') return new Bag(seed);
  if (randomizer === 'memoryless') return new MemorylessGen(seed);
  throw new Error(`unknown randomizer: ${randomizer} (use 'bag' or 'memoryless')`);
}

/**
 * Seed of the random stream that places garbage holes in the game with this
 * seed. Separate from the piece stream, so garbage never changes the pieces.
 */
export const garbageSeed = (seed) => seed * 7 + 12345;

/* -------------------------------------------------------------- mechanics */

/** True if this rotation overlaps anything (stack or frame) at (row, col). */
export function collides(board, rotation, row, col) {
  const cells = rotation.cells;
  for (let i = 0; i < cells.length; i++) {
    if (board[idx(row + cells[i][0], col + cells[i][1])] !== 0) return true;
  }
  return false;
}

/**
 * Lowest row this rotation reaches in this column, dropping straight down
 * from the top. -1 if it already collides at the top: no legal placement.
 */
export function dropRow(board, rotation, col) {
  if (collides(board, rotation, 0, col)) return -1;
  let row = 0;
  while (!collides(board, rotation, row + 1, col)) row++;
  return row;
}

/** Stamps the piece into the board. Mutates. */
export function placePiece(board, rotation, row, col) {
  const cells = rotation.cells;
  for (let i = 0; i < cells.length; i++) {
    board[idx(row + cells[i][0], col + cells[i][1])] = rotation.value;
  }
}

/**
 * Removes every full row and drops what is above it. Mutates; returns the
 * number of rows cleared. One bottom-up pass: a read pointer scans every row
 * and a write pointer copies down each row that is not full.
 */
export function clearLines(board) {
  let write = PLAY_BOTTOM;
  let cleared = 0;

  for (let read = PLAY_BOTTOM; read >= 0; read--) {
    let full = true;
    for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
      if (board[idx(read, c)] === 0) {
        full = false;
        break;
      }
    }
    if (full) {
      cleared++;
      continue;
    }
    if (write !== read) {
      for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
        board[idx(write, c)] = board[idx(read, c)];
      }
    }
    write--;
  }
  for (; write >= 0; write--) {
    for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) board[idx(write, c)] = 0; // blank the top
  }
  return cleared;
}

/**
 * Every placement reachable by "rotate, slide, drop": the standard action
 * space of a Tetris AI. Returns [{ rotation, col, row }, ...], rotation by
 * rotation, columns left to right. Search code relies on this order: on equal
 * scores the first placement wins.
 */
export function placements(board, pieceIdx) {
  const out = [];
  const rotations = PIECES[pieceIdx].rotations;
  for (let r = 0; r < rotations.length; r++) {
    const rot = rotations[r];
    const lastCol = PLAY_RIGHT - rot.width + 1;
    for (let col = PLAY_LEFT; col <= lastCol; col++) {
      const row = dropRow(board, rot, col);
      if (row >= 0) out.push({ rotation: r, col, row });
    }
  }
  return out;
}

/** Plays `move` ({ rotation, col, row }) with this piece on `board`. Mutates; returns the lines cleared. */
export function playMove(board, pieceIdx, move) {
  placePiece(board, PIECES[pieceIdx].rotations[move.rotation], move.row, move.col);
  return clearLines(board);
}

/** Column heights, measured from the floor. Empty column = 0. */
export function columnHeights(board, out = new Int32Array(PLAY_COLS)) {
  for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
    let h = 0;
    for (let r = 0; r <= PLAY_BOTTOM; r++) {
      if (board[idx(r, c)] !== 0) {
        h = PLAY_BOTTOM - r + 1;
        break;
      }
    }
    out[c - PLAY_LEFT] = h;
  }
  return out;
}

/* ---------------------------------------------------------------- garbage */

/**
 * Pushes a garbage row in from the bottom: a full row with one hole, the way
 * multiplayer Tetris punishes you for your opponent's clears. Everything
 * moves up one row. `holeCol` counts playfield columns, 0..9.
 *
 * Why it exists: in a normal 7-bag game both the search bot and the network
 * survive 5,000+ pieces every time, so they cannot be told apart. Garbage
 * makes every game end eventually, so how long you last becomes measurable.
 *
 * The row is stored as FRAME: it counts as filled everywhere (collision,
 * features, the network's inputs). Returns false if a block was already in
 * the top row: pushing it off the board is a top-out.
 */
export function addGarbageRow(board, holeCol) {
  let overflow = false;
  for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) if (board[idx(0, c)] !== 0) overflow = true;
  for (let r = 0; r < PLAY_BOTTOM; r++) {
    for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) board[idx(r, c)] = board[idx(r + 1, c)];
  }
  for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
    board[idx(PLAY_BOTTOM, c)] = c - PLAY_LEFT === holeCol ? 0 : FRAME;
  }
  return !overflow;
}

/* ------------------------------------------------------------------- hold */

/**
 * HOLD (modern Tetris rule): instead of placing the current piece, put it in
 * the hold box. If the box was empty the next piece comes in, otherwise the
 * held piece comes back. Once per piece: after a hold, the player must place
 * what it got.
 *
 * A chooser that supports hold gets a 4th argument { held, canHold } and may
 * return HOLD instead of a placement. Choosers that ignore the 4th argument
 * simply never hold.
 */
export const HOLD = Object.freeze({ hold: true });

/* ------------------------------------------------------------ one episode */

/**
 * One full game with any chooser, under a chosen rule set. Training, the
 * benchmarks and the tests all use this, so they all play the same game.
 *
 *   chooser(board, current, next, holdCtx) -> { rotation, col, row } | HOLD | null
 *     next     the next piece, or null when preview is off
 *     holdCtx  { held, canHold } when hold is on, otherwise undefined
 *     null     no move: the game is lost
 *
 * @param {Function} chooser
 * @param {number} seed  fixes the pieces and the garbage holes
 * @param {object} [rules]
 * @param {number} [rules.cap=500]           stop after this many pieces
 * @param {number} [rules.garbageEvery=0]    0 = off; N = a garbage row every N pieces
 * @param {'bag'|'memoryless'} [rules.randomizer='bag']
 * @param {boolean} [rules.preview=true]     whether the chooser sees the next piece
 * @param {boolean} [rules.hold=false]       whether the chooser may hold
 * @returns {{ lines: number, pieces: number, died: boolean, holds: number, tetrises: number }}
 */
export function playEpisode(
  chooser,
  seed,
  { cap = 500, garbageEvery = 0, randomizer = 'bag', preview = true, hold = false } = {}
) {
  const board = createBoard();
  const gen = createPieceGenerator(randomizer, seed);
  const holeRand = mulberry32(garbageSeed(seed));
  let current = gen.next();
  let next = gen.next();
  let held = null;
  let lines = 0;
  let pieces = 0;
  let holds = 0;
  let tetrises = 0;
  let died = false;

  while (pieces < cap) {
    let move = chooser(board, current, preview ? next : null, hold ? { held, canHold: true } : undefined);
    if (hold && move && move.hold) {
      holds++;
      if (held == null) {
        held = current;
        current = next;
        next = gen.next();
      } else {
        [held, current] = [current, held];
      }
      move = chooser(board, current, preview ? next : null, { held, canHold: false });
    }
    if (move == null || move.hold) {
      died = true;
      break;
    }
    const cleared = playMove(board, current, move);
    lines += cleared;
    if (cleared === 4) tetrises++;
    pieces++;
    if (garbageEvery > 0 && pieces % garbageEvery === 0) {
      if (!addGarbageRow(board, Math.floor(holeRand() * PLAY_COLS))) {
        died = true;
        break;
      }
    }
    current = next;
    next = gen.next();
  }
  return { lines, pieces, died, holds, tetrises };
}
