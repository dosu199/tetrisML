/**
 * tetris-core.js - pure game logic. No DOM, no timers, no globals.
 *
 * Board layout is unchanged from your original: a 16x27 grid with a 3-cell
 * frame down the left, down the right, and across the bottom. The frame is
 * why collision checks need no bounds tests.
 *
 *        col 0 1 2 | 3 ......... 12 | 13 14 15
 *   row 0        # # #  playfield    #  #  #
 *   ...
 *   row 23       # # #  playfield    #  #  #
 *   row 24-26    # # # # # # # # # # #  #  #   <- floor
 *
 * The board is a flat Uint8Array instead of an array of arrays. Cell (r, c)
 * lives at index r * WIDTH + c. This matters: copying the board is the single
 * hottest operation in the AI search (thousands of copies per piece) and
 * `board.slice()` on a typed array is roughly 30x faster than the nested
 * loops in the old copyBoard().
 */

export const WIDTH = 16;
export const HEIGHT = 27;

export const FRAME = 8; // the value your original used for wall/floor cells

export const PLAY_LEFT = 3; // first playfield column
export const PLAY_RIGHT = 12; // last playfield column, inclusive
export const PLAY_COLS = PLAY_RIGHT - PLAY_LEFT + 1; // 10
export const PLAY_BOTTOM = HEIGHT - 4; // 23, last playfield row, inclusive
export const PLAY_ROWS = PLAY_BOTTOM + 1; // 24

export const idx = (r, c) => r * WIDTH + c;

/* ------------------------------------------------------------------ board */

export function createBoard() {
  const board = new Uint8Array(WIDTH * HEIGHT);
  // left and right walls
  for (let r = 0; r < HEIGHT; r++) {
    for (let c = 0; c < PLAY_LEFT; c++) board[idx(r, c)] = FRAME;
    for (let c = PLAY_RIGHT + 1; c < WIDTH; c++) board[idx(r, c)] = FRAME;
  }
  // floor
  for (let r = PLAY_BOTTOM + 1; r < HEIGHT; r++) {
    for (let c = 0; c < WIDTH; c++) board[idx(r, c)] = FRAME;
  }
  return board;
}

/** Fast copy. Replaces the old copyBoard(). */
export const copyBoard = (board) => board.slice();

/* ------------------------------------------------------------------ pieces */

// Colour values kept identical to your original so the images still map:
// Z=1, I=2, O=3, S=4, L=5, J=6, T=7
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
 * Turns a 4x4 grid into a normalised rotation: a list of the four occupied
 * cells, shifted so the topmost row is 0 and the leftmost column is 0.
 * Normalising means an anchor of (row 0, col PLAY_LEFT) always sits flush in
 * the top-left of the playfield, which makes the column loop trivial and
 * removes the empty-padding-column bugs the original 4x4 layout invited.
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

  // 4x4 grid in normalised position, for drawing the "next piece" preview
  const preview = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  for (const [r, c] of cells) preview[r][c] = value;

  return { cells, width, height, value, preview };
}

export const PIECES = Object.entries(SHAPES).map(([name, { value, grids }]) => ({
  name,
  value,
  rotations: grids.map((g) => makeRotation(g, value)),
}));

export const PIECE_INDEX = Object.fromEntries(PIECES.map((p, i) => [p.name, i]));

/* --------------------------------------------------------------- randomness */

/** Small, fast, seedable PRNG. Same seed always gives the same game. */
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
 * Standard 7-bag randomiser: shuffle all seven pieces, deal them out, repeat.
 * This replaces the 5-array sack in the original, which could deal the same
 * piece many times in a row. 7-bag is what modern Tetris uses and it makes
 * fitness comparisons between weight vectors much less noisy.
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

/* --------------------------------------------------------------- mechanics */

/** True if this rotation overlaps anything (stack or frame) at (row, col). */
export function collides(board, rotation, row, col) {
  const cells = rotation.cells;
  for (let i = 0; i < cells.length; i++) {
    if (board[idx(row + cells[i][0], col + cells[i][1])] !== 0) return true;
  }
  return false;
}

/**
 * Lowest row this rotation can reach in this column, dropping straight down
 * from the top. Returns -1 if the column is blocked at spawn height, which is
 * how a topped-out board reports "no legal placement here".
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
 * Removes every full row and drops what's above it down. Mutates, returns the
 * number of rows cleared.
 *
 * The original scanned with a `counter` that was never reset per row and
 * mutated the loop variable while iterating; this is a straightforward
 * read-pointer/write-pointer compaction instead.
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
  // blank out whatever is left at the top
  for (; write >= 0; write--) {
    for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) board[idx(write, c)] = 0;
  }
  return cleared;
}

/**
 * Every placement reachable by "rotate, slide, drop" - the standard Tetris AI
 * action space. Returns [{ rotation, col, row }, ...].
 *
 * The original walked the piece to the right wall and then stepped left,
 * carrying `x` over from the previous column so later columns started their
 * drop buried inside the stack and were silently skipped. This enumerates
 * columns directly, so every legal placement is always considered.
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
