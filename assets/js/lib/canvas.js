/**
 * canvas.js - drawing helpers shared by the pages that draw boards on
 * <canvas> (Play vs Bot, How it works, Watch it learn). The look matches the
 * Play page's board: every cell a rounded slot, pieces as rounded blocks with
 * a darker bottom edge.
 */

import { PLAY_LEFT, PLAY_COLS, PLAY_ROWS, idx } from '../../../src/tetris-core.js';
import { PIECE_COLORS, GARBAGE_COLOR } from './piece-style.js';

/** A CSS custom property of the page, e.g. cssVar('--ui-accent'). */
export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/**
 * Sizes a canvas for `w` x `h` CSS pixels, sharp on high-density screens
 * (up to 3x), and returns its 2D context, scaled so you draw in CSS pixels.
 */
export function setupCanvas(canvas, w, h) {
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.aspectRatio = `${w} / ${h}`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/** The colour of a cell value: a piece's colour, or grey for garbage and the frame. */
export const blockColor = (value) => PIECE_COLORS[value] ?? GARBAGE_COLOR;

/** The rounded square of the cell at (x, y), in cells of size `s`, as a path. */
function cellPath(ctx, x, y, s) {
  const inset = Math.max(1, s * 0.06);
  ctx.beginPath();
  ctx.roundRect(x * s + inset, y * s + inset, s - 2 * inset, s - 2 * inset, s * 0.25);
}

/** A block of `color` at cell (x, y), with a darker bottom edge so it reads as a block. */
export function drawBlock(ctx, x, y, s, color, alpha = 1) {
  ctx.save();
  ctx.globalAlpha = alpha;
  cellPath(ctx, x, y, s);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.clip();
  ctx.fillStyle = 'rgba(0, 0, 0, 0.22)';
  ctx.fillRect(x * s, y * s + s * 0.82, s, s);
  ctx.restore();
}

/** Outlines of cells (x, y): where a piece would land (ghost), or an option the network considers. */
export function outlineCells(ctx, cells, s, stroke, width = 2, fill = null, dash = []) {
  ctx.save();
  ctx.setLineDash(dash);
  for (const [x, y] of cells) {
    cellPath(ctx, x, y, s);
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fill();
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Draws the playfield of `board`: the well, an empty slot per cell, and a
 * block for every filled cell.
 *
 * @param {object} opts
 * @param {number} opts.cell          cell size in CSS pixels
 * @param {number} [opts.firstRow=0]  draw rows firstRow..23 (to show only the bottom of the board)
 */
export function drawPlayfield(ctx, board, { cell: s, firstRow = 0 }) {
  const rows = PLAY_ROWS - firstRow;
  const { bg, slot } = boardColors();
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, PLAY_COLS * s, rows * s);
  ctx.fillStyle = slot;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < PLAY_COLS; c++) {
      cellPath(ctx, c, r, s);
      ctx.fill();
    }
  }
  for (let r = firstRow; r < PLAY_ROWS; r++) {
    for (let c = 0; c < PLAY_COLS; c++) {
      const v = board[idx(r, c + PLAY_LEFT)];
      if (v) drawBlock(ctx, c, r - firstRow, s, blockColor(v));
    }
  }
}

let cachedBoardColors = null;

/**
 * The board colours from the stylesheet (--ui-board, --ui-slot). Read once
 * (the theme never changes while a page is open), but only cached once the
 * stylesheet has provided them.
 */
export function boardColors() {
  if (cachedBoardColors) return cachedBoardColors;
  const bg = cssVar('--ui-board');
  const slot = cssVar('--ui-slot');
  if (bg && slot) cachedBoardColors = { bg, slot };
  return { bg: bg || '#18161c', slot: slot || '#221f27' };
}
