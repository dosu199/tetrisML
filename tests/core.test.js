/**
 * core.test.js - the game rules (src/tetris-core.js) and the bot's features
 * and search (src/tetris-ai.js).   npm test   or   node --test tests/
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  WIDTH,
  HEIGHT,
  PLAY_LEFT,
  PLAY_RIGHT,
  PLAY_BOTTOM,
  FRAME,
  idx,
  createBoard,
  copyBoard,
  clearLines,
  placements,
  dropRow,
  columnHeights,
  addGarbageRow,
  createPieceGenerator,
  playEpisode,
  PIECES,
  PIECE_INDEX,
  Bag,
} from '../src/tetris-core.js';
import { extractFeatures, chooseMove, PUBLISHED_WEIGHTS, FEATURE } from '../src/tetris-ai.js';

function fillRow(board, row, from = PLAY_LEFT, to = PLAY_RIGHT, value = 1) {
  for (let c = from; c <= to; c++) board[idx(row, c)] = value;
}

const botChooser = (weights, lookahead) => (board, current, next) => chooseMove(board, current, next, weights, { lookahead });

test('board: a 10 x 24 playfield inside a frame', () => {
  const b = createBoard();
  assert.equal(PLAY_RIGHT - PLAY_LEFT + 1, 10);
  assert.equal(PLAY_BOTTOM + 1, 24);
  assert.equal(b[idx(0, 0)], FRAME, 'left wall');
  assert.equal(b[idx(0, WIDTH - 1)], FRAME, 'right wall');
  assert.equal(b[idx(HEIGHT - 1, PLAY_LEFT)], FRAME, 'floor');
  assert.equal(b[idx(0, PLAY_LEFT)], 0, 'playfield starts empty');
});

test('line clears', async (t) => {
  await t.test('a full row clears and leaves the row empty', () => {
    const b = createBoard();
    fillRow(b, PLAY_BOTTOM);
    assert.equal(clearLines(b), 1);
    assert.equal(b[idx(PLAY_BOTTOM, PLAY_LEFT)], 0);
  });
  await t.test('blocks above a cleared row fall', () => {
    const b = createBoard();
    fillRow(b, PLAY_BOTTOM);
    b[idx(PLAY_BOTTOM - 1, PLAY_LEFT + 4)] = 7;
    clearLines(b);
    assert.equal(b[idx(PLAY_BOTTOM, PLAY_LEFT + 4)], 7);
    assert.equal(b[idx(PLAY_BOTTOM - 1, PLAY_LEFT + 4)], 0);
  });
  await t.test('a tetris clears four rows', () => {
    const b = createBoard();
    for (let k = 0; k < 4; k++) fillRow(b, PLAY_BOTTOM - k);
    assert.equal(clearLines(b), 4);
  });
  await t.test('an incomplete row does not clear', () => {
    const b = createBoard();
    fillRow(b, PLAY_BOTTOM, PLAY_LEFT, PLAY_RIGHT - 1);
    assert.equal(clearLines(b), 0);
  });
});

test('garbage pushes the stack up, and a full top row means top-out', () => {
  const b = createBoard();
  b[idx(PLAY_BOTTOM, PLAY_LEFT)] = 1;
  assert.equal(addGarbageRow(b, 4), true);
  assert.equal(b[idx(PLAY_BOTTOM - 1, PLAY_LEFT)], 1, 'old block moved up');
  assert.equal(b[idx(PLAY_BOTTOM, PLAY_LEFT + 4)], 0, 'the hole');
  assert.equal(b[idx(PLAY_BOTTOM, PLAY_LEFT)], FRAME, 'garbage is stored as FRAME');
  b[idx(0, PLAY_LEFT)] = 1;
  assert.equal(addGarbageRow(b, 0), false);
});

test('features', async (t) => {
  await t.test('column height counts from the floor', () => {
    const b = createBoard();
    b[idx(PLAY_BOTTOM, PLAY_LEFT)] = 1;
    b[idx(PLAY_BOTTOM - 1, PLAY_LEFT)] = 1;
    const h = columnHeights(b);
    assert.equal(h[0], 2);
    assert.equal(h[1], 0);
  });
  await t.test('holes under an overhang count, height goes to the top block', () => {
    const b = createBoard();
    b[idx(PLAY_BOTTOM - 2, PLAY_LEFT)] = 1;
    const f = extractFeatures(b, 0);
    assert.equal(f[FEATURE.holes], 2);
    assert.equal(f[FEATURE.aggregateHeight], 3);
  });
  await t.test('bumpiness uses heights, not cell counts', () => {
    const b = createBoard();
    b[idx(PLAY_BOTTOM - 2, PLAY_LEFT)] = 1; // height 3, 2 holes, 1 cell
    assert.equal(extractFeatures(b, 0)[FEATURE.bumpiness], 3);
  });
});

test('placements', async (t) => {
  await t.test('counts on an empty board', () => {
    const b = createBoard();
    const counts = Object.fromEntries(PIECES.map((p) => [p.name, placements(b, PIECE_INDEX[p.name]).length]));
    // widths: I -> 4 and 1; O -> 2; S/Z -> 3 and 2; J/L/T -> 3,2,3,2
    assert.deepEqual(counts, { Z: 17, I: 17, O: 9, S: 17, L: 34, J: 34, T: 34 });
  });
  await t.test('drops land on the floor', () => {
    const b = createBoard();
    const [flatI, tallI] = PIECES[PIECE_INDEX.I].rotations;
    assert.equal(dropRow(b, flatI, PLAY_LEFT), PLAY_BOTTOM);
    assert.equal(dropRow(b, tallI, PLAY_LEFT), PLAY_BOTTOM - 3);
  });
  await t.test('no placement pokes through a wall', () => {
    const b = createBoard();
    const escapes = [];
    for (const p of PIECES) {
      for (const m of placements(b, PIECE_INDEX[p.name])) {
        for (const [, dc] of p.rotations[m.rotation].cells) {
          const c = m.col + dc;
          if (c < PLAY_LEFT || c > PLAY_RIGHT) escapes.push(`${p.name} at column ${c}`);
        }
      }
    }
    assert.deepEqual(escapes, []);
  });
});

test('the search never touches the live board', () => {
  const b = createBoard();
  fillRow(b, PLAY_BOTTOM, PLAY_LEFT, PLAY_RIGHT - 1); // one cell from a clear
  const before = copyBoard(b);
  chooseMove(b, PIECE_INDEX.I, PIECE_INDEX.T, PUBLISHED_WEIGHTS, { lookahead: true });
  assert.deepEqual(b, before);
});

test('randomness', async (t) => {
  await t.test('a 7-bag deals each piece once', () => {
    const bag = new Bag(3);
    const seen = Array.from({ length: 7 }, () => bag.next());
    assert.deepEqual(seen.sort((x, y) => x - y), [0, 1, 2, 3, 4, 5, 6]);
  });
  await t.test('the same seed gives the same pieces and the same game', () => {
    const a = createPieceGenerator('memoryless', 7);
    const b = createPieceGenerator('memoryless', 7);
    assert.deepEqual([a.next(), a.next(), a.next()], [b.next(), b.next(), b.next()]);
    const play = () => playEpisode(botChooser(PUBLISHED_WEIGHTS, false), 42, { cap: 60 });
    assert.deepEqual(play(), play());
  });
  await t.test('an unknown randomizer is an error, not a silent 7-bag', () => {
    assert.throws(() => createPieceGenerator('bga', 1), /unknown randomizer/);
  });
});

test('the bot is not terrible: 200 pieces, more than 50 lines', () => {
  const r = playEpisode(botChooser(PUBLISHED_WEIGHTS, false), 1, { cap: 200 });
  assert.equal(r.died, false);
  assert.ok(r.lines > 50, `${r.lines} lines`);
});
