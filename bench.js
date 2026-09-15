/**
 * bench.js - sanity tests and a speed benchmark. Run with:
 *
 *   node bench.js --test     # correctness only
 *   node bench.js            # tests, then benchmark
 *
 * The benchmark number is the thing to care about: it tells you how many games
 * per second you can run, which decides whether Phase 1 takes ten minutes or
 * ten days.
 */

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
  placePiece,
  clearLines,
  placements,
  dropRow,
  columnHeights,
  PIECES,
  PIECE_INDEX,
  Bag,
} from './src/tetris-core.js';
import { extractFeatures, chooseMove, PUBLISHED_WEIGHTS, FEATURE_NAMES } from './src/tetris-ai.js';
import { playGame } from './src/tetris-headless.js';

let passed = 0;
let failed = 0;

function check(name, actual, expected) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (ok) {
    passed++;
    console.log(`  ok    ${name}`);
  } else {
    failed++;
    console.log(`  FAIL  ${name}\n          expected ${JSON.stringify(expected)}\n          got      ${JSON.stringify(actual)}`);
  }
}

function fillRow(board, row, from = PLAY_LEFT, to = PLAY_RIGHT, value = 1) {
  for (let c = from; c <= to; c++) board[idx(row, c)] = value;
}

console.log('\nsanity tests');
console.log('------------');

/* board geometry */
{
  const b = createBoard();
  check('playfield is 10 wide', PLAY_RIGHT - PLAY_LEFT + 1, 10);
  check('playfield is 24 tall', PLAY_BOTTOM + 1, 24);
  check('left wall is framed', b[idx(0, 0)], FRAME);
  check('right wall is framed', b[idx(0, WIDTH - 1)], FRAME);
  check('floor is framed', b[idx(HEIGHT - 1, PLAY_LEFT)], FRAME);
  check('playfield starts empty', b[idx(0, PLAY_LEFT)], 0);
}

/* line clearing */
{
  const b = createBoard();
  fillRow(b, PLAY_BOTTOM);
  check('a full row clears', clearLines(b), 1);
  check('and leaves the row empty', b[idx(PLAY_BOTTOM, PLAY_LEFT)], 0);
}
{
  const b = createBoard();
  fillRow(b, PLAY_BOTTOM);
  b[idx(PLAY_BOTTOM - 1, PLAY_LEFT + 4)] = 7; // one block floating above
  clearLines(b);
  check('blocks above a cleared row fall', b[idx(PLAY_BOTTOM, PLAY_LEFT + 4)], 7);
  check('and leave nothing behind', b[idx(PLAY_BOTTOM - 1, PLAY_LEFT + 4)], 0);
}
{
  const b = createBoard();
  fillRow(b, PLAY_BOTTOM);
  fillRow(b, PLAY_BOTTOM - 1);
  fillRow(b, PLAY_BOTTOM - 2);
  fillRow(b, PLAY_BOTTOM - 3);
  check('a tetris clears four rows', clearLines(b), 4);
}
{
  const b = createBoard();
  fillRow(b, PLAY_BOTTOM, PLAY_LEFT, PLAY_RIGHT - 1); // one cell short
  check('an incomplete row does not clear', clearLines(b), 0);
}

/* heights, holes, bumpiness */
{
  const b = createBoard();
  b[idx(PLAY_BOTTOM, PLAY_LEFT)] = 1;
  b[idx(PLAY_BOTTOM - 1, PLAY_LEFT)] = 1;
  const h = columnHeights(b);
  check('column height counts from the floor', h[0], 2);
  check('empty columns have height 0', h[1], 0);
}
{
  // a covered gap: block at the top of the column, empty beneath it
  const b = createBoard();
  b[idx(PLAY_BOTTOM - 2, PLAY_LEFT)] = 1;
  const f = extractFeatures(b, 0);
  check('holes under an overhang are counted', f[2], 2);
  check('height is measured to the top block', f[0], 3);
}
{
  // two columns of height 3 and 0 -> bumpiness 3, and crucially the hole
  // beneath does NOT change bumpiness (this was the old bug: cell counts
  // were used instead of heights)
  const b = createBoard();
  b[idx(PLAY_BOTTOM - 2, PLAY_LEFT)] = 1; // height 3, 2 holes, 1 cell
  const f = extractFeatures(b, 0);
  check('bumpiness uses heights, not cell counts', f[3], 3);
}

/* placements */
{
  const b = createBoard();
  const counts = {};
  for (const p of PIECES) counts[p.name] = placements(b, PIECE_INDEX[p.name]).length;
  // widths: I -> 4 and 1; O -> 2; S/Z -> 3 and 2; J/L/T -> 3,2,3,2
  check('placement counts on an empty board', counts, {
    Z: 17,
    I: 17,
    O: 9,
    S: 17,
    L: 34,
    J: 34,
    T: 34,
  });
}
{
  const b = createBoard();
  const flatI = PIECES[PIECE_INDEX.I].rotations[0];
  check('flat I lands on the floor', dropRow(b, flatI, PLAY_LEFT), PLAY_BOTTOM);
  const tallI = PIECES[PIECE_INDEX.I].rotations[1];
  check('vertical I lands 4 tall', dropRow(b, tallI, PLAY_LEFT), PLAY_BOTTOM - 3);
}
{
  // no placement may poke through a wall
  const b = createBoard();
  for (const p of PIECES) {
    for (const m of placements(b, PIECE_INDEX[p.name])) {
      const rot = p.rotations[m.rotation];
      for (const [dr, dc] of rot.cells) {
        const c = m.col + dc;
        if (c < PLAY_LEFT || c > PLAY_RIGHT) {
          failed++;
          console.log(`  FAIL  ${p.name} placement escapes the playfield at column ${c}`);
        }
      }
    }
  }
  passed++;
  console.log('  ok    every placement stays inside the playfield');
}

/* the big one: search must not touch the live board */
{
  const b = createBoard();
  fillRow(b, PLAY_BOTTOM, PLAY_LEFT, PLAY_RIGHT - 1); // one cell from a clear
  const before = copyBoard(b);
  chooseMove(b, PIECE_INDEX.I, PIECE_INDEX.T, PUBLISHED_WEIGHTS, { lookahead: true });
  check('chooseMove leaves the real board untouched', Array.from(b), Array.from(before));
}

/* determinism */
{
  const a = playGame(PUBLISHED_WEIGHTS, 42, { maxPieces: 60, lookahead: false });
  const c = playGame(PUBLISHED_WEIGHTS, 42, { maxPieces: 60, lookahead: false });
  check('same seed gives the same game', a.lines, c.lines);
  const d = new Bag(7);
  const e = new Bag(7);
  check('bags with the same seed agree', [d.next(), d.next(), d.next()], [e.next(), e.next(), e.next()]);
}
{
  const bag = new Bag(3);
  const seen = [];
  for (let i = 0; i < 7; i++) seen.push(bag.next());
  check('a 7-bag deals each piece once', [...seen].sort((x, y) => x - y), [0, 1, 2, 3, 4, 5, 6]);
}

/* the bot should not be terrible */
{
  const r = playGame(PUBLISHED_WEIGHTS, 1, { maxPieces: 200, lookahead: false });
  const ok = !r.toppedOut && r.lines > 50;
  if (ok) {
    passed++;
    console.log(`  ok    bot survives 200 pieces and clears ${r.lines} lines`);
  } else {
    failed++;
    console.log(`  FAIL  bot did poorly: ${JSON.stringify(r)}`);
  }
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);

if (process.argv.includes('--test')) process.exit(0);

/* --------------------------------------------------------------- benchmark */

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

function run(label, games, opts) {
  const t0 = performance.now();
  const results = [];
  for (let seed = 1; seed <= games; seed++) results.push(playGame(PUBLISHED_WEIGHTS, seed, opts));
  const elapsed = (performance.now() - t0) / 1000;

  const lines = results.map((r) => r.lines);
  const deaths = results.filter((r) => r.toppedOut).length;
  const pieces = results.reduce((a, r) => a + r.pieces, 0);

  console.log(`\n${label}`);
  console.log(`  games            ${games}`);
  console.log(`  median lines     ${median(lines)}`);
  console.log(`  mean lines       ${(lines.reduce((a, b) => a + b, 0) / games).toFixed(1)}`);
  console.log(`  worst / best     ${Math.min(...lines)} / ${Math.max(...lines)}`);
  console.log(`  topped out       ${deaths} of ${games}`);
  console.log(`  time             ${elapsed.toFixed(2)}s`);
  console.log(`  throughput       ${(games / elapsed).toFixed(1)} games/s, ${Math.round(pieces / elapsed)} pieces/s`);
  return games / elapsed;
}

console.log('\nbenchmark (published weights, 500-piece cap)');
console.log('--------------------------------------------');

const rate1 = run('1-ply (what you train with)', 100, { maxPieces: 500, lookahead: false });
run('2-ply lookahead (what you play with)', 20, { maxPieces: 500, lookahead: true });

const gaGames = 50 * 5 * 40; // population x seeds x generations
console.log(`\na typical GA run is ~${gaGames.toLocaleString()} games`);
console.log(`at ${rate1.toFixed(1)} games/s that is about ${(gaGames / rate1 / 60).toFixed(1)} minutes, single-threaded.\n`);

/* feature sanity printout */
const b = createBoard();
placePiece(b, PIECES[PIECE_INDEX.T].rotations[0], PLAY_BOTTOM - 1, PLAY_LEFT + 2);
const f = extractFeatures(b, 0);
console.log('features of a board with one T on the floor:');
FEATURE_NAMES.forEach((n, i) => console.log(`  ${n.padEnd(16)} ${f[i]}`));
console.log('');