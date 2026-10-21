/**
 * presentation.js - the live parts of presentation.html.
 *
 * Uses the real game code and the real trained network, read-only:
 *   src/tetris-core.js   board, pieces, piece generators
 *   src/tetris-dqn.js    the network and the candidate moves
 *   src/tetris-ai.js     the board numbers (holes, bumpiness, ...)
 *   dqn-model.json       the trained network
 * It draws its own boards on <canvas>, so the game page and style.css are
 * never involved.
 */

import { createBoard, PIECES, Bag, MemorylessGen, PLAY_LEFT, PLAY_COLS, PLAY_ROWS, idx, FRAME } from './src/tetris-core.js';
import { loadDqn, nextStates, moveReward } from './src/tetris-dqn.js';
import { extractFeatures } from './src/tetris-ai.js';

const GAMMA = 0.99;
const $ = (id) => document.getElementById(id);

/* ---------------------------------------------------------------- colours */

// piece value -> colour (Z=1, I=2, O=3, S=4, L=5, J=6, T=7, garbage/frame=8)
const PIECE_COLORS = ['#000', '#ef4444', '#22d3ee', '#facc15', '#22c55e', '#f97316', '#3b82f6', '#a855f7', '#6b7280'];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
// 0 = worst option (red) ... 1 = best option (green)
const scoreColor = (t, a = 1) => `hsla(${Math.round(t * 125)}, 80%, 52%, ${a})`;
const fmt = (v, d = 0) => (Number.isFinite(v) ? v.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }) : '–');

/* ---------------------------------------------------------------- canvases */

function setupCanvas(canvas, w, h) {
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.aspectRatio = `${w} / ${h}`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function drawCell(ctx, x, y, s, color) {
  ctx.fillStyle = color;
  ctx.fillRect(x * s + 1, y * s + 1, s - 2, s - 2);
  // a soft highlight so blocks read as blocks
  ctx.fillStyle = 'rgba(255,255,255,0.18)';
  ctx.fillRect(x * s + 1, y * s + 1, s - 2, Math.max(2, s * 0.18));
}

/** Draws the playfield of `board` (rows firstRow..PLAY_ROWS-1). */
function drawBoard(ctx, board, s, firstRow = 0) {
  const rows = PLAY_ROWS - firstRow;
  ctx.fillStyle = css('--board-bg');
  ctx.fillRect(0, 0, PLAY_COLS * s, rows * s);
  ctx.strokeStyle = css('--board-grid');
  ctx.lineWidth = 1;
  for (let c = 1; c < PLAY_COLS; c++) {
    ctx.beginPath();
    ctx.moveTo(c * s + 0.5, 0);
    ctx.lineTo(c * s + 0.5, rows * s);
    ctx.stroke();
  }
  for (let r = 1; r < rows; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * s + 0.5);
    ctx.lineTo(PLAY_COLS * s, r * s + 0.5);
    ctx.stroke();
  }
  for (let r = firstRow; r < PLAY_ROWS; r++) {
    for (let c = 0; c < PLAY_COLS; c++) {
      const v = board[idx(r, c + PLAY_LEFT)];
      if (v) drawCell(ctx, c, r - firstRow, s, PIECE_COLORS[v === FRAME ? 8 : v] || PIECE_COLORS[8]);
    }
  }
}

/** Board cells covered by a placement, as [x, y] in playfield coordinates. */
function moveCells(pieceIdx, move) {
  return PIECES[pieceIdx].rotations[move.rotation].cells.map(([r, c]) => [move.col + c - PLAY_LEFT, move.row + r]);
}

function outlineCells(ctx, cells, s, stroke, width, fill) {
  for (const [x, y] of cells) {
    if (fill) {
      ctx.fillStyle = fill;
      ctx.fillRect(x * s + 1, y * s + 1, s - 2, s - 2);
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.strokeRect(x * s + 1.5, y * s + 1.5, s - 3, s - 3);
  }
}

function drawMiniPiece(canvas, pieceIdx) {
  const ctx = setupCanvas(canvas, 72, 48);
  ctx.clearRect(0, 0, 72, 48);
  if (pieceIdx == null) return;
  const cells = PIECES[pieceIdx].rotations[0].cells;
  const w = Math.max(...cells.map((c) => c[1])) + 1;
  const h = Math.max(...cells.map((c) => c[0])) + 1;
  const s = 16;
  const ox = (72 - w * s) / 2;
  const oy = (48 - h * s) / 2;
  ctx.save();
  ctx.translate(ox, oy);
  for (const [r, c] of cells) drawCell(ctx, c, r, s, PIECE_COLORS[PIECES[pieceIdx].value]);
  ctx.restore();
}

/** Raw board numbers for display: lines, holes, bumpiness, total height. */
function rawFeatures(board, lines) {
  const f = extractFeatures(board, lines);
  return { lines: f[1], holes: f[2], bumpiness: f[3], height: f[0] };
}

/* ======================================================= 1. WATCH IT THINK */

const CELL = 25;
const SPEEDS = [0.5, 1, 2, 5, 15, 60]; // moves per second
const FEATURE_ROWS = [
  { key: 'lines', label: 'Lines', max: 4, cls: 'f-lines' },
  { key: 'holes', label: 'Holes', max: 30, cls: 'f-holes' },
  { key: 'bumpiness', label: 'Bumpiness', max: 30, cls: 'f-bump' },
  { key: 'height', label: 'Height', max: 160, cls: 'f-height' },
];

let net = null;
const game = {
  board: null,
  gen: null,
  cur: 0,
  next: 0,
  cands: [], // sorted best first
  lines: 0,
  pieces: 0,
  games: 1,
  best: 0,
  playing: true,
  over: false,
  hover: -1,
  prevValue: null, // network's value for the board before the current move
  gaps: [],
};

const boardCanvas = $('board');
const boardCtx = setupCanvas(boardCanvas, PLAY_COLS * CELL, PLAY_ROWS * CELL);

/** Every option for the current piece, scored the way the network decides. */
function scoreCandidates() {
  const lookahead = $('lookahead').checked;
  const opts = nextStates(game.board, game.cur, net.input);
  for (const o of opts) {
    o.value = net.predict(o.state); // V(board after this move)
    if (lookahead) {
      let follow = -Infinity;
      for (const t of nextStates(o.board, game.next, net.input)) {
        const w = moveReward(t.lines) + GAMMA * net.predict(t.state);
        if (w > follow) follow = w;
      }
      o.score = follow === -Infinity ? -Infinity : moveReward(o.lines) + GAMMA * follow;
    } else {
      o.score = o.value;
    }
  }
  opts.sort((a, b) => b.score - a.score);
  return opts;
}

function newGame(keepCounters = true) {
  const seed = (Math.random() * 2 ** 31) >>> 0;
  game.gen = $('randomPieces').checked ? new MemorylessGen(seed) : new Bag(seed);
  game.board = createBoard();
  game.cur = game.gen.next();
  game.next = game.gen.next();
  game.lines = 0;
  game.pieces = 0;
  game.over = false;
  game.prevValue = null;
  if (!keepCounters) {
    game.games = 1;
    game.best = 0;
  }
  $('gameOver').hidden = true;
  think();
}

function think() {
  game.cands = scoreCandidates();
  game.hover = -1;
  if (game.cands.length === 0) {
    gameOver();
    return;
  }
  render();
}

function applyMove() {
  if (game.over || game.cands.length === 0) return;
  const chosen = game.cands[0];
  // the Bellman check: before vs. reward + 0.99 x after
  if (game.prevValue != null) {
    const target = moveReward(chosen.lines) + GAMMA * chosen.value;
    updateBellman(game.prevValue, moveReward(chosen.lines), chosen.value, target);
  }
  game.prevValue = chosen.value;
  game.board = chosen.board;
  game.lines += chosen.lines;
  game.pieces++;
  game.cur = game.next;
  game.next = game.gen.next();
  think();
}

function gameOver() {
  game.over = true;
  game.best = Math.max(game.best, game.lines);
  render();
  const overlay = $('gameOver');
  overlay.textContent = `Topped out after ${game.pieces.toLocaleString('en-US')} pieces and ${game.lines.toLocaleString('en-US')} lines. Starting a new game…`;
  overlay.hidden = false;
  setTimeout(() => {
    game.games++;
    newGame();
  }, 1600);
}

/* ----------------------------------------------------------- rendering */

function render() {
  const fast = game.playing && SPEEDS[$('speed').value] >= 15;
  drawBoard(boardCtx, game.board, CELL);

  const cands = game.cands;
  if (cands.length) {
    const finite = cands.filter((c) => Number.isFinite(c.score));
    const hi = finite.length ? finite[0].score : 0;
    const lo = finite.length ? finite[finite.length - 1].score : 0;
    const t = (c) => (Number.isFinite(c.score) && hi > lo ? (c.score - lo) / (hi - lo) : 0);

    if (game.hover >= 0) {
      const c = cands[game.hover];
      outlineCells(boardCtx, moveCells(game.cur, c.move), CELL, scoreColor(t(c)), 3, scoreColor(t(c), 0.45));
    } else {
      if (!fast) {
        // every option, faint; the better it scores, the greener and stronger
        for (let i = cands.length - 1; i >= 1; i--) {
          const k = t(cands[i]);
          outlineCells(boardCtx, moveCells(game.cur, cands[i].move), CELL, scoreColor(k, 0.25 + 0.5 * k), 1.5);
        }
      }
      const piece = PIECES[game.cur];
      outlineCells(boardCtx, moveCells(game.cur, cands[0].move), CELL, '#ffffff', 2.5, PIECE_COLORS[piece.value] + 'cc');
    }
  }

  $('lines').textContent = game.lines.toLocaleString('en-US');
  $('pieces').textContent = game.pieces.toLocaleString('en-US');
  $('games').textContent = game.games;
  $('best').textContent = Math.max(game.best, game.lines).toLocaleString('en-US');
  drawMiniPiece($('curPiece'), game.cur);
  drawMiniPiece($('nextPiece'), game.next);

  renderFeatures();
  if (!fast || game.pieces % 10 === 0) renderOptions();
}

function renderFeatures() {
  const c = game.cands[game.hover >= 0 ? game.hover : 0];
  $('seeWhich').textContent = game.hover > 0 ? `option #${game.hover + 1}` : 'its pick';
  const box = $('features');
  if (!c) {
    box.innerHTML = '';
    $('verdict').textContent = '–';
    return;
  }
  const f = rawFeatures(c.board, c.lines);
  if (!box.children.length) {
    box.innerHTML = FEATURE_ROWS.map(
      (r) => `<div class="feat ${r.cls}"><span>${r.label}</span><div class="feat-bar"><span></span></div><b></b></div>`
    ).join('');
  }
  FEATURE_ROWS.forEach((r, i) => {
    const el = box.children[i];
    el.querySelector('.feat-bar span').style.width = `${Math.min(100, (f[r.key] / r.max) * 100)}%`;
    el.querySelector('b').textContent = f[r.key];
  });
  $('verdict').textContent = fmt(c.score, 1);
}

function renderOptions() {
  const cands = game.cands;
  const list = $('options');
  $('optCount').textContent = cands.length ? `(${cands.length} options)` : '';
  if (!cands.length) {
    list.innerHTML = '';
    return;
  }
  const finite = cands.filter((c) => Number.isFinite(c.score));
  const hi = finite[0]?.score ?? 0;
  const lo = finite[finite.length - 1]?.score ?? 0;
  const t = (c) => (Number.isFinite(c.score) && hi > lo ? (c.score - lo) / (hi - lo) : 0);
  const show = [...cands.slice(0, 5).map((c, i) => [c, i])];
  if (cands.length > 6) show.push(null);
  if (cands.length > 5) show.push([cands[cands.length - 1], cands.length - 1]);

  list.innerHTML = show
    .map((item) => {
      if (!item) return '<li class="sep">…</li>';
      const [c, i] = item;
      const rot = c.move.rotation;
      const label = `${rot === 0 ? 'No turn' : `Turn ×${rot}`}, column ${c.move.col - PLAY_LEFT + 1}${c.lines ? ` · clears ${c.lines}` : ''}`;
      const k = t(c);
      return `<li data-i="${i}" class="${i === 0 ? 'chosen' : ''}${i === game.hover ? ' hover' : ''}">
        <span class="rank">${i === cands.length - 1 && i > 4 ? 'last' : `#${i + 1}`}</span>
        <span>${label}${i === 0 ? ' ✓' : ''}</span>
        <span class="obar"><span style="width:${Math.max(4, k * 100)}%;background:${scoreColor(k)}"></span></span>
        <span class="oval">${fmt(c.score, 0)}</span></li>`;
    })
    .join('');
}

$('options').addEventListener('mouseover', (e) => {
  const li = e.target.closest('li[data-i]');
  if (!li) return;
  game.hover = Number(li.dataset.i);
  render();
});
$('options').addEventListener('mouseleave', () => {
  game.hover = -1;
  render();
});

/* ------------------------------------------------------------- controls */

let timer = null;
function schedule() {
  clearTimeout(timer);
  if (!game.playing) return;
  timer = setTimeout(() => {
    if (!game.over) applyMove();
    schedule();
  }, 1000 / SPEEDS[$('speed').value]);
}

function setPlaying(on) {
  game.playing = on;
  $('playBtn').textContent = on ? 'Pause' : 'Play';
  render();
  schedule();
}

$('playBtn').addEventListener('click', () => setPlaying(!game.playing));
$('stepBtn').addEventListener('click', () => {
  if (game.playing) setPlaying(false);
  applyMove();
});
$('resetBtn').addEventListener('click', () => newGame(false));
$('speed').addEventListener('input', () => {
  const v = SPEEDS[$('speed').value];
  $('speedOut').textContent = `${v} move${v === 1 ? '' : 's'}/s`;
  schedule();
});
$('randomPieces').addEventListener('change', () => newGame(false));
$('lookahead').addEventListener('change', () => {
  if (!game.over) think();
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) clearTimeout(timer);
  else schedule();
});

/* ------------------------------------------------------ the Bellman box */

function updateBellman(before, reward, after, target) {
  const gap = Math.abs(target - before) / Math.max(1e-9, Math.abs(before));
  game.gaps.push(gap);
  if (game.gaps.length > 200) game.gaps.shift();
  const avg = game.gaps.reduce((a, b) => a + b, 0) / game.gaps.length;
  $('bBefore').textContent = fmt(before, 1);
  $('bReward').textContent = fmt(reward, 0);
  $('bAfter').textContent = `${fmt(GAMMA * after, 1)}`;
  $('bTarget').textContent = fmt(target, 1);
  $('bGap').textContent = `${fmt(gap * 100, 1)}%`;
  $('bAvg').textContent = `${fmt(avg * 100, 1)}%`;
}

/* ====================================================== 2. HOW IT LEARNED */

/* a. example board ---------------------------------------------------- */
function drawExample() {
  const rows = [
    '..........',
    '..........',
    '#.........',
    '##....#...',
    '###..##.#.',
    '####.####.',
    '##.#######',
    '#########.',
  ];
  const board = createBoard();
  const first = PLAY_ROWS - rows.length;
  rows.forEach((line, i) => {
    [...line].forEach((ch, c) => {
      if (ch === '#') board[idx(first + i, c + PLAY_LEFT)] = 1 + ((c * 3 + i) % 7);
    });
  });
  const s = 22;
  const canvas = $('exampleBoard');
  const ctx = setupCanvas(canvas, PLAY_COLS * s, rows.length * s);
  drawBoard(ctx, board, s, first);
  // mark holes, and the top of every column
  ctx.font = `bold ${s * 0.7}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (let c = 0; c < PLAY_COLS; c++) {
    let seenBlock = false;
    let top = -1;
    for (let r = 0; r < rows.length; r++) {
      const filled = rows[r][c] === '#';
      if (filled && top < 0) top = r;
      if (filled) seenBlock = true;
      else if (seenBlock) {
        ctx.fillStyle = '#f87171';
        ctx.beginPath();
        ctx.arc(c * s + s / 2, r * s + s / 2, s * 0.22, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    if (top >= 0) {
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(c * s + 3, top * s + 1);
      ctx.lineTo(c * s + s - 3, top * s + 1);
      ctx.stroke();
    }
  }
  const f = rawFeatures(board, 0);
  $('exampleFeatures').innerHTML = [
    ['Lines', f.lines],
    ['Holes', f.holes],
    ['Bumpiness', f.bumpiness],
    ['Height', f.height],
  ]
    .map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`)
    .join('');
}

/* b. reward chart ----------------------------------------------------- */
function drawRewards() {
  const rows = [
    ['Survive a piece', 0],
    ['Clear 1 line', 1],
    ['Clear 2 lines', 2],
    ['Clear 3 lines', 3],
    ['Clear 4 (tetris)', 4],
  ];
  const max = moveReward(4);
  $('rewardChart').innerHTML = rows
    .map(([label, n]) => {
      const r = moveReward(n);
      return `<div class="rbar"><span>${label}</span><div class="track"><div class="fill" style="width:${Math.max(1.5, (r / max) * 100)}%"></div></div><b>${r}</b></div>`;
    })
    .join('');
}

/* d. exploration mini-game ------------------------------------------- */
const EPS = [0, 0.01, 0.1, 0.3, 1];
const explore = { board: null, gen: null, pieces: 0, games: 0, total: 0, visible: false, timer: null };
const EX_CELL = 14;
const exCtx = setupCanvas($('exploreBoard'), PLAY_COLS * EX_CELL, PLAY_ROWS * EX_CELL);
const exRand = Math.random;

function exploreNewGame() {
  explore.board = createBoard();
  explore.gen = new Bag((Math.random() * 2 ** 31) >>> 0);
  explore.pieces = 0;
}

function exploreStep() {
  const eps = EPS[$('epsilon').value];
  const opts = nextStates(explore.board, explore.gen.next(), net.input);
  if (opts.length === 0) {
    explore.games++;
    explore.total += explore.pieces;
    exploreNewGame();
  } else {
    let pick;
    if (exRand() < eps) pick = opts[Math.floor(exRand() * opts.length)];
    else {
      let best = -Infinity;
      for (const o of opts) {
        const v = net.predict(o.state);
        if (v > best) {
          best = v;
          pick = o;
        }
      }
    }
    explore.board = pick.board;
    explore.pieces++;
  }
  drawBoard(exCtx, explore.board, EX_CELL);
  $('exPieces').textContent = explore.pieces.toLocaleString('en-US');
  $('exGames').textContent = explore.games;
  $('exAvg').textContent = explore.games ? fmt(explore.total / explore.games, 0) : '–';
}

function exploreRun() {
  clearInterval(explore.timer);
  if (explore.visible && !document.hidden) explore.timer = setInterval(exploreStep, 50);
}

$('epsilon').addEventListener('input', () => {
  $('epsOut').textContent = `${EPS[$('epsilon').value] * 100}%`;
  explore.games = 0;
  explore.total = 0;
  exploreNewGame();
});
document.addEventListener('visibilitychange', exploreRun);

/* e. training chart --------------------------------------------------- */
// The real log of `node train_dqn.js` (the run that produced dqn-model.json):
// every 100 games -> average pieces over the last 100 training games, and the
// greedy test (5 games, no random moves, capped at 1,000 pieces).
const TRAIN = {
  games: [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000],
  avg: [29.6, 30.2, 29.7, 29.8, 29.6, 31.3, 32.7, 33.4, 37.2, 36.8, 38.3, 40.7, 42.9, 45.1, 46.9, 52.8, 59.7, 69.8, 78, 192, 206, 374.6, 443.1, 404.7, 676.3, 499.7, 537.5, 489.6, 640.5, 523.5],
  test: [32, 32, 33, 36, 50, 51, 48, 58, 59, 58, 56, 57, 58, 61, 80, 136, 177, 78, 101, 71, 119, 246, 65, 248, 638, 161, 816, 1000, 83, 881],
};

function drawTrainChart() {
  const W = 900;
  const H = 320;
  const m = { l: 52, r: 18, t: 18, b: 40 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;
  const yMin = 20;
  const yMax = 1500;
  const x = (g) => m.l + (g / 3000) * iw;
  const y = (v) => m.t + ih - ((Math.log(v) - Math.log(yMin)) / (Math.log(yMax) - Math.log(yMin))) * ih;
  const accent = css('--accent');
  const good = css('--good');
  const muted = css('--muted');

  const path = (vals) => vals.map((v, i) => `${i ? 'L' : 'M'}${x(TRAIN.games[i]).toFixed(1)},${y(v).toFixed(1)}`).join('');
  let best = -1;
  const saved = TRAIN.test.map((v) => (v > best ? ((best = v), true) : false));

  const yTicks = [30, 100, 300, 1000];
  const xTicks = [0, 500, 1000, 1500, 2000, 2500, 3000];
  const svg = `
  <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Training curve: pieces survived per game over 3,000 training games">
    <rect x="${x(0)}" y="${m.t}" width="${x(2000) - x(0)}" height="${ih}" fill="${accent}" opacity="0.07" rx="6"/>
    <text class="lbl" x="${x(1000)}" y="${m.t + 16}" text-anchor="middle">exploring: random moves fall from 100% to 0.1%</text>
    <g class="grid">${yTicks.map((v) => `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`).join('')}</g>
    <g class="axis">
      ${yTicks.map((v) => `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v.toLocaleString('en-US')}</text>`).join('')}
      ${xTicks.map((g) => `<text x="${x(g)}" y="${H - m.b + 18}" text-anchor="middle">${g.toLocaleString('en-US')}</text>`).join('')}
      <text x="${m.l + iw / 2}" y="${H - 4}" text-anchor="middle">training games</text>
      <text x="14" y="${m.t + ih / 2}" text-anchor="middle" transform="rotate(-90 14 ${m.t + ih / 2})">pieces per game (log scale)</text>
    </g>
    <path d="${path(TRAIN.avg)}" fill="none" stroke="${muted}" stroke-width="2" stroke-dasharray="5 4"/>
    <path d="${path(TRAIN.test)}" fill="none" stroke="${accent}" stroke-width="2.5"/>
    ${TRAIN.test
      .map((v, i) =>
        saved[i]
          ? `<circle cx="${x(TRAIN.games[i])}" cy="${y(v)}" r="4.5" fill="${good}"><title>game ${TRAIN.games[i]}: ${v} pieces (saved)</title></circle>`
          : `<circle cx="${x(TRAIN.games[i])}" cy="${y(v)}" r="3" fill="${accent}"><title>game ${TRAIN.games[i]}: ${v} pieces</title></circle>`
      )
      .join('')}
    <text class="tip" x="${x(2800) - 8}" y="${y(1000) - 10}" text-anchor="end">saved: 1,000 pieces</text>
    <text class="tip" x="${x(2900) + 6}" y="${y(83) + 16}" text-anchor="middle">83</text>
  </svg>`;
  $('trainChart').innerHTML =
    svg +
    `<div class="legend">
      <span><i style="background:${accent}"></i>test game, no random moves</span>
      <span><i style="background:${muted}"></i>average of training games (with random moves)</span>
      <span><i style="background:${good};height:9px;width:9px;border-radius:50%"></i>new best, saved</span>
    </div>`;
}

/* ===================================================== 3. HOW GOOD IS IT */

const SCORES = [
  {
    title: 'Normal pieces',
    sub: '7-bag, no preview · games survived to 10,000 pieces',
    unit: '/ 50',
    rows: [
      ['Search bot', 42],
      ['Network', 50, 'net'],
    ],
    max: 50,
  },
  {
    title: 'Fully random pieces',
    sub: 'no preview · games survived to 2,000 pieces',
    unit: '/ 50',
    rows: [
      ['Search bot', 41],
      ['Network', 38, 'net'],
    ],
    max: 50,
  },
  {
    title: 'Fully random pieces, next piece visible',
    sub: 'games survived to 2,000 pieces',
    unit: '/ 50',
    rows: [
      ['Search bot', 50],
      ['Network', 38, 'net'],
      ['Network + lookahead', 48, 'net'],
    ],
    max: 50,
  },
  {
    title: 'Rising garbage, every 5 pieces',
    sub: 'next piece visible · everyone loses eventually · average pieces survived',
    unit: '',
    rows: [
      ['Search bot', 254],
      ['Network', 136, 'net'],
      ['Network + lookahead', 319, 'net'],
    ],
    max: 340,
  },
];

function drawScoreboard() {
  $('scoreboard').innerHTML = SCORES.map((card) => {
    const top = Math.max(...card.rows.map((r) => r[1]));
    return `<div class="card score-card"><h3>${card.title}</h3><p class="sub">${card.sub}</p>
      ${card.rows
        .map(([who, v, cls]) => {
          const win = v === top;
          return `<div class="srow ${cls || ''} ${win ? 'win' : ''}">
            <span class="who">${who}${win ? '<span class="badge">best</span>' : ''}</span>
            <div class="track"><div class="fill" style="width:${(v / card.max) * 100}%"></div></div>
            <b>${v}${card.unit ? ` <small>${card.unit}</small>` : ''}</b></div>`;
        })
        .join('')}</div>`;
  }).join('');
}

/* ------------------------------------------------------------------ start */

async function start() {
  drawRewards();
  drawExample();
  drawTrainChart();
  drawScoreboard();
  $('speedOut').textContent = `${SPEEDS[$('speed').value]} moves/s`;
  $('epsOut').textContent = `${EPS[$('epsilon').value] * 100}%`;

  try {
    const res = await fetch('./dqn-model.json');
    if (!res.ok) throw new Error(`dqn-model.json: HTTP ${res.status}`);
    net = loadDqn(await res.json());
  } catch (e) {
    const o = $('gameOver');
    o.textContent = `Could not load the network (${e.message}). Open this page through a local server (npx serve .).`;
    o.hidden = false;
    return;
  }

  newGame(false);
  setPlaying(true);

  exploreNewGame();
  drawBoard(exCtx, explore.board, EX_CELL);
  new IntersectionObserver((entries) => {
    explore.visible = entries[0].isIntersecting;
    exploreRun();
  }).observe($('exploreBoard'));

  // redraw theme-dependent pieces if the colour scheme changes
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    drawExample();
    drawTrainChart();
    render();
  });
}

start();
