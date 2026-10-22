/**
 * learning.js - the live part of learning.html.
 *
 * Six copies of the network, saved during training (snapshots/game-N.json),
 * play side by side. Game k of every board uses the same pieces (seed
 * base + k), so the only difference between the boards is the network.
 * Uses the real game code read-only and draws its own boards on <canvas>.
 */

import { createBoard, Bag, PLAY_LEFT, PLAY_COLS, PLAY_ROWS, idx, FRAME } from './src/tetris-core.js';
import { loadDqn, nextStates } from './src/tetris-dqn.js';

const $ = (id) => document.getElementById(id);
const fmt = (v) => (Number.isFinite(v) ? Math.round(v).toLocaleString('en-US') : '–');

const STAGES = [
  {
    games: 0,
    title: 'Before training',
    chip: 'random weights',
    tone: '',
    text: 'Its weights are random numbers, so its scores mean nothing. It stacks pieces almost anywhere.',
    measured: { mean: 21.2, lost: 50 },
  },
  {
    games: 1000,
    title: 'After 1,000 games',
    chip: 'exploring: half its moves random',
    tone: '',
    text: 'It has started to avoid the worst spots, but it hasn’t yet learned that clearing lines is what keeps it alive.',
    measured: { mean: 61.3, lost: 50 },
  },
  {
    games: 1700,
    title: 'After 1,700 games',
    chip: 'exploring: 15% random moves',
    tone: '',
    text: 'It now clears lines on purpose and keeps the stack lower, but holes still pile up and end the game.',
    measured: { mean: 216.9, lost: 50 },
  },
  {
    games: 2200,
    title: 'After 2,200 games',
    chip: 'exploring finished',
    tone: '',
    text: 'Now it plays its own best moves. Games last a few hundred pieces, and it is learning fast from them.',
    measured: { mean: 375.5, lost: 50 },
  },
  {
    games: 2800,
    title: 'After 2,800 games',
    chip: 'saved ✓ this is the network in the game',
    tone: 'good',
    text: 'Fully trained. In the test it never lost a game. This copy is dqn-model.json, the one you play against.',
    measured: { mean: 2000, lost: 0 },
  },
  {
    games: 2900,
    title: 'After 2,900 games',
    chip: 'one step too far',
    tone: 'bad',
    text: 'Only 100 games later, training knocked it over. It plays far worse, which is why training keeps the best copy.',
    measured: { mean: 90.9, lost: 50 },
  },
];

const SPEEDS = [2, 5, 10, 30, 60];
const CELL = 12;

/* --------------------------------------------------------------- drawing */

// piece value -> colour (Z=1, I=2, O=3, S=4, L=5, J=6, T=7, garbage/frame=8)
const PIECE_COLORS = ['#000', '#ef4444', '#22d3ee', '#facc15', '#22c55e', '#f97316', '#3b82f6', '#a855f7', '#6b7280'];
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function setupCanvas(canvas, w, h) {
  const dpr = Math.max(1, Math.min(3, window.devicePixelRatio || 1));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.aspectRatio = `${w} / ${h}`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function drawBoard(ctx, board) {
  const s = CELL;
  ctx.fillStyle = css('--board-bg');
  ctx.fillRect(0, 0, PLAY_COLS * s, PLAY_ROWS * s);
  ctx.strokeStyle = css('--board-grid');
  ctx.lineWidth = 1;
  for (let c = 1; c < PLAY_COLS; c++) {
    ctx.beginPath();
    ctx.moveTo(c * s + 0.5, 0);
    ctx.lineTo(c * s + 0.5, PLAY_ROWS * s);
    ctx.stroke();
  }
  for (let r = 0; r < PLAY_ROWS; r++) {
    for (let c = 0; c < PLAY_COLS; c++) {
      const v = board[idx(r, c + PLAY_LEFT)];
      if (!v) continue;
      ctx.fillStyle = PIECE_COLORS[v === FRAME ? 8 : v] || PIECE_COLORS[8];
      ctx.fillRect(c * s + 1, r * s + 1, s - 2, s - 2);
    }
  }
}

/* ------------------------------------------------------------------ race */

let baseSeed = (Math.random() * 1e9) >>> 0;
let playing = true;
let timer = null;
const boards = [];

function buildCards() {
  $('race').innerHTML = STAGES.map(
    (st, i) => `
    <article class="card stage">
      <div class="stage-board">
        <canvas id="cv${i}" aria-label="${st.title}"></canvas>
        <div id="ov${i}" class="overlay" hidden></div>
      </div>
      <div>
        <h3>${st.title}</h3>
        <span class="chip ${st.tone}">${st.chip}</span>
        <p>${st.text}</p>
        <dl class="stage-stats">
          <div><dt>This game</dt><dd id="pc${i}">0</dd></div>
          <div><dt>Lines</dt><dd id="ln${i}">0</dd></div>
          <div class="wide"><dt>Average pieces per game</dt><dd id="av${i}">–</dd></div>
          <div class="wide"><dt>Games lost so far</dt><dd id="gl${i}">0</dd></div>
        </dl>
      </div>
    </article>`
  ).join('');
}

function startGame(b) {
  b.board = createBoard();
  b.gen = new Bag(baseSeed + b.gameNo); // game k: same pieces on every board
  b.pieces = 0;
  b.lines = 0;
  b.wait = 0;
  b.overlay.hidden = true;
}

function resetAll() {
  for (const b of boards) {
    b.gameNo = 0;
    b.lost = 0;
    b.totalPieces = 0;
    startGame(b);
    paint(b);
  }
}

function step(b) {
  if (b.wait > 0) {
    // showing "topped out"; start the next game when the pause is over
    if (--b.wait === 0) {
      b.gameNo++;
      startGame(b);
    }
    return;
  }
  const opts = nextStates(b.board, b.gen.next(), b.net.input);
  if (opts.length === 0) {
    b.lost++;
    b.totalPieces += b.pieces;
    b.overlay.textContent = `Topped out after ${fmt(b.pieces)} pieces`;
    b.overlay.hidden = false;
    b.wait = Math.max(2, Math.round(SPEEDS[$('raceSpeed').value] * 0.9));
    return;
  }
  let best = opts[0];
  let bestV = -Infinity;
  for (const o of opts) {
    const v = b.net.predict(o.state);
    if (v > bestV) {
      bestV = v;
      best = o;
    }
  }
  b.board = best.board;
  b.pieces++;
  b.lines += best.lines;
}

function paint(b) {
  drawBoard(b.ctx, b.board);
  const i = b.i;
  $(`pc${i}`).textContent = fmt(b.pieces);
  $(`ln${i}`).textContent = fmt(b.lines);
  $(`av${i}`).textContent = b.lost ? fmt(b.totalPieces / b.lost) : 'no game lost yet';
  $(`gl${i}`).textContent = b.lost;
}

function tick() {
  for (const b of boards) {
    step(b);
    paint(b);
  }
  updateStatus();
}

function updateStatus() {
  const sp = SPEEDS[$('raceSpeed').value];
  $('raceStatus').textContent = playing
    ? `Playing: every board places ${sp} pieces per second. Game 1 on every board gets the same pieces, game 2 the same, and so on, so the boards are a fair comparison.`
    : 'Paused. Press Play to continue, or New pieces to start everyone again with a different set of pieces.';
}

function schedule() {
  clearInterval(timer);
  if (playing && !document.hidden) timer = setInterval(tick, 1000 / SPEEDS[$('raceSpeed').value]);
  updateStatus();
}

/* --------------------------------------------------------- measured chart */

function drawMeasured() {
  const max = Math.log10(2000);
  const min = Math.log10(10);
  $('measured').innerHTML =
    STAGES.map((st) => {
      const w = ((Math.log10(st.measured.mean) - min) / (max - min)) * 100;
      const cls = st.tone === 'good' ? 'best' : st.tone === 'bad' ? 'drop' : '';
      return `<div class="mrow ${cls}">
        <span>${st.title}</span>
        <div class="track"><div class="fill" style="width:${Math.max(3, w)}%"></div></div>
        <span><b>${fmt(st.measured.mean)}</b> <small>pieces · lost ${st.measured.lost}/50</small></span>
      </div>`;
    }).join('') +
    '<p class="hint">Average pieces survived per game (log scale: each step to the right is about 10× more). Games stop at 2,000 pieces, so 2,000 means it never lost.</p>';
}

/* ------------------------------------------------------------------ start */

async function start() {
  buildCards();
  drawMeasured();
  $('raceSpeedOut').textContent = `${SPEEDS[$('raceSpeed').value]} moves/s`;

  let nets;
  try {
    nets = await Promise.all(
      STAGES.map((st) =>
        fetch(`./snapshots/game-${st.games}.json`).then((r) => {
          if (!r.ok) throw new Error(`snapshots/game-${st.games}.json: HTTP ${r.status}`);
          return r.json();
        })
      )
    );
  } catch (e) {
    $('raceStatus').textContent = `Could not load the saved networks (${e.message}). Open this page through a local server (npx serve .).`;
    return;
  }

  STAGES.forEach((st, i) => {
    const canvas = $(`cv${i}`);
    boards.push({
      i,
      net: loadDqn(nets[i]),
      ctx: setupCanvas(canvas, PLAY_COLS * CELL, PLAY_ROWS * CELL),
      overlay: $(`ov${i}`),
      gameNo: 0,
      lost: 0,
      totalPieces: 0,
    });
  });
  resetAll();

  $('raceBtn').addEventListener('click', () => {
    playing = !playing;
    $('raceBtn').textContent = playing ? 'Pause' : 'Play';
    schedule();
  });
  $('newPieces').addEventListener('click', () => {
    baseSeed = (Math.random() * 1e9) >>> 0;
    resetAll();
    updateStatus();
  });
  $('raceSpeed').addEventListener('input', () => {
    $('raceSpeedOut').textContent = `${SPEEDS[$('raceSpeed').value]} moves/s`;
    schedule();
  });
  document.addEventListener('visibilitychange', schedule);
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => boards.forEach(paint));

  schedule();
}

start();
