/**
 * versus.js - "Play vs Bot": you against the network (or the search bot).
 *
 * Both boards get exactly the same pieces (same seed), both can see the next
 * piece and use hold. Two modes:
 *   battle  clearing 2 / 3 / 4 lines sends 1 / 2 / 4 garbage rows to the other
 *           board; your own clears cancel garbage coming at you. Top out = lose.
 *   race    first to 40 lines wins; topping out loses.
 *
 * The opponent decides with the real code (src/tetris-dqn.js or
 * src/tetris-ai.js) at the speed you pick. Uses the game code read-only and
 * draws on <canvas>, so the other pages are not affected.
 */

import {
  createBoard,
  PIECES,
  Bag,
  collides,
  placePiece,
  clearLines,
  addGarbageRow,
  PLAY_LEFT,
  PLAY_COLS,
  PLAY_ROWS,
  idx,
  FRAME,
  mulberry32,
} from './src/tetris-core.js';
import { loadDqn, chooseDqnAction } from './src/tetris-dqn.js';
import { chooseMoveAction, PUBLISHED_WEIGHTS } from './src/tetris-ai.js';

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ rules */

const ATTACK = [0, 0, 1, 2, 4]; // garbage rows sent for 0..4 lines cleared
const POINTS = [0, 40, 100, 300, 1200]; // classic scoring, as in the game
const RACE_LINES = 40;
const LOCK_DELAY = 500; // ms a landed piece can still slide or turn
const MAX_LOCK_RESETS = 15;
const DAS = 160; // ms before a held left/right starts repeating
const ARR = 45; // ms between repeats
const SOFT_DROP = 35; // ms per row while holding down

/* --------------------------------------------------------------- drawing */

const CELL = 26;
const IMAGES = { 1: 'red', 2: 'skyBlue', 3: 'yellow', 4: 'green', 5: 'orange', 6: 'blue', 7: 'purple' };
const COLORS = { 1: '#e6483d', 2: '#4ec3e0', 3: '#f2c94c', 4: '#5fc27e', 5: '#f2994a', 6: '#4a7ef2', 7: '#9b51e0', 8: '#6b7280' };
const sprites = {};
for (const [v, name] of Object.entries(IMAGES)) {
  const img = new Image();
  img.src = `images/${name}.png`;
  sprites[v] = img;
}
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

function drawCell(ctx, x, y, s, v, alpha = 1) {
  ctx.globalAlpha = alpha;
  const img = sprites[v];
  if (v !== FRAME && img && img.complete && img.naturalWidth) {
    ctx.drawImage(img, x * s, y * s, s, s);
  } else {
    ctx.fillStyle = COLORS[v] || COLORS[8];
    ctx.fillRect(x * s + 1, y * s + 1, s - 2, s - 2);
  }
  ctx.globalAlpha = 1;
}

function drawBoard(ctx, board) {
  const s = CELL;
  ctx.fillStyle = css('--board-bg') || '#0a0e1a';
  ctx.fillRect(0, 0, PLAY_COLS * s, PLAY_ROWS * s);
  ctx.strokeStyle = css('--board-grid') || '#161c30';
  ctx.lineWidth = 1;
  for (let c = 1; c < PLAY_COLS; c++) {
    ctx.beginPath();
    ctx.moveTo(c * s + 0.5, 0);
    ctx.lineTo(c * s + 0.5, PLAY_ROWS * s);
    ctx.stroke();
  }
  for (let r = 1; r < PLAY_ROWS; r++) {
    ctx.beginPath();
    ctx.moveTo(0, r * s + 0.5);
    ctx.lineTo(PLAY_COLS * s, r * s + 0.5);
    ctx.stroke();
  }
  for (let r = 0; r < PLAY_ROWS; r++) {
    for (let c = 0; c < PLAY_COLS; c++) {
      const v = board[idx(r, c + PLAY_LEFT)];
      if (v) drawCell(ctx, c, r, s, v);
    }
  }
}

function drawPiece(ctx, piece, r, row, col, alpha = 1) {
  const rot = PIECES[piece].rotations[r];
  for (const [dr, dc] of rot.cells) drawCell(ctx, col + dc - PLAY_LEFT, row + dr, CELL, rot.value, alpha);
}

function drawGhost(ctx, piece, r, row, col) {
  const rot = PIECES[piece].rotations[r];
  ctx.strokeStyle = 'rgba(255,255,255,0.55)';
  ctx.lineWidth = 2;
  for (const [dr, dc] of rot.cells) {
    ctx.strokeRect((col + dc - PLAY_LEFT) * CELL + 2, (row + dr) * CELL + 2, CELL - 4, CELL - 4);
  }
}

const miniCtx = {};
function drawMini(id, piece) {
  const w = 64;
  const h = 40;
  miniCtx[id] ??= setupCanvas($(id), w, h);
  const ctx = miniCtx[id];
  ctx.clearRect(0, 0, w, h);
  if (piece == null) return;
  const rot = PIECES[piece].rotations[0];
  const s = 14;
  const ox = (w - rot.width * s) / 2;
  const oy = (h - rot.height * s) / 2;
  ctx.save();
  ctx.translate(ox, oy);
  for (const [r, c] of rot.cells) drawCell(ctx, c, r, s, rot.value);
  ctx.restore();
}

/* --------------------------------------------------------------- players */

function newPlayer(seed) {
  const bag = new Bag(seed);
  return {
    board: createBoard(),
    bag,
    cur: bag.next(),
    next: bag.next(),
    held: null,
    canHold: true,
    active: null, // { r, row, col } for you; the AI uses aiAnim
    lines: 0,
    score: 0,
    pieces: 0,
    tetrises: 0,
    sent: 0,
    pending: 0, // garbage rows on their way to this board
    dead: false,
    holeRand: mulberry32(seed + 777), // same garbage holes on both boards
  };
}

const rotOf = (p, r) => PIECES[p.cur].rotations[r];

function spawn(p) {
  const rot = rotOf(p, 0);
  const col = PLAY_LEFT + Math.floor((PLAY_COLS - rot.width) / 2);
  p.active = { r: 0, row: 0, col };
  if (collides(p.board, rot, 0, col)) p.dead = true;
}

function swapHold(p) {
  if (p.held == null) {
    p.held = p.cur;
    p.cur = p.next;
    p.next = p.bag.next();
  } else {
    [p.held, p.cur] = [p.cur, p.held];
  }
  p.canHold = false;
}

/** Locks a piece in place, scores it, and handles garbage both ways. */
function lock(p, r, row, col) {
  placePiece(p.board, rotOf(p, r), row, col);
  const cleared = clearLines(p.board);
  p.pieces++;
  p.lines += cleared;
  p.score += POINTS[cleared];
  if (cleared === 4) p.tetrises++;

  if (mode === 'battle') {
    let attack = ATTACK[cleared];
    const cancel = Math.min(attack, p.pending); // your clears cancel incoming garbage first
    p.pending -= cancel;
    attack -= cancel;
    if (attack > 0) {
      other(p).pending += attack;
      p.sent += attack;
    }
    if (cleared === 0 && p.pending > 0) {
      // garbage arrives when you place a piece without clearing a line
      const hole = Math.floor(p.holeRand() * PLAY_COLS);
      for (let i = 0; i < p.pending && !p.dead; i++) if (!addGarbageRow(p.board, hole)) p.dead = true;
      p.pending = 0;
    }
  }

  p.cur = p.next;
  p.next = p.bag.next();
  p.canHold = true;
  p.active = null;
}

/* ------------------------------------------------------------------ state */

let you = null;
let ai = null;
const other = (p) => (p === you ? ai : you);
let mode = 'battle';
let state = 'loading'; // loading | idle | countdown | playing | paused | over
let chooser = null;
let aiPps = 1;
let aiAnim = null; // { r, row, col, target, start, dur }
let aiNextAt = 0;
let elapsed = 0;
let countdownLeft = 0;
let gravityAcc = 0;
let lockStart = null;
let lockResets = 0;
let last = 0;
const keys = { left: null, right: null, down: false };

let net = null;
let weights = PUBLISHED_WEIGHTS;

const youCtx = setupCanvas($('youBoard'), PLAY_COLS * CELL, PLAY_ROWS * CELL);
const aiCtx = setupCanvas($('aiBoard'), PLAY_COLS * CELL, PLAY_ROWS * CELL);

function makeChooser(kind) {
  if (kind === 'bot') return (b, c, n, h) => chooseMoveAction(b, c, n, weights, h, { lookahead: true });
  if (kind === 'dqn') return (b, c, n, h) => chooseDqnAction(b, c, net, n, h);
  return (b, c, n, h) => chooseDqnAction(b, c, net, n, h, { lookahead: true });
}

/* ------------------------------------------------------------- your moves */

function grounded() {
  const a = you.active;
  return collides(you.board, rotOf(you, a.r), a.row + 1, a.col);
}

function afterMoveOnGround() {
  // sliding or turning a landed piece gives it a little more time
  if (lockStart != null && lockResets < MAX_LOCK_RESETS) {
    lockStart = performance.now();
    lockResets++;
  }
}

function tryMove(dc, dr) {
  const a = you.active;
  if (!a || collides(you.board, rotOf(you, a.r), a.row + dr, a.col + dc)) return false;
  a.row += dr;
  a.col += dc;
  if (dc) afterMoveOnGround();
  return true;
}

function tryRotate(dir) {
  const a = you.active;
  if (!a) return;
  const n = PIECES[you.cur].rotations.length;
  if (n === 1) return;
  const r = (a.r + dir + n) % n;
  const rot = rotOf(you, r);
  // simple wall kicks: try the spot, then a little left/right/up
  for (const [dr, dc] of [[0, 0], [0, -1], [0, 1], [0, -2], [0, 2], [-1, 0], [-1, -1], [-1, 1]]) {
    if (!collides(you.board, rot, a.row + dr, a.col + dc)) {
      a.r = r;
      a.row += dr;
      a.col += dc;
      afterMoveOnGround();
      return;
    }
  }
}

function lockYou() {
  const a = you.active;
  lock(you, a.r, a.row, a.col);
  lockStart = null;
  lockResets = 0;
  gravityAcc = 0;
  if (!you.dead) spawn(you);
  checkEnd();
}

function hardDrop() {
  const a = you.active;
  if (!a) return;
  while (!collides(you.board, rotOf(you, a.r), a.row + 1, a.col)) a.row++;
  lockYou();
}

function holdYou() {
  if (!you.canHold) return;
  swapHold(you);
  spawn(you);
  lockStart = null;
  lockResets = 0;
  gravityAcc = 0;
  checkEnd();
}

const ACTIONS = {
  left: () => tryMove(-1, 0),
  right: () => tryMove(1, 0),
  down: () => tryMove(0, 1),
  cw: () => tryRotate(1),
  ccw: () => tryRotate(-1),
  drop: hardDrop,
  hold: holdYou,
};

/* -------------------------------------------------------- the opponent */

function aiDecide() {
  let m = chooser(ai.board, ai.cur, ai.next, { held: ai.held, canHold: true });
  if (m && m.hold) {
    swapHold(ai);
    m = chooser(ai.board, ai.cur, ai.next, { held: ai.held, canHold: false });
  }
  if (!m || m.hold || m.score === -Infinity) return null;
  return m;
}

function aiStep(t) {
  if (ai.dead) return;
  if (aiAnim) {
    if (t - aiAnim.start >= aiAnim.dur) {
      lock(ai, aiAnim.r, aiAnim.target, aiAnim.col);
      aiAnim = null;
      checkEnd();
    }
    return;
  }
  if (t < aiNextAt) return;
  const m = aiDecide();
  if (!m) {
    ai.dead = true;
    checkEnd();
    return;
  }
  const interval = 1000 / aiPps;
  aiAnim = { r: m.rotation, col: m.col, target: m.row, start: t, dur: Math.min(220, interval * 0.6) };
  aiNextAt = t + interval;
}

/* ------------------------------------------------------------ the match */

function setOverlay(title, html) {
  $('overlayTitle').innerHTML = title;
  $('overlayText').innerHTML = html;
  $('overlay').hidden = false;
}

function start() {
  if (state === 'loading') return;
  mode = $('mode').value;
  aiPps = Number($('speed').value);
  chooser = makeChooser($('opponent').value);
  const full = $('opponent').selectedOptions[0].textContent.replace(/ \(.*\)$/, '');
  const short = { dqn2: 'Network+', dqn: 'Network', bot: 'Bot' }[$('opponent').value];
  $('aiName').innerHTML = `<span class="long">${full}</span><span class="short">${short}</span>`;
  $('goal').textContent = mode === 'race' ? `Race to ${RACE_LINES} lines` : 'Battle';
  const seed = (Math.random() * 2 ** 31) >>> 0;
  you = newPlayer(seed);
  ai = newPlayer(seed); // same pieces for both
  spawn(you);
  aiAnim = null;
  elapsed = 0;
  gravityAcc = 0;
  lockStart = null;
  lockResets = 0;
  countdownLeft = 3000;
  state = 'countdown';
  for (const id of ['youBadge', 'aiBadge']) {
    $(id).textContent = '';
    $(id).className = 'vs-badge';
  }
  $('aiOverlay').hidden = true;
  for (const id of ['mode', 'opponent', 'speed']) $(id).disabled = true;
  $('startBtn').textContent = 'Restart';
  $('pauseBtn').disabled = false;
  $('pauseBtn').textContent = 'Pause';
  // on phones, bring the boards and touch buttons into view
  if (window.matchMedia('(pointer: coarse)').matches) $('youBoard').scrollIntoView({ block: 'center', behavior: 'smooth' });
}

function setPaused(on) {
  if (on && state === 'playing') {
    state = 'paused';
    setOverlay('Paused', 'Press <b>P</b> or <b>Pause</b> to continue.');
    $('pauseBtn').textContent = 'Resume';
  } else if (!on && state === 'paused') {
    state = 'playing';
    $('overlay').hidden = true;
    $('pauseBtn').textContent = 'Pause';
  }
}

function checkEnd() {
  if (state !== 'playing' && state !== 'countdown') return;
  let winner = null;
  if (you.dead && ai.dead) winner = 'draw';
  else if (you.dead) winner = 'ai';
  else if (ai.dead) winner = 'you';
  else if (mode === 'race') {
    if (you.lines >= RACE_LINES) winner = 'you';
    else if (ai.lines >= RACE_LINES) winner = 'ai';
  }
  if (winner) endMatch(winner);
}

function fmtTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function endMatch(winner) {
  state = 'over';
  const title = winner === 'you' ? 'You win! 🎉' : winner === 'ai' ? 'The network wins' : 'Draw';
  const why =
    mode === 'race'
      ? winner === 'you'
        ? you.dead
          ? ''
          : `You reached ${RACE_LINES} lines first.`
        : ai.dead
          ? ''
          : `It reached ${RACE_LINES} lines first.`
      : '';
  const pps = (p) => (elapsed > 0 ? (p.pieces / (elapsed / 1000)).toFixed(2) : '0');
  setOverlay(
    title,
    `${why} ${winner === 'you' && ai.dead ? 'Its stack reached the top.' : winner === 'ai' && you.dead ? 'Your stack reached the top.' : ''}<br />` +
      `Time ${fmtTime(elapsed)} · you ${you.lines} lines, ${pps(you)} pieces/s` +
      (mode === 'battle' ? `, ${you.sent} garbage sent` : '') +
      `<br />Press <b>Restart</b> to play again.`
  );
  if (winner !== 'draw') {
    $(winner === 'you' ? 'youBadge' : 'aiBadge').textContent = 'Winner';
    $(winner === 'you' ? 'youBadge' : 'aiBadge').className = 'vs-badge win';
    $(winner === 'you' ? 'aiBadge' : 'youBadge').textContent = 'Topped out';
    $(winner === 'you' ? 'aiBadge' : 'youBadge').className = 'vs-badge lose';
    if (mode === 'race' && !you.dead && !ai.dead) $(winner === 'you' ? 'aiBadge' : 'youBadge').textContent = 'Second';
  }
  if (winner === 'you' && ai.dead) {
    $('aiOverlayTitle').textContent = 'Topped out';
    $('aiOverlay').hidden = false;
  }
  for (const id of ['mode', 'opponent', 'speed']) $(id).disabled = false;
  $('pauseBtn').disabled = true;
}

/* -------------------------------------------------------------- the loop */

function gravityMs() {
  const level = Math.floor(you.lines / 10);
  return Math.max(70, 800 * 0.85 ** level);
}

function handleRepeat(t) {
  for (const dir of ['left', 'right']) {
    const k = keys[dir];
    if (k && t - k.since >= DAS && t - k.last >= ARR) {
      ACTIONS[dir]();
      k.last = t;
    }
  }
}

function frame(t) {
  const dt = last ? Math.min(100, t - last) : 0;
  last = t;

  if (state === 'countdown') {
    countdownLeft -= dt;
    if (countdownLeft <= 0) {
      state = 'playing';
      $('overlay').hidden = true;
      aiNextAt = t + 300;
    } else {
      setOverlay(`<span class="big">${Math.ceil(countdownLeft / 1000)}</span>`, 'Get ready…');
    }
  } else if (state === 'playing') {
    elapsed += dt;
    handleRepeat(t);
    if (you.active) {
      gravityAcc += dt;
      const g = keys.down ? SOFT_DROP : gravityMs();
      while (gravityAcc >= g) {
        gravityAcc -= g;
        if (!tryMove(0, 1)) break;
        lockStart = null;
      }
      if (grounded()) {
        if (lockStart == null) lockStart = t;
        else if (t - lockStart >= LOCK_DELAY) lockYou();
      } else {
        lockStart = null;
      }
    }
    if (state === 'playing') aiStep(t);
  }

  render(t);
  requestAnimationFrame(frame);
}

function render(t) {
  if (!you) return;
  // your board: stack, ghost, piece
  drawBoard(youCtx, you.board);
  const a = you.active;
  if (a && !you.dead && state !== 'over') {
    let g = a.row;
    while (!collides(you.board, rotOf(you, a.r), g + 1, a.col)) g++;
    drawGhost(youCtx, you.cur, a.r, g, a.col);
    drawPiece(youCtx, you.cur, a.r, a.row, a.col);
  }
  // the opponent's board, with its piece falling into place
  drawBoard(aiCtx, ai.board);
  if (aiAnim && state !== 'over') {
    const k = Math.min(1, (t - aiAnim.start) / aiAnim.dur);
    drawPiece(aiCtx, ai.cur, aiAnim.r, Math.round(aiAnim.target * k), aiAnim.col);
  }

  for (const [p, pre] of [
    [you, 'you'],
    [ai, 'ai'],
  ]) {
    drawMini(`${pre}Hold`, p.held);
    drawMini(`${pre}Next`, p.next);
    $(`${pre}Lines`).textContent = p.lines;
    $(`${pre}Score`).textContent = p.score.toLocaleString('en-US');
    $(`${pre}Pps`).textContent = elapsed > 1000 ? (p.pieces / (elapsed / 1000)).toFixed(1) : '0.0';
    $(`${pre}Garbage`).style.height = `${Math.min(100, (p.pending / 12) * 100)}%`;
  }
  $('timer').textContent = fmtTime(elapsed);
}

/* ---------------------------------------------------------------- input */

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

window.addEventListener('keydown', (e) => {
  if (e.target.closest?.('select')) return;
  if (e.code === 'KeyP' || e.code === 'Escape') {
    if (state === 'playing') setPaused(true);
    else if (state === 'paused') setPaused(false);
    return;
  }
  const action = KEYMAP[e.code];
  if (!action || state !== 'playing') return;
  e.preventDefault(); // no page scrolling while playing
  if (e.repeat) return; // we do our own repeat for left/right/down
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
});

window.addEventListener('keyup', (e) => {
  const action = KEYMAP[e.code];
  if (action === 'left' || action === 'right') keys[action] = null;
  if (action === 'down') keys.down = false;
});

// touch buttons: tap to act; hold left/right/down to repeat
for (const btn of document.querySelectorAll('#touch button')) {
  const action = btn.dataset.key;
  let timer = null;
  const stop = () => {
    clearInterval(timer);
    timer = null;
    if (action === 'down') keys.down = false;
  };
  btn.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    if (state !== 'playing') return;
    if (action === 'down') {
      keys.down = true;
      return;
    }
    ACTIONS[action]();
    if (action === 'left' || action === 'right') {
      const t0 = performance.now();
      timer = setInterval(() => {
        if (performance.now() - t0 >= DAS && state === 'playing') ACTIONS[action]();
      }, ARR);
    }
  });
  for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) btn.addEventListener(ev, stop);
}

$('startBtn').addEventListener('click', (e) => {
  e.currentTarget.blur(); // so Space drops a piece instead of pressing the button again
  start();
});
$('pauseBtn').addEventListener('click', (e) => {
  e.currentTarget.blur();
  setPaused(state === 'playing');
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) setPaused(true);
});

/* ---------------------------------------------------------------- start */

async function init() {
  you = newPlayer(1);
  ai = newPlayer(1);
  requestAnimationFrame(frame);
  try {
    const [model, w] = await Promise.all([
      fetch('./dqn-model.json').then((r) => {
        if (!r.ok) throw new Error(`dqn-model.json: HTTP ${r.status}`);
        return r.json();
      }),
      fetch('./weights.json')
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null),
    ]);
    net = loadDqn(model);
    if (w?.weights) weights = w.weights;
    state = 'idle';
  } catch (e) {
    setOverlay('Could not load the network', `${e.message}. Open this page through a local server (npx serve .).`);
  }
}

init();
