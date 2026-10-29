/**
 * versus.js - "Play vs Bot": you against the network (or the search bot).
 *
 * Both boards get exactly the same pieces (same seed); both see the next
 * piece and may hold. Two modes:
 *   battle  clearing 2 / 3 / 4 lines sends 1 / 2 / 4 garbage rows to the other
 *           board; your own clears cancel garbage coming at you. Top out = lose.
 *   race    first to 40 lines wins; topping out loses.
 *
 * Your piece is moved by src/tetris-controls.js (the same controls as the
 * Play page). The opponent decides with the real players
 * (src/tetris-players.js) in a background worker, at the speed you pick.
 * Both boards are drawn on <canvas>.
 */

import { createBoard, PIECES, Bag, placePiece, clearLines, addGarbageRow, placements, mulberry32, PLAY_LEFT, PLAY_COLS, PLAY_ROWS } from '../../src/tetris-core.js';
import { PieceController, TIMING, KEYMAP, LINE_POINTS, actionForKey, gravityMs, landingRow } from '../../src/tetris-controls.js';
import { PLAYERS } from '../../src/tetris-players.js';
import { setupCanvas, drawPlayfield, drawBlock, blockColor, outlineCells } from './lib/canvas.js';
import { loadModels } from './lib/models.js';
import { AiClient } from './lib/ai-client.js';
import { bindTouchPad } from './lib/touch-pad.js';

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ rules */

const ATTACK = [0, 0, 1, 2, 4]; // garbage rows sent for clearing 0..4 lines
const RACE_LINES = 40;
const GARBAGE_SEED_OFFSET = 777; // garbage holes: their own random stream, the same on both boards
const COUNTDOWN_MS = 3000;
const AI_FIRST_MOVE_MS = 300; // after the countdown
const AI_DROP_MS = 220; // the opponent's piece falls into place in this long at most...
const AI_DROP_SHARE = 0.6; // ...and in at most this share of its turn
const GARBAGE_BAR_ROWS = 12; // incoming rows that fill the red bar
const SNACK_MS = 3500; // how long a message at the bottom stays

/* --------------------------------------------------------------- drawing */

const CELL = 26;
const MINI = { w: 64, h: 40, cell: 14 }; // the hold and next boxes

/** The cells of piece `piece` in rotation `r` at (row, col), as [x, y] on the playfield. */
const pieceCells = (piece, r, row, col) => PIECES[piece].rotations[r].cells.map(([dr, dc]) => [col + dc - PLAY_LEFT, row + dr]);

function drawPiece(ctx, piece, r, row, col) {
  const color = blockColor(PIECES[piece].value);
  for (const [x, y] of pieceCells(piece, r, row, col)) drawBlock(ctx, x, y, CELL, color);
}

function drawGhost(ctx, piece, r, row, col) {
  outlineCells(ctx, pieceCells(piece, r, row, col), CELL, 'rgba(230, 224, 233, 0.4)', 2);
}

const miniContexts = {};
function drawMini(id, piece) {
  miniContexts[id] ??= setupCanvas($(id), MINI.w, MINI.h);
  const ctx = miniContexts[id];
  ctx.clearRect(0, 0, MINI.w, MINI.h);
  if (piece == null) return;
  const rot = PIECES[piece].rotations[0];
  ctx.save();
  ctx.translate((MINI.w - rot.width * MINI.cell) / 2, (MINI.h - rot.height * MINI.cell) / 2);
  for (const [r, c] of rot.cells) drawBlock(ctx, c, r, MINI.cell, blockColor(rot.value));
  ctx.restore();
}

const boardContexts = {
  you: setupCanvas($('youBoard'), PLAY_COLS * CELL, PLAY_ROWS * CELL),
  ai: setupCanvas($('aiBoard'), PLAY_COLS * CELL, PLAY_ROWS * CELL),
};

/* ---------------------------------------------------------------- players */

function newPlayer(seed) {
  const bag = new Bag(seed);
  return {
    board: createBoard(),
    bag,
    cur: bag.next(),
    next: bag.next(),
    held: null,
    canHold: true,
    lines: 0,
    score: 0,
    pieces: 0,
    tetrises: 0,
    sent: 0,
    pending: 0, // garbage rows on their way to this board
    dead: false,
    holeRand: mulberry32(seed + GARBAGE_SEED_OFFSET),
  };
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

/**
 * Battle garbage after `p` cleared `cleared` lines: the clear first cancels
 * garbage coming at p, the rest goes to the other board. A piece that clears
 * nothing lets the waiting garbage in (all rows with the same hole).
 */
function exchangeGarbage(p, other, cleared) {
  let attack = ATTACK[cleared];
  const cancel = Math.min(attack, p.pending);
  p.pending -= cancel;
  attack -= cancel;
  if (attack > 0) {
    other.pending += attack;
    p.sent += attack;
    if (other === match.you) showSnack(`${opponentTitle()} sent you ${rows(attack)}. Clear lines to cancel them.`);
  }
  if (cleared === 0 && p.pending > 0) {
    const hole = Math.floor(p.holeRand() * PLAY_COLS);
    for (let i = 0; i < p.pending && !p.dead; i++) if (!addGarbageRow(p.board, hole)) p.dead = true;
    p.pending = 0;
  }
}

/** Places p's piece for good, scores it, sends and receives garbage, and moves on to the next piece. */
function lockPiece(p, r, row, col) {
  placePiece(p.board, PIECES[p.cur].rotations[r], row, col);
  const cleared = clearLines(p.board);
  p.pieces++;
  p.lines += cleared;
  p.score += LINE_POINTS[cleared];
  if (cleared === 4) p.tetrises++;
  if (match.mode === 'battle') exchangeGarbage(p, p === match.you ? match.ai : match.you, cleared);
  p.cur = p.next;
  p.next = p.bag.next();
  p.canHold = true;
}

/** "1 row", "3 rows". */
const rows = (n) => `${n} row${n === 1 ? '' : 's'}`;

let snackTimer = null;
/** A short message at the bottom of the screen, gone after a few seconds. */
function showSnack(text) {
  const snack = $('snack');
  snack.textContent = text;
  snack.classList.remove('is-hidden');
  clearTimeout(snackTimer);
  snackTimer = setTimeout(() => snack.classList.add('is-hidden'), SNACK_MS);
}

function hideSnack() {
  clearTimeout(snackTimer);
  $('snack').classList.add('is-hidden');
}

/* ------------------------------------------------------------------ state */

let state = 'loading'; // loading | idle | countdown | playing | paused | over
let matchId = 0; // bumps on every new match, so a late answer from the opponent is dropped
let models = null;
let aiClient = null; // asks the opponent for its moves (lib/ai-client.js)

/** The opponent: can change at any time, even mid-match. */
const opponent = { player: 'dqn3', pps: 1, mistakes: 0 }; // mistakes: chance of a random drop (Super easy)

/**
 * The match in progress (or the last one). Times are on the game clock
 * (`elapsed`, ms of play), which stops while paused.
 */
let match = {
  mode: 'battle',
  you: newPlayer(1),
  ai: newPlayer(1),
  elapsed: 0,
  countdownLeft: 0,
  aiAnim: null, // the opponent's piece falling into place: { r, col, target, start, dur }
  aiNextAt: 0, // when the opponent may place its next piece
  aiThinking: false,
};

const controller = new PieceController({
  board: () => match.you.board,
  piece: () => match.you.cur,
  lock: (r, row, col) => {
    lockPiece(match.you, r, row, col);
    if (!match.you.dead) spawnYourPiece();
    checkEnd();
  },
  hold: () => {
    if (!match.you.canHold) return;
    swapHold(match.you);
    spawnYourPiece();
    checkEnd();
  },
});

function spawnYourPiece() {
  if (!controller.spawn()) match.you.dead = true;
}

/* ------------------------------------------------------------ the opponent */

const positionOf = (p, canHold) => ({ board: p.board, cur: p.cur, next: p.next, held: p.held, canHold });

/** The opponent's placement, or null when it has no legal move. May hold first. */
async function opponentMove(p, id) {
  let reply = await aiClient.decide(opponent.player, positionOf(p, true));
  if (id !== matchId) return null;
  if (reply.hold) {
    swapHold(p);
    reply = await aiClient.decide(opponent.player, positionOf(p, false));
    if (reply.hold) return null;
  }
  return reply.move;
}

function opponentStep() {
  const m = match;
  if (m.ai.dead) return;
  if (m.aiAnim) {
    if (m.elapsed - m.aiAnim.start >= m.aiAnim.dur) {
      lockPiece(m.ai, m.aiAnim.r, m.aiAnim.target, m.aiAnim.col);
      m.aiAnim = null;
      checkEnd();
    }
    return;
  }
  if (m.aiThinking || m.elapsed < m.aiNextAt) return;
  m.aiThinking = true;
  const id = matchId;
  const askedAt = m.elapsed; // thinking time counts towards its turn
  opponentMove(m.ai, id).then((move) => {
    if (id !== matchId) return; // a new match started meanwhile
    m.aiThinking = false;
    if (state === 'over') return;
    if (!move) {
      m.ai.dead = true;
      checkEnd();
      return;
    }
    let chosen = move;
    if (opponent.mistakes && Math.random() < opponent.mistakes) {
      // Super easy: now and then it drops the piece in a random spot
      const all = placements(m.ai.board, m.ai.cur);
      if (all.length) chosen = all[Math.floor(Math.random() * all.length)];
    }
    const interval = 1000 / opponent.pps;
    m.aiAnim = { r: chosen.rotation, col: chosen.col, target: chosen.row, start: m.elapsed, dur: Math.min(AI_DROP_MS, interval * AI_DROP_SHARE) };
    m.aiNextAt = askedAt + interval;
  });
}

/* --------------------------------------------------------------- settings */

const checkedValue = (name) => document.querySelector(`input[name=${name}]:checked`).value;

/* Opponent and speed can change at any time, even mid-match (the bar above
   the boards): the next piece it places uses the new settings. The mode is
   chosen on the setup card and changes only with a new match. */
function applyOpponent() {
  const speed = $('speed');
  opponent.pps = Number(speed.value);
  opponent.mistakes = Number(speed.selectedOptions[0].dataset.mistakes || 0);
  opponent.player = $('opponent').value;
  match.aiNextAt = Math.min(match.aiNextAt, match.elapsed + 1000 / opponent.pps); // a faster speed kicks in at once
  const { name, short } = PLAYERS[opponent.player];
  $('aiName').innerHTML = `<span class="long">${name}</span><span class="short">${short}</span>`;
}

/* the setup card's radio buttons and the bar's dropdowns always agree */
for (const name of ['opponent', 'speed']) {
  for (const radio of document.querySelectorAll(`input[name=${name}]`)) {
    radio.addEventListener('change', () => {
      $(name).value = radio.value;
      applyOpponent();
    });
  }
  $(name).addEventListener('change', (e) => {
    document.querySelector(`input[name=${name}][value="${e.target.value}"]`).checked = true;
    applyOpponent();
    e.target.blur(); // arrow keys go back to your piece, not the dropdown
  });
}

/* ---------------------------------------------------------------- overlay */

const OVERLAY_ACTIONS = {
  rematch: () => start(),
  restart: () => start(),
  settings: () => backToSetup(),
  resume: () => setPaused(false),
};

const button = (action, label, primary = false) =>
  `<button type="button" data-action="${action}"${primary ? ' class="btn-go"' : ''}>${label}</button>`;

/** The message on your board: a title (may contain markup) and its content. */
function setOverlay(titleHtml, html) {
  $('overlay').classList.remove('intro-screen');
  $('overlayTitle').innerHTML = titleHtml;
  $('overlayText').innerHTML = html;
  $('overlay').hidden = false;
}

function syncPauseButton() {
  $('pauseBtn').textContent = state === 'paused' ? 'Resume' : 'Pause';
  $('pauseBtn').disabled = state === 'over' || state === 'idle';
}

/** Asks on your board before a running match is thrown away (restart, or back to the settings). */
function confirmOnBoard(title, yesLabel, yesAction) {
  if (state === 'playing' || state === 'countdown') {
    state = 'paused';
    controller.releaseAll();
  }
  setOverlay(
    title,
    '<p class="intro-sub">The current match will be lost.</p>' +
      `<div class="vs-confirm">${button(yesAction, yesLabel, true)}${button('resume', 'Keep playing')}</div>`
  );
  syncPauseButton();
  $('overlay').querySelector('[data-action=resume]').focus();
}

let shownCountdown = 0; // the number on the overlay, so it is rewritten only when it changes
function showCountdown() {
  const seconds = Math.ceil(match.countdownLeft / 1000);
  if (seconds === shownCountdown && !$('overlay').hidden) return;
  shownCountdown = seconds;
  setOverlay(
    `<span class="big">${seconds}</span>`,
    '<p class="intro-sub">Get ready…</p>' +
      (match.mode === 'battle' ? '<p class="vs-tip">The red bar beside a board shows garbage on its way.</p>' : '')
  );
}

/* -------------------------------------------------------------- the match */

/* the two screens: the setup card before a match, the boards during one */
function showMatchView(on) {
  $('lobby').hidden = on;
  $('matchView').hidden = !on;
}

function backToSetup() {
  matchId++; // drops any answer the opponent is still thinking about
  state = 'idle';
  controller.releaseAll();
  $('overlay').hidden = true;
  showMatchView(false);
  $('startBtn').focus();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

const matchRunning = () => state === 'countdown' || state === 'playing' || state === 'paused';

function start() {
  if (state === 'loading') return;
  matchId++;
  const seed = (Math.random() * 2 ** 31) >>> 0;
  match = {
    mode: checkedValue('mode'),
    you: newPlayer(seed),
    ai: newPlayer(seed), // the same pieces for both
    elapsed: 0,
    countdownLeft: COUNTDOWN_MS,
    aiAnim: null,
    aiNextAt: 0,
    aiThinking: false,
  };
  applyOpponent();
  controller.releaseAll();
  spawnYourPiece();
  state = 'countdown';
  shownCountdown = 0;
  showMatchView(true);
  const modeName = match.mode === 'race' ? `Race to ${RACE_LINES} lines` : 'Battle';
  $('barMode').textContent = modeName;
  $('goal').textContent = match.mode === 'race' ? `First to ${RACE_LINES} lines wins. Topping out loses.` : 'Clear 2 or more lines at once to send garbage.';
  hideSnack();
  for (const id of ['youBadge', 'aiBadge']) {
    $(id).textContent = '';
    $(id).className = 'vs-badge';
  }
  $('aiOverlay').hidden = true;
  syncPauseButton();
  // bring the boards (and on phones the touch buttons) into view
  $('matchView').scrollIntoView({ block: 'start', behavior: 'smooth' });
}

function setPaused(on) {
  if (on && (state === 'countdown' || state === 'playing')) {
    const wasCounting = state === 'countdown';
    state = 'paused';
    controller.releaseAll();
    setOverlay(
      'Paused',
      wasCounting
        ? '<p class="intro-sub">Tap <b>Resume</b> to continue.</p>'
        : '<p class="intro-sub"><span class="keys-only">Press <kbd>P</kbd> or </span>Tap <b>Resume</b> to continue.</p>'
    );
  } else if (!on && state === 'paused') {
    // paused during the countdown: carry on counting
    state = match.countdownLeft > 0 ? 'countdown' : 'playing';
    if (state === 'playing') $('overlay').hidden = true;
    else shownCountdown = 0; // redraw the countdown over the pause message
  }
  syncPauseButton();
}

function checkEnd() {
  if (state !== 'playing' && state !== 'countdown') return;
  const { you, ai, mode } = match;
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

function formatTime(ms) {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** "The network" or "The bot": who you are playing against. */
const opponentTitle = () => (PLAYERS[opponent.player].bot ? 'The bot' : 'The network');

/** The result's title, and what decided it: a full stack, or the race finish line. */
function resultText(winner) {
  const { you, ai } = match;
  if (winner === 'you') return { title: 'You win! 🎉', why: ai.dead ? 'Its stack reached the top.' : `You reached ${RACE_LINES} lines first.` };
  if (winner === 'ai') {
    return {
      title: `${opponentTitle()} wins`,
      why: you.dead ? 'Your stack reached the top.' : `It reached ${RACE_LINES} lines first.`,
    };
  }
  return { title: 'Draw', why: 'Both stacks reached the top.' };
}

function endMatch(winner) {
  state = 'over';
  controller.releaseAll();
  const { you, ai, mode, elapsed } = match;
  const { title, why } = resultText(winner);
  const pps = (p) => (elapsed > 0 ? (p.pieces / (elapsed / 1000)).toFixed(2) : '0');
  setOverlay(
    title,
    `${why}<br />` +
      `Time ${formatTime(elapsed)} · you ${you.lines} lines, ${pps(you)} pieces/s` +
      (mode === 'battle' ? `, ${you.sent} garbage sent` : '') +
      `<div class="vs-confirm">${button('rematch', 'Rematch', true)}${button('settings', 'Change settings')}</div>` +
      '<span class="vs-rematch-hint keys-only">or press <kbd>Enter</kbd> for a rematch</span>'
  );
  if (winner !== 'draw') {
    const winnerBadge = $(winner === 'you' ? 'youBadge' : 'aiBadge');
    const loserBadge = $(winner === 'you' ? 'aiBadge' : 'youBadge');
    winnerBadge.textContent = 'Winner';
    winnerBadge.className = 'vs-badge win';
    loserBadge.textContent = mode === 'race' && !you.dead && !ai.dead ? 'Second' : 'Topped out';
    loserBadge.className = 'vs-badge lose';
  }
  if (winner === 'you' && ai.dead) {
    $('aiOverlayTitle').textContent = 'Topped out';
    $('aiOverlay').hidden = false;
  }
  syncPauseButton();
}

/* -------------------------------------------------------------- the loop */

let lastFrame = 0;

function frame(now) {
  const dt = lastFrame ? Math.min(TIMING.maxFrame, now - lastFrame) : 0;
  lastFrame = now;

  if (state === 'countdown') {
    match.countdownLeft -= dt;
    if (match.countdownLeft <= 0) {
      state = 'playing';
      $('overlay').hidden = true;
      match.aiNextAt = match.elapsed + AI_FIRST_MOVE_MS;
    } else {
      showCountdown();
    }
  } else if (state === 'playing') {
    match.elapsed += dt;
    controller.step(now, dt, gravityMs(match.you.lines));
    if (state === 'playing') opponentStep();
  }

  if (!$('matchView').hidden) render();
  requestAnimationFrame(frame);
}

/** Draws both boards and their numbers. */
function render() {
  const { you, ai, aiAnim, elapsed } = match;
  const drawBoard = (ctx, board) => drawPlayfield(ctx, board, { cell: CELL });

  // your board: stack, ghost, piece
  drawBoard(boardContexts.you, you.board);
  const a = controller.active;
  if (a && !you.dead && state !== 'over') {
    const rotation = PIECES[you.cur].rotations[a.r];
    drawGhost(boardContexts.you, you.cur, a.r, landingRow(you.board, rotation, a.row, a.col), a.col);
    drawPiece(boardContexts.you, you.cur, a.r, a.row, a.col);
  }
  // the opponent's board, with its piece falling into place
  drawBoard(boardContexts.ai, ai.board);
  if (aiAnim && state !== 'over') {
    const k = Math.min(1, (elapsed - aiAnim.start) / aiAnim.dur);
    drawPiece(boardContexts.ai, ai.cur, aiAnim.r, Math.round(aiAnim.target * k), aiAnim.col);
  }

  for (const [p, prefix] of [
    [you, 'you'],
    [ai, 'ai'],
  ]) {
    drawMini(`${prefix}Hold`, p.held);
    drawMini(`${prefix}Next`, p.next);
    setText(`${prefix}Lines`, p.lines);
    setText(`${prefix}Score`, p.score.toLocaleString('en-US'));
    setText(`${prefix}Pps`, elapsed > 1000 ? (p.pieces / (elapsed / 1000)).toFixed(1) : '0.0');
    $(`${prefix}Garbage`).style.height = `${Math.min(100, (p.pending / GARBAGE_BAR_ROWS) * 100)}%`;
    setText(`${prefix}Sent`, rows(p.sent));
  }
  const incoming = $('youIncoming');
  incoming.hidden = you.pending === 0 || state === 'over';
  setText('youIncoming', `${rows(you.pending)} incoming`);
  setText('timer', formatTime(elapsed));
}

/** Writes text only when it changed (no DOM work for numbers that stay the same). */
function setText(id, value) {
  const el = $(id);
  const text = String(value);
  if (el.textContent !== text) el.textContent = text;
}

/* ---------------------------------------------------------------- input */

window.addEventListener('keydown', (e) => {
  if (e.target.closest?.('select')) return;
  if (e.code === 'Enter' && state === 'over' && !e.target.closest?.('button')) {
    e.preventDefault();
    start(); // rematch with the same settings
    return;
  }
  if (e.code === 'KeyP' || e.code === 'Escape') {
    if (state === 'playing') setPaused(true);
    else if (state === 'paused') setPaused(false);
    return;
  }
  const action = actionForKey(e);
  if (!action || state !== 'playing') return;
  e.preventDefault(); // no page scrolling while playing
  if (e.repeat) return; // the controller repeats left/right itself
  controller.keyDown(action);
});

window.addEventListener('keyup', (e) => {
  const action = KEYMAP[e.code]; // not actionForKey: a key released while Ctrl is down must still let go
  if (action) controller.keyUp(action);
});
window.addEventListener('blur', () => controller.releaseAll()); // keyup never arrives in a background window

bindTouchPad($('touch'), {
  press: (action) => controller.run(action),
  setSoftDrop: (on) => controller.setSoftDrop(on),
  isEnabled: () => state === 'playing',
});

$('overlay').addEventListener('click', (e) => {
  const action = e.target.closest?.('button[data-action]')?.dataset.action;
  OVERLAY_ACTIONS[action]?.();
});

$('startBtn').addEventListener('click', (e) => {
  e.currentTarget.blur(); // so Space drops a piece instead of pressing the button again
  start();
});
$('restartBtn').addEventListener('click', (e) => {
  e.currentTarget.blur();
  if (matchRunning()) confirmOnBoard('Restart this match?', 'Restart', 'restart');
  else start();
});
$('settingsBtn').addEventListener('click', (e) => {
  e.currentTarget.blur();
  if (matchRunning()) confirmOnBoard('Leave this match?', 'Change settings', 'settings');
  else backToSetup();
});
$('pauseBtn').addEventListener('click', (e) => {
  e.currentTarget.blur();
  setPaused(state === 'playing' || state === 'countdown');
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden) setPaused(true);
});

/* ---------------------------------------------------------------- start */

async function init() {
  applyOpponent();
  requestAnimationFrame(frame);
  try {
    models = await loadModels();
    aiClient = new AiClient(models);
    state = 'idle';
    $('startBtn').disabled = false;
    $('startBtn').textContent = 'Start';
  } catch (e) {
    $('startBtn').textContent = 'Could not load the network';
    $('lobbyError').textContent = `${e.message}. Check your connection and reload the page.`;
    $('lobbyError').hidden = false;
  }
}

init();
