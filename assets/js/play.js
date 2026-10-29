/**
 * play.js - the Play page (index.html): you play; "Network plays" hands the
 * same game to the network and back. The game itself is tetris-visual.js;
 * this file wires up the page around it: You play / Network plays, the
 * player list, the hint, the score and level, the overlay, the toolbar and
 * the touch buttons.
 */

import { startVisualGame } from './tetris-visual.js';
import { HOLD } from '../../src/tetris-core.js';
import { PLAYERS, DEFAULT_PLAYER, makeChooser } from '../../src/tetris-players.js';
import { loadModels } from './lib/models.js';
import { AiClient } from './lib/ai-client.js';
import { bindTouchPad } from './lib/touch-pad.js';

const $ = (id) => document.getElementById(id);
const STORAGE_KEY = 'tetris-play';
const LINES_PER_LEVEL = 10; // the game speeds up every 10 lines (gravityMs in src/tetris-controls.js)
const CLEAR_NAMES = ['', 'single', 'double', 'triple', 'Tetris!'];
const isTouch = window.matchMedia('(pointer: coarse)').matches;

/* ----------------------------------------------------------------- models */

let models;
try {
  models = await loadModels();
} catch (e) {
  // say so on the board instead of leaving a blank game that never starts
  $('playTitle').textContent = "Couldn't load the network";
  $('playText').innerHTML = '<p class="intro-sub">Check your connection, then reload the page.</p>';
  throw e;
}
const ai = new AiClient(models); // hints are worked out in the background
const chooserOf = (id) => makeChooser(id, models);

/* --------------------------------------------------------- the player list */

const playerInputs = [...document.querySelectorAll('input[name=player]')];
const selectedPlayer = () => playerInputs.find((input) => input.checked)?.value ?? DEFAULT_PLAYER;

function loadSavedPlayer() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}').agent;
  } catch {
    return undefined;
  }
}

function savePlayer() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ agent: selectedPlayer() }));
  } catch {
    // storage blocked (private mode): the choice is simply not remembered
  }
}

const saved = loadSavedPlayer();
const initialPlayer = PLAYERS[saved] ? saved : DEFAULT_PLAYER;
for (const input of playerInputs) input.checked = input.value === initialPlayer;

/* ---------------------------------------------------------------- overlay */

const READY_TEXT = isTouch
  ? '<p class="intro-sub">You play. Press <b>Network plays</b> to let the network take over.</p>'
  : `<p class="intro-sub">You play. Press <b>Network plays</b> to let the network take over.</p>
     <ol class="intro-steps">
       <li><span><kbd>←</kbd><kbd>→</kbd> move, <kbd>↑</kbd> rotate</span></li>
       <li><span><kbd>↓</kbd> soft drop, <kbd>Space</kbd> drop</span></li>
       <li><span><kbd>C</kbd> hold, <kbd>P</kbd> pause</span></li>
     </ol>`;

/** What the overlay says in each phase of your game (none: hidden). */
const OVERLAYS = {
  ready: () => ({ title: 'Ready to play?', body: READY_TEXT, button: 'Start', action: 'start' }),
  paused: () => ({
    title: 'Paused',
    body: `<p class="intro-sub">${isTouch ? 'Tap <b>Resume</b> to continue.' : 'Press <kbd>P</kbd> to continue.'}</p>`,
    button: 'Resume',
    action: 'pause',
  }),
  over: (info) => ({
    title: 'Game over',
    body:
      `<p class="intro-sub">${info.lines} lines · score ${info.score.toLocaleString('en-US')}</p>` +
      (isTouch ? '' : '<p class="intro-sub">Press <kbd>Enter</kbd> or <kbd>Space</kbd> to play again.</p>'),
    button: 'Play again',
    action: 'restart',
  }),
};

function onPhase(phase, info) {
  document.body.classList.toggle('paused', phase === 'paused');
  $('pauseBtn').setAttribute('aria-label', phase === 'paused' ? 'Resume' : 'Pause');
  const content = OVERLAYS[phase]?.(info);
  $('playOverlay').hidden = !content;
  if (!content) return;
  $('playOverlay').classList.toggle('intro-screen', phase === 'ready');
  $('playTitle').textContent = content.title;
  $('playText').innerHTML = content.body;
  $('playGo').textContent = content.button;
  $('playGo').dataset.action = content.action;
  $('playGo').hidden = false;
}

function showHint(hint) {
  // the outline is drawn on the board; this line says what it means
  $('holdTetromino').classList.toggle('hint-hold', !!hint?.hold);
  if (!hint || !$('hintToggle').checked) return;
  const who = PLAYERS[selectedPlayer()].bot ? 'The search bot' : 'The network';
  $('hintNote').textContent = hint.hold
    ? `${who} would hold this piece and keep it for later.`
    : `${who} would put it in the dashed spot.`;
}

/* ------------------------------------------------------- score and level */

/** The level line: a wavy stroke as far as you have come in this level, a flat one for the rest. */
function showLevel(lines) {
  const level = Math.floor(lines / LINES_PER_LEVEL) + 1;
  const done = lines % LINES_PER_LEVEL;
  $('levelName').textContent = `Level ${level}`;
  const left = LINES_PER_LEVEL - done;
  $('levelLeft').textContent = `${left} line${left === 1 ? '' : 's'} to level ${level + 1}`;
  const width = 292 * (done / LINES_PER_LEVEL);
  let wave = '';
  if (width > 0) {
    wave = 'M4 7';
    for (let x = 4; x <= 4 + width; x += 2) wave += ` L${x.toFixed(0)} ${(7 + 3.5 * Math.sin((x - 4) / 6)).toFixed(2)}`;
  }
  const paths = $('levelWave').querySelectorAll('path');
  paths[0].setAttribute('d', wave);
  paths[1].setAttribute('d', `M${Math.min(296, 4 + width + (width > 0 ? 10 : 0)).toFixed(0)} 7 H296`);
}

let shownLevelLines = -1;
function showStats(s) {
  $('stLines').textContent = s.lines;
  $('stPieces').textContent = s.pieces;
  $('stBest').textContent = Math.max(s.bestLines, s.lines);
  $('stGame').textContent = `Game ${s.games}` + (s.games > 1 ? ` · best ${s.bestLines} lines` : '');
  $('scoreNote').textContent = s.lastClear ? `+${s.lastClear.points} · ${CLEAR_NAMES[s.lastClear.lines]}` : '';
  if (s.lines !== shownLevelLines) {
    shownLevelLines = s.lines;
    showLevel(s.lines);
  }
}

/* ------------------------------------------------------------------- game */

const game = startVisualGame({
  chooser: chooserOf(selectedPlayer()),
  askHint: (position) => ai.decide(selectedPlayer(), position).then((r) => (r.hold ? HOLD : r.move)),
  hold: true,
  human: true,
  msPerRow: 45,
  onState: onPhase,
  onHint: showHint,
  onStats: showStats,
});

/* --------------------------------------------------------------- controls */

const networkPlays = () => document.body.classList.contains('human') === false;

/** You play / Network plays: the same game carries on with the other player. */
function setNetworkPlays(on) {
  document.body.classList.toggle('human', !on);
  $('youBtn').setAttribute('aria-pressed', String(!on));
  $('netBtn').setAttribute('aria-pressed', String(on));
  if (game.isHuman() === on) game.setHuman(!on);
}

function setHintOn(on) {
  $('hintToggle').checked = on;
  $('hintNote').textContent = on ? 'Thinking…' : 'Turn it on to see where the network would put your piece.';
  game.setHint(on);
}

for (const [id, on] of [
  ['youBtn', false],
  ['netBtn', true],
]) {
  $(id).addEventListener('click', (e) => {
    e.currentTarget.blur(); // so Space and the arrows go to the game, not the button
    setNetworkPlays(on);
  });
}

$('hintToggle').addEventListener('change', (e) => {
  e.target.blur();
  setHintOn(e.target.checked);
});

window.addEventListener('keydown', (e) => {
  if (e.code !== 'KeyH' || e.repeat || networkPlays()) return;
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.('select, input, textarea, button')) return;
  setHintOn(!$('hintToggle').checked);
});

for (const input of playerInputs) {
  input.addEventListener('change', () => {
    input.blur();
    game.setChooser(chooserOf(input.value));
    savePlayer();
  });
}

$('playGo').addEventListener('click', (e) => {
  e.currentTarget.blur();
  game.press(e.currentTarget.dataset.action);
  if (isTouch) $('gameContainer').scrollIntoView({ block: 'center', behavior: 'smooth' });
});

$('pauseBtn').addEventListener('click', (e) => {
  e.currentTarget.blur();
  game.press('pause');
});
$('restartBtn').addEventListener('click', (e) => {
  e.currentTarget.blur();
  game.restart();
});

document.addEventListener('visibilitychange', () => {
  if (document.hidden) game.pause(true);
});

bindTouchPad($('touchPad'), { press: (action) => game.press(action), setSoftDrop: (on) => game.softDrop(on) });

showLevel(0);
