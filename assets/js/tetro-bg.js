/*
 * tetro-bg.js - slowly falling tetrominoes, all 7 kinds, as decoration, in
 * the palette's four colours (--pal-1..4; see theme.css).
 *
 *  - a subtle layer behind every page (fixed, behind all content; boards
 *    and cards cover it, so none fall over the game itself)
 *
 * Pure decoration: nothing reads it, nothing depends on it. Turned off for
 * people who ask their system for reduced motion. To tune it: OPACITY and the
 * piece counts in start() below.
 */
(() => {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // the 7 tetrominoes as [col, row] cells, in turn in the palette's four colours
  const SHAPES = [
    { cells: [[0, 0], [1, 0], [1, 1], [2, 1]], color: 'var(--pal-1)' }, // Z
    { cells: [[0, 0], [1, 0], [2, 0], [3, 0]], color: 'var(--pal-2)' }, // I
    { cells: [[0, 0], [1, 0], [0, 1], [1, 1]], color: 'var(--pal-3)' }, // O
    { cells: [[1, 0], [2, 0], [0, 1], [1, 1]], color: 'var(--pal-4)' }, // S
    { cells: [[2, 0], [0, 1], [1, 1], [2, 1]], color: 'var(--pal-1)' }, // L
    { cells: [[0, 0], [0, 1], [1, 1], [2, 1]], color: 'var(--pal-2)' }, // J
    { cells: [[1, 0], [0, 1], [1, 1], [2, 1]], color: 'var(--pal-3)' }, // T
  ];

  function pieceSvg(shape) {
    const w = Math.max(...shape.cells.map(([c]) => c)) + 1;
    const h = Math.max(...shape.cells.map(([, r]) => r)) + 1;
    const rects = shape.cells
      .map(([c, r]) => `<rect x="${c * 10 + 0.6}" y="${r * 10 + 0.6}" width="8.8" height="8.8" rx="1.6"/>`)
      .join('');
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w * 10} ${h * 10}" width="${w}em" height="${h}em" style="fill: ${shape.color}">${rects}</svg>`;
  }

  const rand = (a, b) => a + Math.random() * (b - a);

  /** Fills `layer` with `count` pieces; each piece cycles through all 7 kinds. */
  function fill(layer, count, { size, speed }) {
    layer.innerHTML = '';
    for (let i = 0; i < count; i++) {
      const el = document.createElement('span');
      el.className = 'tetro';
      const dur = rand(speed[0], speed[1]);
      el.style.left = `${((i + rand(0.1, 0.9)) / count) * 100}%`; // spread over the width
      el.style.fontSize = `${rand(size[0], size[1])}px`; // 1em = one cell
      el.style.setProperty('--r', `${[0, 90, 180, 270][i % 4]}deg`);
      el.style.animationDuration = `${dur}s`;
      el.style.animationDelay = `${-rand(0, dur)}s`; // already spread out on load
      el.innerHTML = pieceSvg(SHAPES[i % SHAPES.length]);
      // a new random kind every time it starts falling again
      el.addEventListener('animationiteration', () => {
        el.innerHTML = pieceSvg(SHAPES[Math.floor(Math.random() * SHAPES.length)]);
      });
      layer.appendChild(el);
    }
  }

  // The styles live here, not in a CSS file, so the pieces can never show up
  // unstyled (for example when the browser still has an older CSS cached).
  const OPACITY = 0.16; // how visible the pieces are (0 = invisible, 1 = full colour)
  const CSS = `
    .tetro-bg { position: fixed; inset: 0; z-index: -1; overflow: hidden; pointer-events: none; }
    .tetro {
      position: absolute; top: 0; line-height: 0; will-change: transform;
      animation-name: tetro-drop; animation-timing-function: linear; animation-iteration-count: infinite;
    }
    .tetro svg { display: block; }
    .tetro-bg .tetro { opacity: ${OPACITY}; }
    @keyframes tetro-drop {
      from { transform: translateY(-120px) rotate(var(--r, 0deg)); }
      to { transform: translateY(calc(100vh + 120px)) rotate(var(--r, 0deg)); }
    }`;

  function start() {
    const style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);

    const page = document.createElement('div');
    page.className = 'tetro-bg';
    page.setAttribute('aria-hidden', 'true');
    document.body.prepend(page);
    // about one piece per 45px of screen width: 42 on a laptop, 9 on a phone
    const count = Math.max(9, Math.min(60, Math.round(window.innerWidth / 45)));
    fill(page, count, { size: [12, 22], speed: [16, 32] });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
