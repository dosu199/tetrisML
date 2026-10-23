/*
 * tetro-bg.js - slowly falling tetrominoes, all 7 kinds, as decoration.
 *
 *  - a faint layer behind every page (fixed, behind all content)
 *  - the same pieces, a little brighter, on the intro screens (.intro-fall)
 *
 * Pure decoration: nothing reads it, nothing depends on it. Turned off for
 * people who ask their system for reduced motion.
 */
(() => {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  // the 7 tetrominoes as [col, row] cells, in the game's colours
  const SHAPES = [
    { cells: [[0, 0], [1, 0], [1, 1], [2, 1]], color: '#e6483d' }, // Z
    { cells: [[0, 0], [1, 0], [2, 0], [3, 0]], color: '#4ec3e0' }, // I
    { cells: [[0, 0], [1, 0], [0, 1], [1, 1]], color: '#f2c94c' }, // O
    { cells: [[1, 0], [2, 0], [0, 1], [1, 1]], color: '#5fc27e' }, // S
    { cells: [[2, 0], [0, 1], [1, 1], [2, 1]], color: '#f2994a' }, // L
    { cells: [[0, 0], [0, 1], [1, 1], [2, 1]], color: '#4a7ef2' }, // J
    { cells: [[1, 0], [0, 1], [1, 1], [2, 1]], color: '#9b51e0' }, // T
  ];

  function pieceSvg(shape) {
    const w = Math.max(...shape.cells.map(([c]) => c)) + 1;
    const h = Math.max(...shape.cells.map(([, r]) => r)) + 1;
    const rects = shape.cells
      .map(([c, r]) => `<rect x="${c * 10 + 0.6}" y="${r * 10 + 0.6}" width="8.8" height="8.8" rx="1.6"/>`)
      .join('');
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${w * 10} ${h * 10}" width="${w}em" height="${h}em" fill="${shape.color}">${rects}</svg>`;
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

  function start() {
    const page = document.createElement('div');
    page.className = 'tetro-bg';
    page.setAttribute('aria-hidden', 'true');
    document.body.prepend(page);
    const narrow = window.innerWidth < 720;
    fill(page, narrow ? 9 : 16, { size: [12, 22], speed: [18, 34] });

    for (const layer of document.querySelectorAll('.intro-fall')) fill(layer, 7, { size: [9, 14], speed: [7, 11] });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
