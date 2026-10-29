/**
 * palette.js - the visitor's choice of colours: one of three palettes from
 * Figma's colour combinations (Ink wash, Spiced chai, Retro sunset). Each
 * palette sets every colour of the site, pieces included (see the palette
 * blocks in assets/css/theme.css); this picks which one.
 *
 * Loaded as a plain script at the top of <head> (not a module, not deferred),
 * so the saved palette is on <html data-palette="..."> before the page is
 * drawn: no flash of the wrong colours. It then adds the picker (a round
 * button and a small menu) to the top bar.
 *
 * Pages that draw on <canvas> listen for the 'palettechange' event on
 * document and redraw.
 */
(function () {
  const STORAGE_KEY = 'tetris-palette';
  const PALETTES = [
    { id: 'ink', name: 'Ink wash' },
    { id: 'chai', name: 'Spiced chai' },
    { id: 'retro', name: 'Retro sunset' },
  ];
  const DEFAULT = PALETTES[0].id;
  const root = document.documentElement;
  const isPalette = (id) => PALETTES.some((p) => p.id === id);

  function saved() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch {
      return null; // storage blocked (private mode): the default colours
    }
  }

  root.dataset.palette = isPalette(saved()) ? saved() : DEFAULT;

  function setPalette(id) {
    if (!isPalette(id) || id === root.dataset.palette) return;
    root.dataset.palette = id;
    try {
      localStorage.setItem(STORAGE_KEY, id);
    } catch {
      // storage blocked: the choice lasts until the page is left
    }
    document.dispatchEvent(new CustomEvent('palettechange', { detail: id }));
  }

  const swatch = (id) => `<span class="palette-swatch" data-palette="${id}" aria-hidden="true"></span>`;

  function buildPicker() {
    const end = document.querySelector('.topnav-end');
    if (!end) return;
    const picker = document.createElement('div');
    picker.className = 'palette-picker';
    picker.innerHTML = `
      <button type="button" class="palette-btn" aria-expanded="false" aria-controls="paletteMenu" aria-label="Colours">
        <span class="palette-swatch" aria-hidden="true"></span>
      </button>
      <div class="palette-menu" id="paletteMenu" role="group" aria-label="Colours" hidden>
        <p class="palette-title">Colours</p>
        ${PALETTES.map((p) => `<button type="button" data-value="${p.id}" aria-pressed="false">${swatch(p.id)}${p.name}</button>`).join('')}
      </div>`;
    end.prepend(picker);

    const button = picker.querySelector('.palette-btn');
    const menu = picker.querySelector('.palette-menu');
    const options = [...menu.querySelectorAll('button')];
    const showChoice = () => {
      for (const o of options) o.setAttribute('aria-pressed', String(o.dataset.value === root.dataset.palette));
    };
    const open = (on) => {
      menu.hidden = !on;
      button.setAttribute('aria-expanded', String(on));
      if (on) {
        showChoice();
        (options.find((o) => o.dataset.value === root.dataset.palette) ?? options[0]).focus();
      }
    };

    button.addEventListener('click', () => open(menu.hidden));
    for (const o of options) {
      // stays open, so the colours can be compared one after another
      o.addEventListener('click', () => {
        setPalette(o.dataset.value);
        showChoice();
      });
    }
    document.addEventListener('click', (e) => {
      if (!menu.hidden && !picker.contains(e.target)) open(false);
    });
    picker.addEventListener('keydown', (e) => {
      // keys pressed here are for the picker, not the game (Space, arrows, Esc = pause)
      e.stopPropagation();
      if (e.key === 'Escape' && !menu.hidden) {
        open(false);
        button.focus();
      }
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', buildPicker);
  else buildPicker();
})();
