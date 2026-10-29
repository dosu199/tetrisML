/**
 * touch-pad.js - the on-screen buttons for phones and tablets
 * (<button data-key="left|right|down|cw|ccw|drop|hold|pause">).
 *
 * Tap to act. Holding left or right repeats like a held key (after
 * TIMING.das, every TIMING.arr ms); holding down soft-drops until released.
 */

import { TIMING } from '../../../src/tetris-controls.js';

/**
 * @param {Element} root  the element that holds the buttons
 * @param {object} handlers
 * @param {(action: string) => void} handlers.press      one tap (and each repeat)
 * @param {(on: boolean) => void} handlers.setSoftDrop  down is held (true) / released (false)
 * @param {() => boolean} [handlers.isEnabled]  buttons do nothing while this is false
 */
export function bindTouchPad(root, { press, setSoftDrop, isEnabled = () => true }) {
  for (const button of root.querySelectorAll('button[data-key]')) {
    const action = button.dataset.key;
    let repeat = null;
    const release = () => {
      clearInterval(repeat);
      repeat = null;
      if (action === 'down') setSoftDrop(false);
    };
    button.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      release(); // a second finger on the same button must not leave a repeat running
      if (!isEnabled()) return;
      if (action === 'down') {
        setSoftDrop(true);
        return;
      }
      press(action);
      if (action === 'left' || action === 'right') {
        const pressedAt = performance.now();
        repeat = setInterval(() => {
          if (performance.now() - pressedAt >= TIMING.das && isEnabled()) press(action);
        }, TIMING.arr);
      }
    });
    for (const type of ['pointerup', 'pointerleave', 'pointercancel']) button.addEventListener(type, release);
  }
}
