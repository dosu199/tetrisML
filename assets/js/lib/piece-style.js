/**
 * piece-style.js - the colours of the pieces on the canvas boards, by cell
 * value (Z=1, I=2, O=3, S=4, L=5, J=6, T=7, frame/garbage=8; see
 * tetris-core.js). The Play page's DOM board uses the same colours in CSS
 * (.v1-.v8 in assets/css/style.css).
 */

export const PIECE_COLORS = Object.freeze({ 1: '#e6483d', 2: '#4ec3e0', 3: '#f2c94c', 4: '#5fc27e', 5: '#f2994a', 6: '#4a7ef2', 7: '#9b51e0' });

/** Garbage rows (and the frame). */
export const GARBAGE_COLOR = '#6b7280';
