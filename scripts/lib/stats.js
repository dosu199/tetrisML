/** stats.js - the few summary numbers the scripts print. */

export const sum = (xs) => xs.reduce((a, b) => a + b, 0);

export const mean = (xs) => sum(xs) / xs.length;

export function median(xs) {
  const sorted = [...xs].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
