/**
 * replay-memory.js - the last N transitions of training, and random batches
 * drawn from them.
 *
 * A transition is "state -> return -> next state": the input of a board, the
 * reward collected after it (one move's, or N moves' discounted), and the
 * board where that left us. Training on random batches instead of the latest
 * moves breaks the correlation between consecutive moves, which keeps deep
 * Q-learning stable.
 */
export class ReplayMemory {
  /**
   * @param {number} capacity   transitions kept; the oldest is overwritten first
   * @param {number} stateSize  inputs per state
   */
  constructor(capacity, stateSize) {
    this.capacity = capacity;
    this.stateSize = stateSize;
    this.states = new Float64Array(capacity * stateSize);
    this.nextStates = new Float64Array(capacity * stateSize);
    this.returns = new Float64Array(capacity);
    this.done = new Uint8Array(capacity);
    this.discounts = new Float64Array(capacity); // what V(next state) is multiplied by: gamma^steps
    this.size = 0;
    this.writePos = 0;
  }

  /** Stores one transition. `done`: the game ended, so the next state is worth nothing. */
  add(state, ret, nextState, done, discount) {
    const k = this.writePos;
    this.discounts[k] = discount;
    this.states.set(state, k * this.stateSize);
    this.nextStates.set(nextState, k * this.stateSize);
    this.returns[k] = ret;
    this.done[k] = done ? 1 : 0;
    this.writePos = (k + 1) % this.capacity;
    if (this.size < this.capacity) this.size++;
  }

  /**
   * A random training batch (with replacement): the inputs X (flat, batch x
   * stateSize) and their Bellman targets
   *
   *   y = return                                   if the game ended
   *   y = return + discount x valueOf(next state)  otherwise
   *
   * @param {number} batchSize  at most this many (fewer while memory is small)
   * @param {() => number} rand
   * @param {(state: Float64Array) => number} valueOf  the network's V
   */
  sampleBatch(batchSize, rand, valueOf) {
    const n = this.stateSize;
    const B = Math.min(batchSize, this.size);
    const X = new Float64Array(B * n);
    const y = new Float64Array(B);
    const next = new Float64Array(n);
    for (let i = 0; i < B; i++) {
      const k = Math.floor(rand() * this.size);
      X.set(this.states.subarray(k * n, (k + 1) * n), i * n);
      next.set(this.nextStates.subarray(k * n, (k + 1) * n));
      y[i] = this.done[k] ? this.returns[k] : this.returns[k] + this.discounts[k] * valueOf(next);
    }
    return { X, y };
  }
}
