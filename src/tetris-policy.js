/**
 * tetris-policy.js - the neural network's side of Phase 2.
 *
 * Phase 1 learned eight numbers inside a scoring function that a human wrote.
 * Phase 2 throws that scoring function away and learns the *decision itself*:
 * a model that looks at a board and names the move, with no features, no
 * search, and no evaluation function in between.
 *
 * This file owns three things, and it owns them because they must agree
 * exactly between the JavaScript that generates the data, the Python that
 * trains on it, and the JavaScript that plays with the result:
 *
 *   1. How a board becomes a vector of numbers      -> buildInput()
 *   2. How a placement becomes a class label        -> moveClass()
 *   3. How a trained model picks a legal move       -> choosePolicyMove()
 *
 * A mismatch in any of those three is the classic way this phase fails
 * silently: the model trains to 90% accuracy and then plays like it has never
 * seen Tetris, because column 7 in training means something different from
 * column 7 at inference. Keeping all three definitions in one file, imported
 * by everything else, is the defence.
 */

import {
  PLAY_LEFT,
  PLAY_RIGHT,
  PLAY_COLS,
  PLAY_BOTTOM,
  PIECES,
  idx,
  placements,
} from './tetris-core.js';

export const NUM_PIECES = PIECES.length; // 7
export const MAX_ROTATIONS = 4;
export const NUM_CLASSES = MAX_ROTATIONS * PLAY_COLS; // 40

/* ------------------------------------------------------------------- input */

/**
 * Board cells, then a one-hot for the current piece, then one for the next.
 *
 *   240  playfield occupancy, row-major from the top, 1 = filled
 *     7  current piece
 *     7  next piece
 *   ---
 *   254
 *
 * Occupancy is deliberately binary rather than the colour value. Which colour
 * a settled block used to be has no bearing on where the next piece should go,
 * and feeding it in would invite the model to learn superstitions about it.
 */
export const BOARD_CELLS = (PLAY_BOTTOM + 1) * PLAY_COLS; // 24 * 10 = 240
export const INPUT_SIZE = BOARD_CELLS + NUM_PIECES * 2; // 254

export function buildInput(board, pieceIdx, nextIdx, out = new Float64Array(INPUT_SIZE)) {
  out.fill(0);
  let k = 0;
  for (let r = 0; r <= PLAY_BOTTOM; r++) {
    for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
      out[k++] = board[idx(r, c)] !== 0 ? 1 : 0;
    }
  }
  out[k + pieceIdx] = 1;
  k += NUM_PIECES;
  if (nextIdx != null) out[k + nextIdx] = 1;
  return out;
}

/** CSV header, in exactly the order buildInput() writes. */
export function inputColumns() {
  const names = [];
  for (let r = 0; r <= PLAY_BOTTOM; r++) {
    for (let c = 0; c < PLAY_COLS; c++) names.push(`b${r}_${c}`);
  }
  for (let p = 0; p < NUM_PIECES; p++) names.push(`piece_${PIECES[p].name}`);
  for (let p = 0; p < NUM_PIECES; p++) names.push(`next_${PIECES[p].name}`);
  return names;
}

/* ------------------------------------------------------------------ labels */

/** (rotation, absolute column) -> a single class in 0..39. */
export const moveClass = (rotation, col) => rotation * PLAY_COLS + (col - PLAY_LEFT);

export const decodeClass = (k) => ({
  rotation: Math.floor(k / PLAY_COLS),
  col: (k % PLAY_COLS) + PLAY_LEFT,
});

/* --------------------------------------------------------------- inference */

/**
 * Loads a model exported by train_policy.py. Deliberately a hand-written
 * forward pass rather than a library: a multi-layer perceptron is a matrix
 * multiply, an activation, and repeat, and writing it out once makes that
 * concrete. It also means the browser needs no dependencies at all.
 */
export function loadPolicy(json) {
  const { layers, scaler } = json;

  function forward(x) {
    let a = x;
    if (scaler) {
      a = new Float64Array(x.length);
      for (let i = 0; i < x.length; i++) a[i] = (x[i] - scaler.mean[i]) / scaler.scale[i];
    }

    for (const layer of layers) {
      const { W, b, activation } = layer;
      const out = new Float64Array(b.length);
      for (let j = 0; j < b.length; j++) out[j] = b[j];
      for (let i = 0; i < a.length; i++) {
        const xi = a[i];
        if (xi === 0) continue; // the board is mostly zeros; skip the row
        const row = W[i];
        for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
      }
      if (activation === 'relu') {
        for (let j = 0; j < out.length; j++) if (out[j] < 0) out[j] = 0;
      } else if (activation === 'tanh') {
        for (let j = 0; j < out.length; j++) out[j] = Math.tanh(out[j]);
      }
      a = out;
    }
    return a; // raw scores; we only ever argmax them, so softmax is redundant
  }

  return {
    classes: json.classes,
    scores(board, pieceIdx, nextIdx) {
      return forward(buildInput(board, pieceIdx, nextIdx));
    },
  };
}

/**
 * Picks the model's highest-scoring move *among the legal ones*.
 *
 * The masking matters more than it looks. A network asked to choose among 40
 * classes will happily name a placement that does not fit on this board, and
 * an unmasked policy spends much of its life proposing impossible moves. The
 * legal set is cheap to compute - we already have placements() - so there is
 * no reason to let the model guess at legality. What we are asking it to learn
 * is judgement, not the rules.
 */
export function choosePolicyMove(board, pieceIdx, nextIdx, policy) {
  const legal = placements(board, pieceIdx);
  if (legal.length === 0) return null;

  const raw = policy.scores(board, pieceIdx, nextIdx);
  const classIndex = new Map(policy.classes.map((c, i) => [c, i]));

  let best = null;
  let bestScore = -Infinity;
  for (const m of legal) {
    const i = classIndex.get(moveClass(m.rotation, m.col));
    // A class the training set never contained cannot be scored; treat it as
    // the worst option rather than crashing.
    const score = i === undefined ? -Infinity : raw[i];
    if (score > bestScore) {
      bestScore = score;
      best = m;
    }
  }
  return best ?? legal[0];
}

/* ------------------------------------------------- the value-network variant

 * Behaviour cloning above asks: "board in, which of 40 moves out?" That turns
 * out to be a brutal thing to learn from raw cells, because the label is an
 * abstract action index whose meaning changes with the piece, and the network
 * has to infer the whole geometry of fitting from scratch.
 *
 * This variant asks a much easier question: "here is a board that COULD
 * result from a move - how good is it?" Then at play time we enumerate the
 * legal placements ourselves, build each resulting board, score them all, and
 * take the best. The network never has to know the rules; it only has to
 * develop taste.
 *
 * This is knowledge distillation: the target is the expert's own evaluation
 * score, so the network is learning to reproduce a function we already have.
 * That sounds circular until you notice what it must do to succeed - it has to
 * learn to see holes, bumpiness and stack height in a grid of raw cells, with
 * nobody telling it those concepts exist.
 *
 * It is also the exact shape Phase 3 needs. A DQN over afterstates is this
 * network with a different training signal.
 */

import { copyBoard, placePiece, clearLines, createBoard } from './tetris-core.js';

export const VALUE_INPUT_SIZE = BOARD_CELLS + 1; // 240 cells + lines cleared

export function buildValueInput(board, linesCleared, out = new Float64Array(VALUE_INPUT_SIZE)) {
  let k = 0;
  for (let r = 0; r <= PLAY_BOTTOM; r++) {
    for (let c = PLAY_LEFT; c <= PLAY_RIGHT; c++) {
      out[k++] = board[idx(r, c)] !== 0 ? 1 : 0;
    }
  }
  out[k] = linesCleared;
  return out;
}

export function valueColumns() {
  const names = [];
  for (let r = 0; r <= PLAY_BOTTOM; r++) {
    for (let c = 0; c < PLAY_COLS; c++) names.push(`a${r}_${c}`);
  }
  names.push('lines');
  return names;
}

export function loadValueNet(json) {
  // Both network families expose the same score(board, lines), so everything
  // downstream - chooseValueMove, play_policy.js, DAgger, the browser - works
  // with either without knowing which one it has.
  if (json.kind === 'value-cnn') return loadCnnNet(json);

  const { layers } = json;
  const scratch = new Float64Array(VALUE_INPUT_SIZE);

  function forward(x) {
    let a = x;
    for (const { W, b, activation } of layers) {
      const out = new Float64Array(b.length);
      for (let j = 0; j < b.length; j++) out[j] = b[j];
      for (let i = 0; i < a.length; i++) {
        const xi = a[i];
        if (xi === 0) continue;
        const row = W[i];
        for (let j = 0; j < out.length; j++) out[j] += xi * row[j];
      }
      if (activation === 'relu') for (let j = 0; j < out.length; j++) if (out[j] < 0) out[j] = 0;
      a = out;
    }
    return a[0];
  }

  return { score: (board, lines) => forward(buildValueInput(board, lines, scratch)) };
}

/** Enumerate placements, build each afterstate, let the network judge. */
/**
 * Picks a move by asking the network to judge resulting boards.
 *
 *   chooseValueMove(board, piece, net)                          1-ply
 *   chooseValueMove(board, piece, net, { next, lookahead: true }) 2-ply
 *
 * 1-PLY: score every board the current piece can produce, take the best.
 *
 * 2-PLY: for each placement of the current piece, also try every placement of
 * the NEXT piece, and judge the current move by the best board it leads to.
 * This is the same lookahead the Phase 1 bot uses, now driven by the network.
 *
 * Why it helps a network in particular: a 1-ply choice trusts one noisy
 * judgement. A 2-ply choice asks "which move leaves me a good follow-up?",
 * and a bad board usually has no good follow-up whichever way the noise
 * falls - so the network's errors get partly averaged away instead of acted
 * on directly. Half of all decisions had a margin smaller than the network's
 * error; this is a direct attack on that.
 *
 * BEAM: full 2-ply is ~34 x 34 = 1,156 network evaluations per piece, which
 * is slow in plain JavaScript. So we first rank the current piece's moves by
 * their 1-ply score and only look ahead from the top `beam` of them. The best
 * move is almost always among the first handful; beam 5 costs ~200
 * evaluations and gives nearly all of the benefit.
 */
export function chooseValueMove(board, pieceIdx, valueNet, options = {}) {
  const { next = null, lookahead = false, beam = 5 } = options;

  const legal = placements(board, pieceIdx);
  if (legal.length === 0) return null;

  // 1-ply scores for every candidate, keeping the boards for the next stage.
  const first = [];
  for (const m of legal) {
    const after = copyBoard(board);
    placePiece(after, PIECES[pieceIdx].rotations[m.rotation], m.row, m.col);
    const lines = clearLines(after);
    first.push({ m, after, lines, score: valueNet.score(after, lines) });
  }

  if (!lookahead || next == null) {
    let best = first[0];
    for (const f of first) if (f.score > best.score) best = f;
    return best.m;
  }

  first.sort((a, b) => b.score - a.score);
  const frontier = first.slice(0, Math.max(1, beam));

  let best = null;
  let bestScore = -Infinity;
  for (const f of frontier) {
    const follow = placements(f.after, next);
    // No legal follow-up means this move tops out on the next piece.
    let value = -Infinity;
    for (const m2 of follow) {
      const after2 = copyBoard(f.after);
      placePiece(after2, PIECES[next].rotations[m2.rotation], m2.row, m2.col);
      const lines2 = clearLines(after2);
      // Credit BOTH moves' line clears. An earlier version passed only the
      // second move's clears, reasoning that the first move's clears were
      // already "baked into" the lower board. That was wrong, and costly: a
      // first move that cleared a line got no credit for it, so the search
      // learned to postpone clears, the stack crept up, and 2-ply played at a
      // THIRD of 1-ply (33 lines vs 51). Summing both - exactly what the
      // Phase 1 bot does - took it to ~177. Capped at 4 to stay inside the
      // range the network saw in training.
      const v = valueNet.score(after2, Math.min(4, f.lines + lines2));
      if (v > value) value = v;
    }
    if (value > bestScore) {
      bestScore = value;
      best = f.m;
    }
  }
  // Every move in the beam dies next turn: fall back to the best 1-ply move.
  return best ?? first[0].m;
}

/* ------------------------------------------------ the convolutional network

 * Mirrors forward() in train_cnn.py exactly. Layout of the input:
 *
 *   a 26 x 12 grid = the 24 x 10 playfield with one cell of padding all round.
 *   Walls and floor count as filled (1), the sky above counts as empty (0).
 *   A second, constant channel holds each cell's height above the floor,
 *   normalised: (25 - paddedRow) / 24.
 *
 * Two shortcuts make this fast enough to run ~200 times per piece:
 *
 *  1. conv1 is linear before its ReLU, so the parts of it that never change -
 *     the bias, the height channel, the walls and the floor - are added up
 *     ONCE at load time into a base map. Per board we only add the
 *     contribution of filled playfield cells, about 40 of 240.
 *  2. Everything is flat Float32Arrays and plain loops; no allocation per call.
 */
function loadCnnNet(json) {
  const R = json.rows; // 24
  const C = json.cols; // 10
  const P = R * C; // 240 positions
  const { C1, C2, H } = json;
  const PR = R + 2;
  const PC = C + 2;

  const flat = (a) => Float32Array.from(a.flat(Infinity));
  const W1 = flat(json.W1); // [18][C1]: rows 0-8 board (dy*3+dx), 9-17 height
  const b1 = flat(json.b1);
  const Wdw = flat(json.Wdw); // [3][3][C1]
  const bdw = flat(json.bdw);
  const Wpw = flat(json.Wpw); // [C1][C2]
  const bpw = flat(json.bpw);
  const Wf1 = flat(json.Wf1); // [C2+1][H]
  const bf1 = flat(json.bf1);
  const Wf2 = flat(json.Wf2); // [H][1]
  const bf2 = json.bf2[0];

  const height = (pr) => (R + 1 - pr) / R;

  // --- base map: bias + height channel + walls/floor, computed once.
  const base = new Float32Array(P * C1);
  for (let y = 0; y < R; y++) {
    for (let x = 0; x < C; x++) {
      const o = (y * C + x) * C1;
      for (let c = 0; c < C1; c++) base[o + c] = b1[c];
      for (let dy = 0; dy < 3; dy++) {
        for (let dx = 0; dx < 3; dx++) {
          const pr = y + dy;
          const pc = x + dx;
          const k = dy * 3 + dx;
          const h = height(pr);
          const wall = pc === 0 || pc === PC - 1 || pr === PR - 1 ? 1 : 0;
          for (let c = 0; c < C1; c++) {
            base[o + c] += h * W1[(9 + k) * C1 + c] + wall * W1[k * C1 + c];
          }
        }
      }
    }
  }

  const a1 = new Float32Array(P * C1);
  const z2 = new Float32Array(C1);
  const pooled = new Float32Array(C2);
  const hidden = new Float32Array(H);

  // dw + pw + ReLU for every position in rows [y0, R), added into `acc`.
  function poolRows(y0, acc) {
    for (let y = y0; y < R; y++) {
      for (let x = 0; x < C; x++) {
        for (let ch = 0; ch < C1; ch++) z2[ch] = bdw[ch];
        for (let dy = 0; dy < 3; dy++) {
          const yy = y + dy - 1;
          if (yy < 0 || yy >= R) continue;
          for (let dx = 0; dx < 3; dx++) {
            const xx = x + dx - 1;
            if (xx < 0 || xx >= C) continue;
            const o = (yy * C + xx) * C1;
            const w = (dy * 3 + dx) * C1;
            for (let ch = 0; ch < C1; ch++) z2[ch] += a1[o + ch] * Wdw[w + ch];
          }
        }
        for (let k = 0; k < C2; k++) {
          let s = bpw[k];
          for (let ch = 0; ch < C1; ch++) s += z2[ch] * Wpw[ch * C2 + k];
          if (s > 0) acc[k] += s;
        }
      }
    }
  }

  function conv1(board) {
    // start from the base map, scatter in each filled playfield cell
    a1.set(base);
    let topRow = R;
    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        if (board[idx(r, PLAY_LEFT + c)] === 0) continue;
        if (r < topRow) topRow = r;
        const pr = r + 1;
        const pc = c + 1;
        for (let dy = 0; dy < 3; dy++) {
          const y = pr - dy;
          if (y < 0 || y >= R) continue;
          for (let dx = 0; dx < 3; dx++) {
            const x = pc - dx;
            if (x < 0 || x >= C) continue;
            const o = (y * C + x) * C1;
            const w = (dy * 3 + dx) * C1;
            for (let ch = 0; ch < C1; ch++) a1[o + ch] += W1[w + ch];
          }
        }
      }
    }
    for (let i = 0; i < a1.length; i++) if (a1[i] < 0) a1[i] = 0;
    return topRow;
  }

  // The sky trick. The network's receptive field is 5 x 5, so an output
  // position in row y only "sees" filled cells in rows y-2 .. y+2. Every row
  // more than two above the top of the stack therefore produces exactly what
  // it would on an EMPTY board - the same numbers every time. Compute those
  // once, as running totals by row, and skip them per board. With a typical
  // stack of 6-8 rows that is well over half the board, for free.
  const skyPrefix = new Float32Array((R + 1) * C2); // [t][k] = rows 0..t-1
  {
    conv1(createBoard());
    for (let y = 0; y < R; y++) {
      const row = new Float32Array(C2);
      poolRows(y, row); // rows y..R-1 ...
      const below = new Float32Array(C2);
      if (y + 1 < R) poolRows(y + 1, below); // ... minus rows y+1..R-1 = row y
      for (let k = 0; k < C2; k++) {
        skyPrefix[(y + 1) * C2 + k] = skyPrefix[y * C2 + k] + (row[k] - below[k]);
      }
    }
  }

  function score(board, lines) {
    const topRow = conv1(board);
    const y0 = Math.max(0, topRow - 2);
    for (let k = 0; k < C2; k++) pooled[k] = skyPrefix[y0 * C2 + k];
    poolRows(y0, pooled);

    // head: [pooled / 240, lines] -> H -> 1
    let out = bf2;
    for (let j = 0; j < H; j++) {
      let s = bf1[j] + lines * Wf1[C2 * H + j];
      for (let k = 0; k < C2; k++) s += (pooled[k] / P) * Wf1[k * H + j];
      if (s > 0) out += s * Wf2[j];
    }
    return out;
  }

  // Prove on load that this matches what Python computed. If an export or
  // layout bug ever sneaks in, it fails here, loudly, instead of quietly
  // producing a network that plays badly for no visible reason.
  if (Array.isArray(json.selfTest)) {
    for (const t of json.selfTest) {
      const b = createBoard();
      for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) b[idx(r, PLAY_LEFT + c)] = t.board[r * C + c];
      const got = score(b, t.lines);
      if (Math.abs(got - t.score) > 1e-3 * Math.max(1, Math.abs(t.score))) {
        throw new Error(`CNN self-test failed: JS ${got} vs Python ${t.score}`);
      }
    }
  }

  return { score, kind: 'cnn', params: json.params };
}
