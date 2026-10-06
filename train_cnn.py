"""
train_cnn.py - Phase 2E. A convolutional value network, written in numpy.

    python3 train_cnn.py --data data/combined2.csv
    python3 train_cnn.py --gradcheck          # verify the backward pass first

WHY A CONVOLUTION
-----------------
The MLP sees 240 unrelated numbers. It has a separate weight for "cell (3,5) is
filled" and for "cell (15,5) is filled", and learns each one only from boards
where that exact cell was used. explore.py showed the result: the weights for
the top five rows are almost zero, because the bot never stacked that high, so
those inputs never received a gradient. The network is blind exactly where a
mistake becomes fatal.

A convolution uses the SAME small set of weights at every position. Whatever it
learns about a hole at row 15 applies automatically at row 3. That built-in
assumption - "the same pattern means the same thing wherever it appears" - is
called an inductive bias, and it is the central idea of convolutional networks.

THE ARCHITECTURE
----------------
    input    26 x 12 x 2   the board padded with walls and floor (=1) and open
                           sky above (=0), plus a height channel saying how high
                           above the floor each cell is
    conv1    3x3, 2 -> C1, ReLU
    conv2    3x3 depthwise + 1x1 pointwise (a "separable" 3x3), C1 -> C2, ReLU
    pool     average over all 240 positions -> C2 numbers
    head     [C2 pooled, lines cleared] -> H, ReLU -> 1

There are no per-position weights anywhere. Averaging over the board is what
makes the network fully translation-invariant; the height channel is what still
lets it know how tall the stack is. Local 3x3 detectors summed over the board
can count exactly the things the expert cares about - surface steps
(bumpiness), filled/empty flips (row and column transitions), overhangs
(holes), surface height (aggregate height) - without anyone telling it they
exist.

With C1=16, C2=32, H=32 it has about 2,000 parameters. The MLP it replaces has
94,000.

Trained with the same pairwise ranking loss as train_rank.py, on the same data,
so the comparison at the end is like for like. Exports cnn-model.json, which
src/tetris-policy.js loads exactly like the MLP models.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROWS, COLS = 24, 10
POS = ROWS * COLS

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/combined2.csv")
parser.add_argument("--model", default="cnn-model.json")
parser.add_argument("--compare", default="rank-dagger2.json")
parser.add_argument("--c1", type=int, default=16)
parser.add_argument("--c2", type=int, default=32)
parser.add_argument("--h", type=int, default=32)
parser.add_argument("--steps", type=int, default=12000)
parser.add_argument("--batch", type=int, default=256, help="pairs per step")
parser.add_argument("--lr", type=float, default=2e-3)
parser.add_argument("--sample", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gradcheck", action="store_true")
parser.add_argument("--init", default=None,
                    help="start from an existing cnn-model.json instead of random weights (fine-tuning)")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# ====================================================================== model

# Height above the floor for every padded row, normalised so the floor is 0
# and the top of the playfield is 1. Same for every column.
COORD = ((ROWS + 1 - np.arange(ROWS + 2)) / ROWS).astype(np.float32)[:, None].repeat(COLS + 2, 1)
COORD_COLS = sliding_window_view(COORD, (3, 3)).reshape(ROWS, COLS, 9).copy()


def pad_boards(boards):
    """(N, 24, 10) of 0/1 -> (N, 26, 12): walls and floor filled, sky empty."""
    P = np.zeros((len(boards), ROWS + 2, COLS + 2), dtype=np.float32)
    P[:, 1 : ROWS + 1, 1 : COLS + 1] = boards
    P[:, :, 0] = 1
    P[:, :, -1] = 1
    P[:, -1, :] = 1
    return P


def init_params(c1, c2, h, dtype=np.float32):
    he = lambda fan_in, shape: (rng.normal(0, np.sqrt(2.0 / fan_in), shape)).astype(dtype)
    return {
        "W1": he(18, (18, c1)),          # rows: board k=dy*3+dx (0..8), height (9..17)
        "b1": np.zeros(c1, dtype),
        "Wdw": he(9, (3, 3, c1)),        # one 3x3 filter per channel
        "bdw": np.zeros(c1, dtype),
        "Wpw": he(c1, (c1, c2)),         # 1x1 channel mixing
        "bpw": np.zeros(c2, dtype),
        "Wf1": he(c2 + 1, (c2 + 1, h)),  # pooled features + lines cleared
        "bf1": np.zeros(h, dtype),
        "Wf2": he(h, (h, 1)),
        "bf2": np.zeros(1, dtype),
    }


def forward(p, boards, lines):
    """Score a batch of boards. Returns the scores and a cache for backward()."""
    P = pad_boards(boards).astype(p["W1"].dtype)
    N = len(P)
    c1 = p["b1"].shape[0]

    # conv1, as a matrix multiply over 3x3 patches ("im2col")
    cols = sliding_window_view(P, (3, 3), axis=(1, 2)).reshape(N, ROWS, COLS, 9)
    z1 = cols @ p["W1"][:9] + (COORD_COLS.astype(P.dtype) @ p["W1"][9:]) + p["b1"]
    a1 = np.maximum(z1, 0)

    # depthwise 3x3: each channel convolved with its own filter, zero-padded
    Pa = np.zeros((N, ROWS + 2, COLS + 2, c1), dtype=P.dtype)
    Pa[:, 1 : ROWS + 1, 1 : COLS + 1, :] = a1
    z2 = np.broadcast_to(p["bdw"], a1.shape).copy()
    for dy in range(3):
        for dx in range(3):
            z2 += Pa[:, dy : dy + ROWS, dx : dx + COLS, :] * p["Wdw"][dy, dx]

    # pointwise 1x1, then average over every position on the board
    z3 = z2 @ p["Wpw"] + p["bpw"]
    a3 = np.maximum(z3, 0)
    pooled = a3.mean(axis=(1, 2))

    f_in = np.concatenate([pooled, lines[:, None].astype(P.dtype)], axis=1)
    hpre = f_in @ p["Wf1"] + p["bf1"]
    h = np.maximum(hpre, 0)
    out = (h @ p["Wf2"] + p["bf2"])[:, 0]
    return out, (cols, z1, Pa, z2, z3, f_in, hpre, h)


def backward(p, cache, g_out):
    """Chain rule back through every layer. g_out is dL/d(score), shape (N,)."""
    cols, z1, Pa, z2, z3, f_in, hpre, h = cache
    c1, c2 = p["Wpw"].shape
    g = {}

    d = g_out[:, None]
    g["Wf2"] = h.T @ d
    g["bf2"] = d.sum(0)
    dh = (d @ p["Wf2"].T) * (hpre > 0)
    g["Wf1"] = f_in.T @ dh
    g["bf1"] = dh.sum(0)
    dpool = (dh @ p["Wf1"].T)[:, :c2]

    # the average spreads each pooled gradient evenly over all 240 positions
    dz3 = (dpool[:, None, None, :] / POS) * (z3 > 0)
    g["Wpw"] = z2.reshape(-1, c1).T @ dz3.reshape(-1, c2)
    g["bpw"] = dz3.sum((0, 1, 2))
    dz2 = dz3 @ p["Wpw"].T

    g["bdw"] = dz2.sum((0, 1, 2))
    g["Wdw"] = np.zeros_like(p["Wdw"])
    dPa = np.zeros_like(Pa)
    for dy in range(3):
        for dx in range(3):
            win = Pa[:, dy : dy + ROWS, dx : dx + COLS, :]
            g["Wdw"][dy, dx] = (win * dz2).sum((0, 1, 2))
            dPa[:, dy : dy + ROWS, dx : dx + COLS, :] += dz2 * p["Wdw"][dy, dx]
    dz1 = dPa[:, 1 : ROWS + 1, 1 : COLS + 1, :] * (z1 > 0)

    g["W1"] = np.concatenate(
        [
            cols.reshape(-1, 9).T @ dz1.reshape(-1, c1),
            COORD_COLS.reshape(POS, 9).T.astype(dz1.dtype) @ dz1.sum(0).reshape(POS, c1),
        ],
        axis=0,
    )
    g["b1"] = dz1.sum((0, 1, 2))
    return g


# ================================================================= gradcheck

if args.gradcheck:
    # Compare every analytic gradient against a finite-difference estimate on a
    # tiny batch in float64. If backward() has a bug, this is where it shows -
    # long before it shows up as a network that mysteriously does not learn.
    p = init_params(4, 6, 5, dtype=np.float64)
    boards = (rng.random((3, ROWS, COLS)) < 0.3).astype(np.float64)
    lines = rng.integers(0, 3, 3).astype(np.float64)
    w = rng.normal(size=3)

    def loss(pp):
        return float((forward(pp, boards, lines)[0] * w).sum())

    out, cache = forward(p, boards, lines)
    g = backward(p, cache, w)
    worst = 0.0
    print("\ngradient check (analytic vs numeric)")
    for name in p:
        flat = p[name].reshape(-1)
        for i in rng.choice(flat.size, size=min(6, flat.size), replace=False):
            old = flat[i]
            flat[i] = old + 1e-6; lp = loss(p)
            flat[i] = old - 1e-6; lm = loss(p)
            flat[i] = old
            num = (lp - lm) / 2e-6
            ana = g[name].reshape(-1)[i]
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, rel)
        print(f"  {name:<4} ok" if worst < 1e-5 else f"  {name:<4} MISMATCH (rel {worst:.2e})")
    print(f"\n  worst relative error {worst:.2e}  ->  {'PASS' if worst < 1e-5 else 'FAIL'}\n")
    raise SystemExit(0 if worst < 1e-5 else 1)

# ====================================================================== data

print(f"\nloading {args.data}")
head = pd.read_csv(args.data, nrows=0).columns
dtype = {c: np.int8 for c in head if c.startswith("a") and "_" in c}
dtype.update({"lines": np.int8, "target": np.float32, "decision_id": np.int32})
df = pd.read_csv(args.data, dtype=dtype)
if "decision_id" not in df.columns:
    raise SystemExit("This CSV has no decision_id column - re-collect with collect_values.js.")
if args.sample and args.sample < len(df):
    keep = rng.choice(df.decision_id.unique(), size=int(df.decision_id.nunique() * args.sample / len(df)), replace=False)
    df = df[df.decision_id.isin(keep)]

cell_cols = [c for c in df.columns if c.startswith("a") and "_" in c]
BOARDS = df[cell_cols].to_numpy(np.uint8).reshape(-1, ROWS, COLS)
LINES = df["lines"].to_numpy(np.float32)
y = df["target"].to_numpy(np.float32)
groups = df["decision_id"].to_numpy()
del df
print(f"  boards     {len(BOARDS):,}  ({groups.max() + 1:,} decisions)")

order = np.argsort(groups, kind="stable")
g_sorted = groups[order]
uniq, starts = np.unique(g_sorted, return_index=True)
ends = np.append(starts[1:], len(g_sorted))

# Reproduce train_rank.py's split EXACTLY (same seed, same first draw), so the
# MLP we compare against is also being scored on decisions it never trained
# on. A different split would let the MLP be graded on its own training data -
# an early version of this script did that, and flattered the MLP by ~2 points.
_shuffled = np.unique(groups)
np.random.default_rng(args.seed).shuffle(_shuffled)
_test_ids = set(_shuffled[int(len(_shuffled) * 0.8) :].tolist())
test_dec = np.array([u in _test_ids for u in uniq])


def build_pairs(mask):
    out = []
    for k in np.nonzero(mask)[0]:
        idx = order[starts[k] : ends[k]]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ia, ib = idx[a], idx[b]
                if y[ia] > y[ib]:
                    out.append((ia, ib))
                elif y[ib] > y[ia]:
                    out.append((ib, ia))
    return np.array(out, dtype=np.int64)


train_pairs = build_pairs(~test_dec)
test_pairs = build_pairs(test_dec)
close = np.abs(y[test_pairs[:, 0]] - y[test_pairs[:, 1]]) < 1.0
test_rows = np.unique(test_pairs)
print(f"  pairs      {len(train_pairs):,} train, {len(test_pairs):,} test ({close.mean():.1%} close)")

# a fixed small probe for progress lines, so periodic evaluation stays cheap
probe = rng.choice(len(test_pairs), size=min(20000, len(test_pairs)), replace=False)


def score_rows(p, rows, chunk=4096):
    s = np.full(len(BOARDS), np.nan, dtype=np.float32)
    for i in range(0, len(rows), chunk):
        r = rows[i : i + chunk]
        s[r] = forward(p, BOARDS[r], LINES[r])[0]
    return s


def metrics(s, pairs, close_mask):
    ok = s[pairs[:, 0]] > s[pairs[:, 1]]
    return ok.mean(), ok[close_mask].mean()


def top1(s):
    hit = tot = 0
    for k in np.nonzero(test_dec)[0]:
        idx = order[starts[k] : ends[k]]
        if len(idx) < 2:
            continue
        tot += 1
        hit += y[idx[np.argmax(s[idx])]] == y[idx].max()
    return hit / max(tot, 1)


# ===================================================================== train

p = init_params(args.c1, args.c2, args.h)
if args.init:
    # Fine-tuning: keep everything the network already knows and continue
    # training on the new, larger dataset. Faster than starting over, and the
    # old skills are a head start rather than something to re-learn.
    with open(args.init) as f:
        _old = json.load(f)
    for k in p:
        p[k] = np.array(_old[k], dtype=np.float32).reshape(p[k].shape)
    print(f"  initialised from {args.init}")
n_params = sum(v.size for v in p.values())
m = {k: np.zeros_like(v) for k, v in p.items()}
v = {k: np.zeros_like(v) for k, v in p.items()}

print(f"\ntraining CNN  conv {args.c1}->{args.c2}, head {args.h}, {n_params:,} parameters")
print(f"              {args.steps} steps, batch {args.batch} pairs, lr {args.lr}")
print("-" * 62)
print("  step      loss   pair acc   close-pair acc   elapsed")

t0 = time.time()
for step in range(1, args.steps + 1):
    sel = rng.integers(0, len(train_pairs), size=args.batch)
    ia, ib = train_pairs[sel, 0], train_pairs[sel, 1]
    rows = np.concatenate([ia, ib])
    out, cache = forward(p, BOARDS[rows], LINES[rows])
    n = len(ia)
    diff = out[:n] - out[n:]

    loss = np.logaddexp(0.0, -diff).mean()
    sig = np.where(diff > 0, np.exp(-np.abs(diff)) / (1 + np.exp(-np.abs(diff))), 1 / (1 + np.exp(-np.abs(diff))))
    g1 = (-sig / n).astype(np.float32)
    grads = backward(p, cache, np.concatenate([g1, -g1]))

    lr = args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * step / args.steps)))  # cosine decay
    for k in p:
        m[k] = 0.9 * m[k] + 0.1 * grads[k]
        v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
        p[k] -= lr * (m[k] / (1 - 0.9**step)) / (np.sqrt(v[k] / (1 - 0.999**step)) + 1e-8)

    if step == 1 or step % 1000 == 0:
        pr = test_pairs[probe]
        s = score_rows(p, np.unique(pr))
        acc, cacc = metrics(s, pr, close[probe])
        print(f"  {step:>5}  {loss:8.4f}   {acc:7.1%}   {cacc:13.1%}   {time.time() - t0:6.0f}s")

# ==================================================================== result

s = score_rows(p, test_rows)
cnn = (*metrics(s, test_pairs, close), top1(s))

print("\nheld-out comparison (same test decisions)")
print("-" * 62)
print("                          all pairs   close pairs   picks bot's best")
try:
    with open(args.compare) as f:
        mlp = json.load(f)
    layers = [(np.array(l["W"], np.float32), np.array(l["b"], np.float32), l["activation"]) for l in mlp["layers"]]
    X = np.concatenate([BOARDS[test_rows].reshape(len(test_rows), -1), LINES[test_rows, None]], axis=1).astype(np.float32)
    a = X
    for W, b, act in layers:
        a = a @ W + b
        if act == "relu":
            a = np.maximum(a, 0)
    sm = np.full(len(BOARDS), np.nan, np.float32)
    sm[test_rows] = a[:, 0]
    mm = (*metrics(sm, test_pairs, close), top1(sm))
    n_mlp = sum(np.array(l["W"]).size + len(l["b"]) for l in mlp["layers"])
    print(f"  MLP  ({n_mlp:>6,} params)   {mm[0]:9.1%}   {mm[1]:11.1%}   {mm[2]:16.1%}")
except FileNotFoundError:
    print(f"  ({args.compare} not found - skipping comparison)")
print(f"  CNN  ({n_params:>6,} params)   {cnn[0]:9.1%}   {cnn[1]:11.1%}   {cnn[2]:16.1%}")

# Five boards with their scores, so the JavaScript loader can prove it computes
# exactly what Python computed. A layout mistake in the export would otherwise
# show up only as a network that plays mysteriously badly.
check_rows = test_rows[:5]
self_test = [
    {"board": BOARDS[r].reshape(-1).tolist(), "lines": int(LINES[r]), "score": float(s[r])}
    for r in check_rows
]

with open(args.model, "w") as f:
    json.dump(
        {
            "kind": "value-cnn",
            "rows": ROWS,
            "cols": COLS,
            "C1": args.c1,
            "C2": args.c2,
            "H": args.h,
            **{k: v_.tolist() for k, v_ in p.items()},
            "params": int(n_params),
            "rankAccuracy": float(cnn[0]),
            "closePairAccuracy": float(cnn[1]),
            "top1Accuracy": float(cnn[2]),
            "trainRows": int(len(train_pairs)),
            "trainedWith": "pairwise logistic (RankNet), numpy CNN",
            "selfTest": self_test,
        },
        f,
    )
print(f"\nwrote {args.model}")
print("\nnext: node play_policy.js --value cnn-model.json --lookahead\n")
