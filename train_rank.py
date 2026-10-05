"""
train_rank.py - Phase 2C. Same network, different question.

    python3 train_rank.py --data data/values.csv

train_value.py asked "what score does this board deserve?" and answered it
well: R squared 0.998. Then the bot built on it played at 12% of its teacher,
because a squared-error loss spends its effort getting the absolute level right
across a range with standard deviation ~21, while the decision only ever needs
to resolve differences of about 1.0 between the top few candidates. Half the
time the model's noise was larger than the margin it was being asked to judge.

So this trains on the thing the decision actually uses: ORDER. Show the network
two boards the bot was genuinely choosing between, and ask only which is
better. The loss is logistic on the difference of the two scores,

    L = log(1 + exp(-(f(better) - f(worse))))

which is RankNet, or equivalently a Bradley-Terry model. It is large when the
order is wrong and near zero the moment it is right - whether the gap is 0.5 or
50. Every training example is now a decision the model will actually face.

Written in plain numpy rather than a framework, for two reasons: PyTorch is a
529 MB dependency for a two-layer network, and writing the backward pass by
hand once is worth more than reading about it ten times.

Exports rank-model.json in exactly the format src/tetris-policy.js already
loads, so nothing on the JavaScript side changes.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/values.csv")
parser.add_argument("--model", default="rank-model.json")
parser.add_argument("--compare", default="value-model.json", help="regression model to compare against")
parser.add_argument("--hidden", default="128,64")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--batch", type=int, default=512)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sample", type=int, default=200_000,
                    help="cap on rows loaded; 0 = all. Guards against running out of memory.")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

# ------------------------------------------------------------------ load

print(f"\nloading {args.data}")

# Declare int8 for the board cells. Pandas would otherwise read 640k x 241
# cells as int64 and spend 1.2 GB on numbers that are only ever 0 or 1.
_header = pd.read_csv(args.data, nrows=0).columns
_dtype = {c: np.int8 for c in _header if c.startswith(("a", "b")) and "_" in c}
_dtype["lines"] = np.int8
_dtype["target"] = np.float32
if "decision_id" in _header:
    _dtype["decision_id"] = np.int32
df = pd.read_csv(args.data, dtype=_dtype)

# Subsample whole DECISIONS, never individual rows: a pair only makes sense
# between two boards the bot was choosing between at the same moment, so
# keeping half of a decision's candidates would quietly corrupt the pairs.
if args.sample and args.sample < len(df) and "decision_id" in df.columns:
    keep_n = max(1, int(len(df["decision_id"].unique()) * args.sample / len(df)))
    keep = rng.choice(df["decision_id"].unique(), size=keep_n, replace=False)
    df = df[df["decision_id"].isin(keep)]
    print(f"  subsampled to {len(df):,} rows / {keep_n:,} decisions (--sample 0 for all)")
if "decision_id" not in df.columns:
    raise SystemExit(
        "This CSV has no decision_id column. Re-run:\n"
        "  node collect_values.js --games 40 --cap 300 --samples 6"
    )

feature_cols = [c for c in df.columns if c not in ("target", "decision_id")]
X = df[feature_cols].to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.float32)
groups = df["decision_id"].to_numpy()

n_decisions = len(np.unique(groups))
print(f"  rows        {len(df):,}")
print(f"  features    {X.shape[1]}")
print(f"  decisions   {n_decisions:,}  (~{len(df) / n_decisions:.1f} candidate boards each)")

# Split by DECISION, not by row. Two boards from the same decision landing on
# opposite sides of the split would leak the answer across it.
uniq = np.unique(groups)
rng.shuffle(uniq)
cut = int(len(uniq) * 0.8)
train_dec, test_dec = set(uniq[:cut].tolist()), set(uniq[cut:].tolist())


def build_pairs(decision_set):
    """(better_row, worse_row) for every orderable pair inside each decision."""
    pairs = []
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    starts = np.searchsorted(g_sorted, np.unique(g_sorted), side="left")
    ends = np.searchsorted(g_sorted, np.unique(g_sorted), side="right")

    for g, s, e in zip(np.unique(g_sorted), starts, ends):
        if g not in decision_set:
            continue
        idx = order[s:e]
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ia, ib = idx[a], idx[b]
                if y[ia] > y[ib]:
                    pairs.append((ia, ib))
                elif y[ib] > y[ia]:
                    pairs.append((ib, ia))
                # equal targets carry no ordering information - skip
    return np.array(pairs, dtype=np.int64)


train_pairs = build_pairs(train_dec)
test_pairs = build_pairs(test_dec)
print(f"  train pairs {len(train_pairs):,}")
print(f"  test pairs  {len(test_pairs):,}")

# The pairs that actually decide games: the ones the regression model could not
# separate, where the true gap is smaller than its typical error (~0.97).
test_gap = np.abs(y[test_pairs[:, 0]] - y[test_pairs[:, 1]])
close = test_gap < 1.0
print(f"  of the test pairs, {close.mean():.1%} are 'close' (true gap < 1.0)")

# ------------------------------------------------------------------ model

hidden = [int(h) for h in args.hidden.split(",")]
sizes = [X.shape[1]] + hidden + [1]

# He initialisation: variance 2/fan_in keeps ReLU activations from collapsing
# or exploding as they pass through the layers.
W = [rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])).astype(np.float32)
     for i in range(len(sizes) - 1)]
b = [np.zeros(sizes[i + 1], dtype=np.float32) for i in range(len(sizes) - 1)]

mW = [np.zeros_like(w) for w in W]
vW = [np.zeros_like(w) for w in W]
mb = [np.zeros_like(x) for x in b]
vb = [np.zeros_like(x) for x in b]


def forward(inp):
    """Returns the output column and everything backward() needs."""
    acts, pres = [inp], []
    a = inp
    for i in range(len(W)):
        z = a @ W[i] + b[i]
        pres.append(z)
        a = np.maximum(z, 0) if i < len(W) - 1 else z  # last layer is linear
        acts.append(a)
    return a, acts, pres


def backward(acts, pres, grad_out, t):
    """Standard chain rule, then one Adam step. grad_out is dL/d(output)."""
    d = grad_out
    gW, gb = [None] * len(W), [None] * len(W)
    for i in reversed(range(len(W))):
        gW[i] = acts[i].T @ d
        gb[i] = d.sum(axis=0)
        if i > 0:
            d = (d @ W[i].T) * (pres[i - 1] > 0)

    b1, b2, eps = 0.9, 0.999, 1e-8
    for i in range(len(W)):
        mW[i] = b1 * mW[i] + (1 - b1) * gW[i]
        vW[i] = b2 * vW[i] + (1 - b2) * gW[i] ** 2
        mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
        vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
        mh = mW[i] / (1 - b1 ** t)
        vh = vW[i] / (1 - b2 ** t)
        W[i] -= args.lr * mh / (np.sqrt(vh) + eps)
        mh = mb[i] / (1 - b1 ** t)
        vh = vb[i] / (1 - b2 ** t)
        b[i] -= args.lr * mh / (np.sqrt(vh) + eps)


def score_all(data, batch=8192):
    out = np.empty(len(data), dtype=np.float32)
    for i in range(0, len(data), batch):
        out[i : i + batch] = forward(data[i : i + batch])[0][:, 0]
    return out


def pair_accuracy(scores, pairs, mask=None):
    ok = scores[pairs[:, 0]] > scores[pairs[:, 1]]
    return ok[mask].mean() if mask is not None else ok.mean()


def top1_accuracy(scores, decision_set):
    """How often the network's favourite board is the bot's favourite board."""
    hit = tot = 0
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    u = np.unique(g_sorted)
    starts = np.searchsorted(g_sorted, u, side="left")
    ends = np.searchsorted(g_sorted, u, side="right")
    for g, s, e in zip(u, starts, ends):
        if g not in decision_set:
            continue
        idx = order[s:e]
        if len(idx) < 2:
            continue
        tot += 1
        if y[idx[np.argmax(scores[idx])]] == y[idx].max():
            hit += 1
    return hit / max(tot, 1)


# ------------------------------------------------------------------ train

print(f"\ntraining  {' x '.join(map(str, hidden))} hidden, {args.steps} steps, batch {args.batch} pairs")
print("-" * 62)
print("  step      loss   pair acc   close-pair acc")

t0 = time.time()
for step in range(1, args.steps + 1):
    sel = rng.integers(0, len(train_pairs), size=args.batch)
    ia, ib = train_pairs[sel, 0], train_pairs[sel, 1]

    # Both boards go through the SAME weights in one batch, then we split the
    # output. That sharing is what makes f a single scoring function rather
    # than two unrelated ones.
    out, acts, pres = forward(np.concatenate([X[ia], X[ib]], axis=0))
    n = len(ia)
    diff = out[:n, 0] - out[n:, 0]

    # L = softplus(-diff);  dL/d(diff) = -sigmoid(-diff)
    # Computed the stable way: exp() of a large positive number overflows, so
    # each branch only ever exponentiates a negative one.
    loss = np.logaddexp(0.0, -diff).mean()
    pos = diff > 0
    sig_neg = np.empty_like(diff)
    sig_neg[pos] = np.exp(-diff[pos]) / (1.0 + np.exp(-diff[pos]))
    sig_neg[~pos] = 1.0 / (1.0 + np.exp(diff[~pos]))
    g = (-sig_neg / n).astype(np.float32)
    backward(acts, pres, np.concatenate([g, -g])[:, None], step)

    if step % 500 == 0 or step == 1:
        s = score_all(X)
        print(
            f"  {step:>4}  {loss:8.4f}   {pair_accuracy(s, test_pairs):7.1%}   "
            f"{pair_accuracy(s, test_pairs, close):11.1%}"
        )

elapsed = time.time() - t0
scores = score_all(X)

# ------------------------------------------------------------------ compare

print(f"\ndone in {elapsed:.0f}s")
print("\nheld-out comparison")
print("-" * 62)
print("                        all pairs   close pairs   picks bot's best")

rank_row = (
    pair_accuracy(scores, test_pairs),
    pair_accuracy(scores, test_pairs, close),
    top1_accuracy(scores, test_dec),
)


def load_js_model(path):
    with open(path) as f:
        m = json.load(f)
    return [(np.array(l["W"], dtype=np.float32), np.array(l["b"], dtype=np.float32), l["activation"])
            for l in m["layers"]]


try:
    layers = load_js_model(args.compare)

    def reg_score(data, batch=8192):
        out = np.empty(len(data), dtype=np.float32)
        for i in range(0, len(data), batch):
            a = data[i : i + batch]
            for Wl, bl, act in layers:
                a = a @ Wl + bl
                if act == "relu":
                    a = np.maximum(a, 0)
            out[i : i + batch] = a[:, 0]
        return out

    rs = reg_score(X)
    print(
        f"  regression (MSE)    {pair_accuracy(rs, test_pairs):9.1%}   "
        f"{pair_accuracy(rs, test_pairs, close):11.1%}   {top1_accuracy(rs, test_dec):16.1%}"
    )
except FileNotFoundError:
    print(f"  ({args.compare} not found - skipping comparison)")

print(f"  ranking (pairwise)  {rank_row[0]:9.1%}   {rank_row[1]:11.1%}   {rank_row[2]:16.1%}")

# ------------------------------------------------------------------ export

with open(args.model, "w") as f:
    json.dump(
        {
            "kind": "value-mlp",
            "inputSize": int(X.shape[1]),
            "hidden": hidden,
            "layers": [
                {"W": W[i].tolist(), "b": b[i].tolist(),
                 "activation": "relu" if i < len(W) - 1 else None}
                for i in range(len(W))
            ],
            "targetMean": 0.0,
            "targetStd": 1.0,
            "rankAccuracy": float(rank_row[0]),
            "closePairAccuracy": float(rank_row[1]),
            "top1Accuracy": float(rank_row[2]),
            "trainRows": int(len(train_pairs)),
            "trainedWith": "pairwise logistic (RankNet)",
        },
        f,
    )

print(f"\nwrote {args.model}")
print("\nnext: node play_policy.js --value rank-model.json\n")
