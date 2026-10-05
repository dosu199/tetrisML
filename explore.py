"""
explore.py - your Tetris project as ordinary numpy and scikit-learn.

    python3 explore.py --data data/values.csv

Everything this project produced is plain data, and all of it opens in the
tools your course teaches. This script is a guided tour of that, in five parts:

  1. Load the trained network (value-model.json) into numpy arrays and run its
     forward pass yourself, in five lines.
  2. Load the board dataset with pandas and look at it.
  3. Rebuild the eight hand-designed features from raw cells, using nothing but
     numpy array operations.
  4. Compare models on the same data - and watch a 9-parameter linear model
     beat a 39,297-parameter neural network, because it gets better inputs.
  5. Draw a learning curve, which tells you whether your bottleneck is data or
     model capacity.

Nothing here changes the game. It is a notebook you can run, edit and argue
with - which is how you actually learn this material.
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/values.csv")
parser.add_argument("--model", default="value-model.json")
parser.add_argument("--skip-curve", action="store_true")
parser.add_argument("--sample", type=int, default=150_000,
                    help="cap on rows loaded; 0 = all. Guards against running out of memory.")
args = parser.parse_args()

ROWS, COLS = 24, 10
line = lambda t: print(f"\n{t}\n" + "-" * len(t))

# =====================================================================
line("1. the trained network, as numpy arrays")

with open(args.model) as f:
    model = json.load(f)

layers = [(np.array(l["W"], dtype=np.float32),
           np.array(l["b"], dtype=np.float32),
           l["activation"]) for l in model["layers"]]

for i, (W, b, act) in enumerate(layers):
    print(f"  layer {i}: W {W.shape}, b {b.shape}, activation {act}")
print(f"  total parameters: {sum(W.size + b.size for W, b, _ in layers):,}")


def net_forward(X):
    """The entire network. A matrix multiply, an activation, repeat."""
    a = X
    for W, b, act in layers:
        a = a @ W + b
        if act == "relu":
            a = np.maximum(a, 0)
    return a[:, 0]


# What did it learn to look at? Weight magnitude per board row.
row_strength = np.linalg.norm(layers[0][0], axis=1)[: ROWS * COLS].reshape(ROWS, COLS).mean(1)
print("\n  attention per board row (row 0 = top):")
for r in range(0, ROWS, 3):
    print(f"    row {r:2d}  {row_strength[r]:6.3f}  {'#' * int(row_strength[r] * 35)}")
print("  The top rows are near zero: the bot never let the stack get that")
print("  high, so those inputs were always 0 and never trained.")

# =====================================================================
line("2. the dataset, with pandas")

# Declare int8 for the board cells. Left to itself pandas reads them as int64
# and spends eight bytes on a number that is only ever 0 or 1 - on a 640k-row
# file that is 1.2 GB instead of 155 MB.
_head = pd.read_csv(args.data, nrows=0).columns
_dtype = {c: np.int8 for c in _head if c.startswith(("a", "b")) and "_" in c}
_dtype["lines"] = np.int8
_dtype["target"] = np.float32
if "decision_id" in _head:
    _dtype["decision_id"] = np.int32

df = pd.read_csv(args.data, dtype=_dtype)
if args.sample and args.sample < len(df):
    df = df.sample(args.sample, random_state=42)
    print(f"  subsampled to {args.sample:,} rows (--sample 0 to use all)")
print(f"  shape      {df.shape}")
print(f"  columns    a0_0 ... a23_9 (240 board cells), lines, target"
      + (", decision_id" if "decision_id" in df.columns else ""))
print(f"  target     mean {df.target.mean():.1f}, std {df.target.std():.1f}, "
      f"range [{df.target.min():.1f}, {df.target.max():.1f}]")

cell_cols = [c for c in df.columns if c.startswith("a") and "_" in c]
boards = df[cell_cols].to_numpy(dtype=np.int8).reshape(-1, ROWS, COLS)
lines_cleared = df["lines"].to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.float32)
print(f"  boards     reshaped to {boards.shape}  (games, rows, columns)")
print(f"  occupancy  {boards.mean():.1%} of cells filled on average")

# =====================================================================
line("3. the eight features, rebuilt in numpy")

# Column heights: distance from the topmost filled cell down to the floor.
any_filled = boards.any(axis=1)                       # (n, 10)
top_row = boards.argmax(axis=1)                       # first 1 from the top
heights = np.where(any_filled, ROWS - top_row, 0)     # (n, 10)

filled_per_col = boards.sum(axis=1)                   # (n, 10)

aggregate = heights.sum(1)
max_height = heights.max(1)
bumpiness = np.abs(np.diff(heights, axis=1)).sum(1)

# A hole is a cell inside a column's occupied span that is not filled - so the
# count falls straight out of two numbers we already have.
holes = (heights - filled_per_col).sum(1)

# Transitions: how ragged the board is. Walls count as filled, sky as empty.
pad_lr = np.pad(boards, ((0, 0), (0, 0), (1, 1)), constant_values=1)
row_transitions = (np.diff(pad_lr, axis=2) != 0).sum((1, 2))

pad_tb = np.concatenate(
    [np.zeros((len(boards), 1, COLS), np.int8), boards, np.ones((len(boards), 1, COLS), np.int8)],
    axis=1,
)
col_transitions = (np.diff(pad_tb, axis=1) != 0).sum((1, 2))

# Wells: a column lower than both neighbours, weighted by depth.
big = np.full((len(heights), 1), 10_000)
neighbours = np.minimum(
    np.concatenate([big, heights[:, :-1]], axis=1),
    np.concatenate([heights[:, 1:], big], axis=1),
)
depth = np.maximum(neighbours - heights, 0)
wells = (depth * (depth + 1) // 2).sum(1)

F = np.column_stack([aggregate, lines_cleared, holes, bumpiness,
                     max_height, row_transitions, col_transitions, wells]).astype(np.float32)
names = ["aggregateHeight", "linesCleared", "holes", "bumpiness",
         "maxHeight", "rowTransitions", "colTransitions", "wells"]

print("  feature            mean      std   corr with target")
for i, n in enumerate(names):
    corr = np.corrcoef(F[:, i], y)[0, 1]
    print(f"  {n:<16} {F[:, i].mean():7.2f}  {F[:, i].std():7.2f}   {corr:+.3f}")

# =====================================================================
line("4. model comparison - same data, different inputs")

X_cells = np.column_stack([boards.reshape(len(boards), -1), lines_cleared]).astype(np.float32)

idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=42)

def score(name, model, X):
    model.fit(X[idx_tr], y[idx_tr])
    r2 = r2_score(y[idx_te], model.predict(X[idx_te]))
    p = sum(v.size for v in [getattr(model, "coef_", np.array([]))]) + 1
    print(f"  {name:<44} R2 {r2:7.4f}")
    return r2

print("  inputs: 241 raw cells")
score("linear regression", LinearRegression(), X_cells)
score("ridge (alpha=1)", Ridge(alpha=1.0), X_cells)
print(f"  {'the trained neural net (39,297 params)':<44} R2 "
      f"{r2_score(y[idx_te], net_forward(X_cells[idx_te]) * model['targetStd'] + model['targetMean']):7.4f}")

print("\n  inputs: the 8 engineered features")
score("linear regression (9 parameters)", LinearRegression(), F)
# A forest stores every split of every tree, so its memory grows with the
# dataset as well as the tree count. 200 unbounded trees on half a million rows
# is how you get killed by the OOM reaper; 60 trees with a depth limit answers
# the same question for a fraction of the cost.
score("random forest (60 trees, depth 16)",
      RandomForestRegressor(n_estimators=60, max_depth=16, n_jobs=2, random_state=42), F)

print("""
  Read that table again. A linear model with NINE numbers beats a neural
  network with 39,297 - not because it is smarter, but because somebody
  already told it what to look at. The network had to discover height,
  holes and bumpiness from raw cells on its own, and it nearly did.

  That is the entire trade of feature engineering against representation
  learning, on your own data. Hand-designed features win when you know the
  domain; learned features win when you do not, or when nobody could write
  the features down (images, speech, language).""")

# =====================================================================
if not args.skip_curve:
    line("5. learning curve - is the bottleneck data, or the model?")

    warnings.filterwarnings("ignore")  # "did not converge" is expected here
    sizes = [2000, 5000, 10000, 20000, 40000, len(idx_tr)]
    print("  training rows     test R2 (raw cells, MLP 128x64)")
    for n in sizes:
        sub = idx_tr[:n]
        m = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=40, early_stopping=True,
                         n_iter_no_change=5, random_state=42)
        m.fit(X_cells[sub], y[sub])
        print(f"  {n:>13,}     {r2_score(y[idx_te], m.predict(X_cells[idx_te])):7.4f}")

    print("""
  If the last numbers are still climbing, collect more data. If they have
  flattened, more data will not help and the next move is a different
  architecture - a small convolutional network, which knows the board is a
  grid instead of 240 unrelated numbers.

  One honest caveat, and a good exercise: max_iter is fixed at 40 EPOCHS, so
  a bigger dataset also gets more gradient steps. Part of the rise is simply
  more training, not more information. Redo this with a fixed number of
  steps rather than epochs and see how much of the slope survives - that is
  the difference between a curve that answers the question and one that
  looks like it does.""")

print()
