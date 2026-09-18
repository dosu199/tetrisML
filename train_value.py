"""
train_value.py - Phase 2, value-network variant.

    python3 train_value.py --data data/values.csv

A regression problem instead of a classification one: given a board, predict
how good the expert thinks it is. Same course toolchain - pandas, train/test
split, a linear baseline, then an MLP - but the target is a number, so the
metric is R squared rather than accuracy.

The interesting part is what the network has to figure out on its own. The
expert's score is a weighted sum of stack height, holes, bumpiness, wells and
transitions. None of those are inputs here - the network sees 240 raw cells and
has to construct those concepts itself in order to fit the target at all.

Exports value-model.json for src/tetris-policy.js to play with.
"""

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/values.csv")
parser.add_argument("--model", default="value-model.json")
parser.add_argument("--hidden", default="128,64")
parser.add_argument("--max-iter", type=int, default=80)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

print(f"\nloading {args.data}")
df = pd.read_csv(args.data)
X = df.drop(columns=["target"]).to_numpy(dtype=np.float32)
y = df["target"].to_numpy(dtype=np.float32)

print(f"  rows      {len(df):,}")
print(f"  features  {X.shape[1]} (240 board cells + lines cleared)")
print(f"  target    mean {y.mean():.1f}, std {y.std():.1f}, range [{y.min():.1f}, {y.max():.1f}]")

# Standardise the target. We only ever compare scores to each other when
# playing, so any monotonic rescaling is harmless - and gradient descent is far
# better behaved when the thing it is fitting has unit variance.
y_mean, y_std = float(y.mean()), float(y.std())
y_scaled = (y - y_mean) / y_std

X_train, X_test, y_train, y_test = train_test_split(
    X, y_scaled, test_size=0.2, random_state=args.seed
)
print(f"\ntrain {len(X_train):,}   test {len(X_test):,}")

print("\ntraining")
print("--------")

lin = LinearRegression().fit(X_train, y_train)
print(f"  linear regression    test R2  {r2_score(y_test, lin.predict(X_test)):6.3f}")

hidden = tuple(int(h) for h in args.hidden.split(","))
mlp = MLPRegressor(
    hidden_layer_sizes=hidden,
    activation="relu",
    solver="adam",
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=args.max_iter,
    early_stopping=True,
    n_iter_no_change=8,
    random_state=args.seed,
).fit(X_train, y_train)

train_r2 = r2_score(y_train, mlp.predict(X_train))
test_r2 = r2_score(y_test, mlp.predict(X_test))
print(f"  neural net {hidden}  train R2 {train_r2:6.3f}   test R2 {test_r2:6.3f}")

layers = [
    {
        "W": W.tolist(),
        "b": b.tolist(),
        "activation": "relu" if i < len(mlp.coefs_) - 1 else None,
    }
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_))
]

with open(args.model, "w") as f:
    json.dump(
        {
            "kind": "value-mlp",
            "inputSize": int(X.shape[1]),
            "hidden": list(hidden),
            "layers": layers,
            "targetMean": y_mean,
            "targetStd": y_std,
            "testR2": float(test_r2),
            "trainRows": int(len(X_train)),
        },
        f,
    )

print(f"\nwrote {args.model}")
print("\nnext: node play_policy.js --value value-model.json\n")
