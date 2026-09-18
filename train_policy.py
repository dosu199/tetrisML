"""
train_policy.py - Phase 2. Teach a model to imitate the Tetris bot.

    python3 train_policy.py --data data/moves.csv
    python3 train_policy.py --data data/moves.csv --hidden 256,128 --max-iter 60

This is deliberately written with the tools from the ZTM course - pandas to
load, a train/test split, a dummy baseline, a linear model, then a neural
network - because the point of this phase is that your Tetris project and your
course finally become the same activity.

What the problem is, stated plainly:

    input   254 numbers: 240 board cells (1 = occupied), a 7-wide one-hot for
            the piece in play, and another for the piece coming next
    output  one of 40 classes, where class = rotation * 10 + column

There is nothing Tetris-specific about that. It is ordinary multiclass
classification on a table, and every technique in the course applies unchanged.

Exports model.json, which src/tetris-policy.js loads to actually play.
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="data/moves.csv")
parser.add_argument("--model", default="model.json")
parser.add_argument("--hidden", default="256,128", help="comma-separated layer sizes")
parser.add_argument("--max-iter", type=int, default=60)
parser.add_argument("--sample", type=int, default=0, help="subsample N rows for a quick run")
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ---------------------------------------------------------------- load

print(f"\nloading {args.data}")
df = pd.read_csv(args.data)
if args.sample and args.sample < len(df):
    df = df.sample(args.sample, random_state=args.seed)

# piece_id / next_id duplicate the one-hot columns; they are in the CSV for
# readability when you open it, not for the model.
feature_cols = [c for c in df.columns if c not in ("piece_id", "next_id", "move_class")]
X = df[feature_cols].to_numpy(dtype=np.float32)
y = df["move_class"].to_numpy()

print(f"  rows      {len(df):,}")
print(f"  features  {X.shape[1]}")
print(f"  classes   {len(np.unique(y))} of a possible 40 seen in the data")

# How lopsided is the label distribution? If one move dominates, high accuracy
# might mean nothing - which is exactly what the dummy baseline below checks.
counts = pd.Series(y).value_counts()
print(f"  most common move accounts for {counts.iloc[0] / len(y):.1%} of rows")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, random_state=args.seed
)
print(f"\ntrain {len(X_train):,}   test {len(X_test):,}")

# ---------------------------------------------------------------- models

results = {}


def evaluate(name, model, fit=True):
    t0 = time.time()
    if fit:
        model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = (train_acc, test_acc, time.time() - t0)
    print(f"  {name:<22} train {train_acc:6.1%}   test {test_acc:6.1%}   ({time.time() - t0:.0f}s)")
    return model


print("\ntraining")
print("--------")

# The floor. Always predict the single most common move. Any model that cannot
# beat this has learned nothing at all.
evaluate("baseline (most frequent)", DummyClassifier(strategy="most_frequent"))

# A linear model. Can it separate 40 classes with a straight cut through
# 254-dimensional space? Mostly not - which is the argument for the network.
evaluate("logistic regression", LogisticRegression(max_iter=300))

hidden = tuple(int(h) for h in args.hidden.split(","))
mlp = evaluate(
    f"neural net {hidden}",
    MLPClassifier(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=args.max_iter,
        early_stopping=True,
        n_iter_no_change=6,
        random_state=args.seed,
        verbose=False,
    ),
)

# Accuracy alone undersells a policy. In play we only ever choose among the
# moves that are legal on this board, so being "nearly right" is often good
# enough - top-3 accuracy is a better proxy for how it will behave.
proba = mlp.predict_proba(X_test)
top3 = top_k_accuracy_score(y_test, proba, k=3, labels=mlp.classes_)
top5 = top_k_accuracy_score(y_test, proba, k=5, labels=mlp.classes_)
print(f"\n  neural net top-3 accuracy  {top3:.1%}")
print(f"  neural net top-5 accuracy  {top5:.1%}")

# Where it goes wrong is more interesting than how often.
pred = mlp.predict(X_test)
wrong = pred != y_test
if wrong.any():
    rot_true, rot_pred = y_test[wrong] // 10, pred[wrong] // 10
    col_true, col_pred = y_test[wrong] % 10, pred[wrong] % 10
    print(f"\n  of the {wrong.sum():,} mistakes:")
    print(f"    right rotation, wrong column   {(rot_true == rot_pred).mean():.1%}")
    print(f"    off by one column              {(np.abs(col_true - col_pred) == 1).mean():.1%}")

# ---------------------------------------------------------------- export

layers = []
for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
    layers.append(
        {
            "W": W.tolist(),
            "b": b.tolist(),
            # Hidden layers use ReLU; the output layer is left raw because the
            # JS side only ever takes an argmax, and softmax preserves order.
            "activation": "relu" if i < len(mlp.coefs_) - 1 else None,
        }
    )

with open(args.model, "w") as f:
    json.dump(
        {
            "kind": "mlp",
            "inputSize": int(X.shape[1]),
            "hidden": list(hidden),
            "classes": [int(c) for c in mlp.classes_],
            "scaler": None,  # inputs are already 0/1
            "layers": layers,
            "testAccuracy": float(results[f"neural net {hidden}"][1]),
            "top3Accuracy": float(top3),
            "trainRows": int(len(X_train)),
            "dataFile": args.data,
        },
        f,
    )

print(f"\nwrote {args.model}")
print("\nnext: node play_policy.js   <- the number that actually matters\n")
