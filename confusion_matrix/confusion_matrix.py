#!/usr/bin/env python3
"""
Confusion-matrix chart for the Financial-Distress Early-Warning data
====================================================================

Run from the folder that contains bank_distress_bot.py, chart_common.py and the banking CSV:

    python confusion_matrix.py                      # random forest predicting customer status
    python confusion_matrix.py --model logistic     # logistic regression instead (forest is the default)
    python confusion_matrix.py --target anomaly     # predicting the bank's anomaly marker
    python confusion_matrix.py --seed 7             # different train/test split
    python confusion_matrix.py --out cm.png         # where to save the image (default: confusion_matrix.png)
    python confusion_matrix.py --no-show            # save the PNG only, don't open a window

Left: counts on the held-out 25% test set. Right: the same matrix as a share of each true class (the diagonal
is recall). Inputs are the account figures behind the bot's rules; age, gender and location are never used.
Read-only: never changes the CSV or the database.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

from chart_common import (LEAKAGE_NOTE, add_model_args, finish, fit_models, load_frame, make_parser, start,
                          target_text, tidy, titles)


def heatmap(ax, mat: np.ndarray, labels: list, fmt, title: str, cmap: str = "Blues") -> None:
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max() if mat.max() > 0 else 1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, loc="left", fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, fmt(mat[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if mat[i, j] > 0.55 * im.norm.vmax else "#222222")
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)


def main():
    ap = make_parser("Draw a confusion matrix from the banking data.", "confusion_matrix.png")
    add_model_args(ap)
    ap.add_argument("--model", choices=("forest", "logistic"), default="forest", help="model to evaluate (default: forest)")
    args = ap.parse_args()
    start(args)
    df, csv_path = load_frame(args.csv)
    fit = fit_models(df, args.target, args.seed)
    name = "Random forest" if args.model == "forest" else "Logistic regression"
    pred = fit["models"][name].predict(fit["Xte"])
    yte, labels = fit["yte"], fit["labels"]

    cm = confusion_matrix(yte, pred, labels=range(len(labels)))
    recall = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    heatmap(a1, cm, labels, lambda v: f"{int(v):,}", "Counts")
    heatmap(a2, recall, labels, lambda v: f"{v:.1%}", "Share of each actual class (recall)", cmap="Oranges")

    note = LEAKAGE_NOTE if args.target == "status" else "Anomaly labels come from the bank's separate model."
    sub = (f"{name} predicting {target_text(args.target)}. Accuracy {accuracy_score(yte, pred):.1%}, balanced accuracy "
           f"{balanced_accuracy_score(yte, pred):.1%} on {len(yte):,} held-out customers from {csv_path.name}.\n{note}")
    titles(fig, "Confusion matrix", sub)
    finish(fig, args)


if __name__ == "__main__":
    main()
