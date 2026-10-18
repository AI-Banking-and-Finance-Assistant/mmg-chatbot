                                              # records the bank's anomaly model marked
    python roc_curves.py --seed 7             # different train/test split
    python roc_curves.py --out roc.png        # where to save the image (default: roc_curves.png)
    python roc_curves.py --no-show            # save the PNG only, don't open a window

Curves are computed on a held-out 25% test set. Inputs are the account figures behind the bot's rules;
age, gender and location are never used. Read-only: never changes the CSV or the database.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

from chart_common import (COLORS, LEAKAGE_NOTE, add_model_args, finish, fit_models, load_frame, make_parser, start,
                          target_text, tidy, titles)

LINESTYLES = {"Random forest": "-", "Logistic regression": "--"}


def main():
    ap = make_parser("Draw ROC curves from the banking data.", "roc_curves.png")
    add_model_args(ap)
    args = ap.parse_args()
    start(args)
    df, csv_path = load_frame(args.csv)
    fit = fit_models(df, args.target, args.seed)
    yte, labels, colors = fit["yte"], fit["labels"], fit["colors"]

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    if args.target == "status":
        # one-vs-rest: each status against the other two
        for name, model in fit["models"].items():
            proba = model.predict_proba(fit["Xte"])
            for k, cls in enumerate(model.classes_):
                fpr, tpr, _ = roc_curve(yte == cls, proba[:, k])
                ax.plot(fpr, tpr, color=colors[cls], ls=LINESTYLES[name], lw=2,
                        label=f"{labels[cls]}, {name} (AUC {auc(fpr, tpr):.3f})")
    else:
        for name, model in fit["models"].items():
            fpr, tpr, _ = roc_curve(yte, model.predict_proba(fit["Xte"])[:, 1])
            ax.plot(fpr, tpr, ls=LINESTYLES[name], lw=2, label=f"{name} (AUC {auc(fpr, tpr):.3f})",
                    color=COLORS["NEUTRAL"] if name == "Random forest" else COLORS["EARLY WARNING"])
        fpr, tpr, _ = roc_curve(yte, fit["score_te"])                      # the bot's own rule score as a baseline
        ax.plot(fpr, tpr, color=COLORS["FINANCIAL DISTRESS"], lw=2, ls=":", label=f"Bot rule score (AUC {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="-.", label="Chance (AUC 0.500)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.03)
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    tidy(ax)

    note = (LEAKAGE_NOTE if args.target == "status"
            else "Anomaly labels come from the bank's separate model, so this is a genuine agreement check.")
    sub = (f"Target: {target_text(args.target)}. {len(yte):,} held-out customers from {csv_path.name}.\n{note}")
    titles(fig, "ROC curves", sub)
    finish(fig, args)


if __name__ == "__main__":
    main()
