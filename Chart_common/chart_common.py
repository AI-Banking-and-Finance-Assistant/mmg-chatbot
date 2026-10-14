#!/usr/bin/env python3
"""
Shared helpers for the chart scripts (nlp_feedback.py, roc_curves.py, feature_importance.py,
demographics.py, confusion_matrix.py, class_distributions.py).

Keep this file in the same folder as bank_distress_bot.py. It reuses the bot's own scoring
(bank_distress_bot.assess), so every chart matches what the assistant reports. Like the bot, the
models here never use age, gender or location as inputs. Nothing here changes the CSV or the database.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bank_distress_bot as bot

COLORS = {"NO FLAG": "#2c6fbb", "EARLY WARNING": "#e69f00", "FINANCIAL DISTRESS": "#c0392b",
          "NEUTRAL": "#2c6fbb", "GREY": "#9aa0a6"}
STATUS_ORDER = ["NO FLAG", "EARLY WARNING", "FINANCIAL DISTRESS"]
STATUS_LABEL = {"NO FLAG": "No flag", "EARLY WARNING": "Early warning", "FINANCIAL DISTRESS": "Financial distress"}

# Model inputs: the raw account figures behind the rules. Never age, gender, city or other personal traits.
MODEL_FEATURES = {
    "balance": "Account balance",
    "balance_after": "Balance after last transaction",
    "balance_drop": "Balance decline (fraction)",
    "cc_balance": "Credit-card balance",
    "credit_limit": "Credit limit",
    "util": "Credit utilisation",
    "min_due": "Minimum payment due",
    "days_late": "Days late on card payment",
    "loan_active_amount": "Active loan amount",
    "loan_ratio": "Loan-to-balance ratio",
    "tx_amount": "Latest transaction amount",
}

LEAKAGE_NOTE = ("Flags are produced by fixed rules from these same inputs, so a model reproducing them shows how the "
                "rules behave, not real-world predictive accuracy.")


# ----------------------------------------------------------------------------
# Command line and output
# ----------------------------------------------------------------------------
def make_parser(description: str, default_out: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", default=bot.os.getenv("DATA_FILE", bot.DEFAULT_CSV),
                    help="bank database CSV (default: same file the assistant uses)")
    ap.add_argument("--out", default=default_out, help=f"image file to save (default: {default_out})")
    ap.add_argument("--no-show", action="store_true", help="save the image without opening a window")
    return ap


def add_model_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--target", choices=("status", "anomaly"), default="status",
                    help="status = NO FLAG / EARLY WARNING / FINANCIAL DISTRESS (default); "
                         "anomaly = the bank's existing anomaly marker")
    ap.add_argument("--seed", type=int, default=42, help="random seed for the train/test split (default: 42)")


def finish(fig: plt.Figure, args) -> None:
    out = Path(args.out).expanduser()
    fig.savefig(out, dpi=150)
    print(f"Chart saved to {out.resolve()}")
    if not args.no_show:
        try:
            plt.show()
        except Exception as e:                      # e.g. no display available
            print(f"(Could not open a window: {type(e).__name__}. The image was still saved.)", file=sys.stderr)


def start(args) -> None:
    if args.no_show:
        plt.switch_backend("Agg")


def titles(fig: plt.Figure, title: str, subtitle: str) -> None:
    """Bold title plus a wrapped grey subtitle, with the plot area pushed down to make room."""
    fw, fh = fig.get_size_inches()
    wrapped = "\n".join(textwrap.fill(p, width=int(fw * 12)) for p in subtitle.split("\n"))
    lines = wrapped.count("\n") + 1
    fig.suptitle(title, x=0.01, y=1 - 0.08 / fh, ha="left", va="top", fontsize=14, fontweight="bold")
    fig.text(0.01, 1 - 0.45 / fh, wrapped, fontsize=9, color="#666666", va="top", linespacing=1.3)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.6 + 0.17 * lines) / fh))


def tidy(ax) -> None:
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_frame(csv_arg: str):
    """One row per customer: bot's assessment (status, score, metrics) joined with the raw CSV columns."""
    csv_path = bot.resolve_path(csv_arg)
    if not csv_path.exists():
        sys.exit(f"Data file not found: {csv_path}\nPlace {bot.DEFAULT_CSV} next to this script "
                 f"or pass its path: python <script>.py path/to/file.csv")
    assessments, _ = bot.load_assessments(str(csv_path))
    raw = pd.read_csv(csv_path)
    for c in bot.DATE_COLS:
        raw[c] = pd.to_datetime(raw[c])
    raw = raw.drop_duplicates("Customer ID", keep="last").set_index("Customer ID")   # same rule as the bot
    rows = [dict(id=cid, status=a["status"], score=a["score"], anomaly=bool(a["anomaly"]),
                 n_signals=len(a["signals"]), codes=[s["code"] for s in a["signals"]], **a["metrics"])
            for cid, a in assessments.items()]
    df = pd.DataFrame(rows).set_index("id")
    df["balance_drop"] = np.where(df["balance_after"] > 0, (df["balance_after"] - df["balance"]) / df["balance_after"], 0.0)
    df["loan_ratio"] = np.where((df["loan_active_amount"] > 0) & (df["balance"] > 0),
                                df["loan_active_amount"] / df["balance"], 0.0)
    df = df.join(raw, how="left")
    if df.empty:
        sys.exit("The data file has no customers to plot.")
    return df, csv_path


def need_columns(df: pd.DataFrame, *names: str) -> list[str]:
    """Find columns case-insensitively; exit with a clear message if any are missing."""
    lookup = {c.lower().strip(): c for c in df.columns}
    found, missing = [], []
    for n in names:
        (found if n.lower() in lookup else missing).append(lookup.get(n.lower(), n))
    if missing:
        sys.exit(f"Column(s) not found in the data file: {', '.join(missing)}.\n"
                 f"Columns available: {', '.join(map(str, df.columns))}")
    return found


# ----------------------------------------------------------------------------
# Models (used by roc_curves.py, feature_importance.py, confusion_matrix.py)
# ----------------------------------------------------------------------------
def fit_models(df: pd.DataFrame, target: str, seed: int) -> dict:
    """Train a random forest and a logistic regression on the bot's raw account features."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = df[list(MODEL_FEATURES)].astype(float)
    if target == "status":
        names = [s for s in STATUS_ORDER if (df["status"] == s).sum() >= 2]
        keep = df["status"].isin(names)
        y = df.loc[keep, "status"].map({s: i for i, s in enumerate(names)}).to_numpy()
        labels = [STATUS_LABEL[s] for s in names]
        colors = [COLORS[s] for s in names]
    else:
        keep = pd.Series(True, index=df.index)
        y = df["anomaly"].astype(int).to_numpy()
        labels, colors = ["Normal", "Anomaly"], [COLORS["NEUTRAL"], COLORS["FINANCIAL DISTRESS"]]
        if (y == 1).sum() < 2 or (y == 0).sum() < 2:
            sys.exit("Fewer than 2 records in one anomaly class, so the model cannot be evaluated.")
    X = X[keep]
    if len(labels) < 2:
        sys.exit("Only one status class has enough customers, so the model cannot be evaluated.")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    models = {
        "Random forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    }
    for m in models.values():
        m.fit(Xtr, ytr)
    return dict(models=models, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, labels=labels, colors=colors,
                score_te=df.loc[Xte.index, "score"].to_numpy(), target=target)


def target_text(target: str) -> str:
    return ("customer status (no flag / early warning / financial distress)" if target == "status"
            else "the bank's anomaly marker")


if __name__ == "__main__":
    sys.exit(
        "chart_common.py has no chart of its own - it's the shared library the other chart scripts import "
        "(bank data loading, styling, and the ML helpers). That's why running it directly produced nothing.\n\n"
        "Run one of these instead, from this same folder:\n"
        "  python class_distributions.py\n"
        "  python confusion_matrix.py\n"
        "  python demographics.py\n"
        "  python feature_importance.py\n"
        "  python nlp_feedback.py\n"
        "  python roc_curves.py\n"
        "  python time_series.py   (doesn't use chart_common.py, but belongs with the rest)\n\n"
        "Add --help to any of them to see its options."
    )
