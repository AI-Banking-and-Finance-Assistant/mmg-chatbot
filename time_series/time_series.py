#!/usr/bin/env python3
"""
Time-series chart for the Financial-Distress Early-Warning data
================================================================

Run from the folder that contains bank_distress_bot.py and the banking CSV:

    python time_series.py                       # flagged customers per month (default)
    python time_series.py --metric score        # average risk score over time
    python time_series.py --metric balance      # average account balance over time
    python time_series.py --metric volume       # total transaction amount over time
    python time_series.py --metric history      # flagged customers per data snapshot (needs the bot's .db)
    python time_series.py --freq W              # weekly instead of monthly (D = daily, W = weekly, M = monthly)
    python time_series.py --out chart.png       # where to save the image (default: time_series.png)
    python time_series.py --no-show             # save the PNG only, don't open a window

It reuses the bot's own scoring rules (bank_distress_bot.assess), so the numbers on the chart always
match what the assistant reports. Like the bot, it never uses age, gender or location. It only reads
the data; it never changes the CSV or the database.

Note: the first four metrics are grouped by each customer's latest "Transaction Date". The "history"
metric instead plots the snapshots the bot saves in distress_bot.db each time it sees a new data extract.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import bank_distress_bot as bot

COLORS = {"EARLY WARNING": "#e69f00", "FINANCIAL DISTRESS": "#c0392b", "NEUTRAL": "#2c6fbb"}
METRICS = ("flagged", "score", "balance", "volume", "history")
FREQ_NAMES = {"D": "day", "W": "week", "M": "month"}


# ----------------------------------------------------------------------------
# Building the data
# ----------------------------------------------------------------------------
def build_frame(assessments: dict) -> pd.DataFrame:
    """One row per customer: status, score and the latest-transaction figures."""
    rows = [dict(status=a["status"], score=a["score"],
                 date=pd.to_datetime(a["metrics"]["tx_date"], errors="coerce"),  # tx_date can be None
                 balance=a["metrics"]["balance"], tx_amount=a["metrics"]["tx_amount"])
            for a in assessments.values()]
    df = pd.DataFrame(rows)
    return df.dropna(subset=["date"])  # customers with no usable transaction date can't be placed on the timeline


def series_from_csv(df: pd.DataFrame, metric: str, freq: str):
    """Return (dict of label -> Series indexed by timestamp, title, y-axis label, colours)."""
    period = df["date"].dt.to_period(freq)
    full = pd.period_range(period.min(), period.max(), freq=freq)
    unit = FREQ_NAMES[freq]

    def to_ts(s: pd.Series) -> pd.Series:
        s = s.reindex(full)
        s.index = s.index.to_timestamp()
        return s

    if metric == "flagged":
        counts = df.groupby([period, "status"]).size().unstack(fill_value=0)
        out, colors = {}, {}
        for status, label in (("EARLY WARNING", "Early warning"), ("FINANCIAL DISTRESS", "Financial distress")):
            col = counts[status] if status in counts else pd.Series(0, index=counts.index)
            out[label] = to_ts(col).fillna(0)
            colors[label] = COLORS[status]
        return out, f"Flagged customers per {unit}", "Customers", colors
    if metric == "score":
        s = to_ts(df.groupby(period)["score"].mean())
        return {"Average risk score": s}, f"Average risk score per {unit}", "Score (points)", {}
    if metric == "balance":
        s = to_ts(df.groupby(period)["balance"].mean())
        return ({"Average account balance": s}, f"Average account balance per {unit}",
                f"Balance ({bot.CONFIG['currency']})", {})
    if metric == "volume":
        s = to_ts(df.groupby(period)["tx_amount"].sum()).fillna(0)
        return ({"Total transaction amount": s}, f"Total transaction amount per {unit}",
                f"Amount ({bot.CONFIG['currency']})", {})
    raise ValueError(metric)


def series_from_snapshots(db_path: Path):
    """Flagged customers at each snapshot the bot has saved. None if there is no usable history."""
    if not db_path.exists():
        return None
    con = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)      # read-only: never creates/changes the DB
    try:
        rows = con.execute(
            "SELECT s.snapshot_id, s.taken_at, c.status, COUNT(*) FROM snapshot_customers c "
            "JOIN snapshots s USING(snapshot_id) GROUP BY s.snapshot_id, c.status ORDER BY s.snapshot_id").fetchall()
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["sid", "taken_at", "status", "n"])
    df["taken_at"] = pd.to_datetime(df["taken_at"])
    wide = df.pivot_table(index=["sid", "taken_at"], columns="status", values="n", fill_value=0).reset_index()
    out, colors = {}, {}
    for status, label in (("EARLY WARNING", "Early warning"), ("FINANCIAL DISTRESS", "Financial distress")):
        vals = wide[status] if status in wide else pd.Series(0, index=wide.index)
        out[label] = pd.Series(vals.values, index=wide["taken_at"])
        colors[label] = COLORS[status]
    return out, "Flagged customers per data snapshot", "Customers", colors


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw(series: dict, title: str, ylabel: str, colors: dict, subtitle: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, s in series.items():
        ax.plot(s.index, s.values, label=label, color=colors.get(label, COLORS["NEUTRAL"]), linewidth=2,
                marker="o" if len(s) <= 60 else None, markersize=4)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left", pad=24)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=9, color="#666666", va="bottom")
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    if len(series) > 1:
        ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description="Draw a time-series chart from the banking data.")
    ap.add_argument("csv", nargs="?", default=bot.os.getenv("DATA_FILE", bot.DEFAULT_CSV),
                    help="bank database CSV (default: same file the assistant uses)")
    ap.add_argument("--metric", choices=METRICS, default="flagged", help="what to plot (default: flagged)")
    ap.add_argument("--freq", choices=list(FREQ_NAMES), default="M",
                    help="grouping period: D=day, W=week, M=month (default: M)")
    ap.add_argument("--db", default=bot.os.getenv("DB_PATH", "distress_bot.db"),
                    help="the assistant's SQLite file (only used for --metric history)")
    ap.add_argument("--out", default="time_series.png", help="image file to save (default: time_series.png)")
    ap.add_argument("--no-show", action="store_true", help="save the image without opening a window")
    args = ap.parse_args()

    if args.metric == "history":
        db_path = bot.resolve_path(args.db)
        result = series_from_snapshots(db_path)
        if result is None or len(next(iter(result[0].values()))) < 2:
            sys.exit(f"Not enough snapshot history in {db_path} yet. The assistant saves one snapshot each time "
                     f"it loads a changed data file, so you need at least two. Try another --metric for now.")
        series, title, ylabel, colors = result
        subtitle = f"Source: {db_path.name}"
    else:
        csv_path = bot.resolve_path(args.csv)
        if not csv_path.exists():
            sys.exit(f"Data file not found: {csv_path}\nPlace {bot.DEFAULT_CSV} next to this script "
                     f"or pass its path: python time_series.py path/to/file.csv")
        assessments, _ = bot.load_assessments(str(csv_path))
        df = build_frame(assessments)
        if df.empty:
            sys.exit("The data file has no customers to plot.")
        series, title, ylabel, colors = series_from_csv(df, args.metric, args.freq)
        subtitle = (f"{len(df):,} customers from {csv_path.name}, grouped by latest transaction date "
                    f"({df['date'].min():%d %b %Y} to {df['date'].max():%d %b %Y})")

    show = not args.no_show
    if not show:
        matplotlib.use("Agg")
    fig = draw(series, title, ylabel, colors, subtitle)
    out = Path(args.out).expanduser()
    fig.savefig(out, dpi=150)
    print(f"Chart saved to {out.resolve()}")
    if show:
        try:
            plt.show()
        except Exception as e:                      # e.g. no display available
            print(f"(Could not open a window: {type(e).__name__}. The image was still saved.)", file=sys.stderr)


if __name__ == "__main__":
    main()
