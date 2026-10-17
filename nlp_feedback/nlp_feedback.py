#!/usr/bin/env python3
"""
Customer-feedback chart for the Financial-Distress Early-Warning data
=====================================================================

Run from the folder that contains bank_distress_bot.py, chart_common.py and the banking CSV:

    python nlp_feedback.py                          # feedback types, resolution and timing by customer status
    python nlp_feedback.py --text-col "Comments"    # also mine free-text feedback (top terms, flagged vs not)
    python nlp_feedback.py --freq W                 # weekly instead of monthly volume (D = daily, W, M)
    python nlp_feedback.py --out feedback.png       # where to save the image (default: nlp_feedback.png)
    python nlp_feedback.py --no-show                # save the PNG only, don't open a window

Panels: feedback-type mix by customer status, resolution outcome by feedback type, days taken to resolve, and
either top terms in free text (if the data has a text column, found automatically or via --text-col) or
feedback volume over time (if it doesn't). The standard banking extract only has categorical feedback
(type and resolution), so the term analysis stays off until a text column exists. Read-only: never changes
the CSV or the database.
"""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import pandas as pd

from chart_common import (COLORS, STATUS_LABEL, finish, load_frame, make_parser, need_columns, start, tidy, titles)

FREQ_NAMES = {"D": "day", "W": "week", "M": "month"}
TEXT_HINTS = ("comment", "text", "message", "description", "review")
PALETTE = ["#2c6fbb", "#e69f00", "#3aa17e", "#8e5ea2", "#c0392b", "#9aa0a6"]


def find_text_column(df: pd.DataFrame, requested: str | None):
    if requested:
        return need_columns(df, requested)[0]
    for c in df.columns:
        if df[c].dtype == object and any(h in str(c).lower() for h in TEXT_HINTS):
            if df[c].dropna().astype(str).str.split().str.len().median() >= 3:     # looks like sentences, not codes
                return c
    return None


def top_terms(df: pd.DataFrame, text_col: str, n_terms: int = 12):
    """Terms most common in flagged customers' feedback, with their share of flagged vs unflagged documents."""
    from sklearn.feature_extraction.text import CountVectorizer
    d = df.dropna(subset=[text_col])
    flagged = d["status"] != "NO FLAG"
    if flagged.sum() == 0 or (~flagged).sum() == 0:
        return None
    vec = CountVectorizer(stop_words="english", binary=True, min_df=2, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    try:
        m = vec.fit_transform(d[text_col].astype(str))
    except ValueError:                                                     # empty vocabulary
        return None
    terms = pd.Index(vec.get_feature_names_out())
    f_share = pd.Series(m[flagged.to_numpy()].mean(axis=0).A1, index=terms)
    u_share = pd.Series(m[(~flagged).to_numpy()].mean(axis=0).A1, index=terms)
    top = f_share.sort_values(ascending=False).head(n_terms).index
    return pd.DataFrame({"flagged": f_share[top], "not flagged": u_share[top]}), int(flagged.sum()), int((~flagged).sum())


def main():
    ap = make_parser("Draw customer-feedback charts from the banking data.", "nlp_feedback.png")
    ap.add_argument("--text-col", default=None, help="name of a free-text feedback column (auto-detected if present)")
    ap.add_argument("--freq", choices=list(FREQ_NAMES), default="M", help="volume grouping: D, W or M (default: M)")
    args = ap.parse_args()
    start(args)
    df, csv_path = load_frame(args.csv)
    type_c, res_c, fdate_c, rdate_c = need_columns(df, "Feedback Type", "Resolution Status", "Feedback Date",
                                                   "Resolution Date")
    fb = df.dropna(subset=[type_c]).copy()
    if fb.empty:
        sys.exit("The data file has no feedback records to plot.")
    types = sorted(fb[type_c].unique(), key=str)
    tcolor = {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(types)}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (a1, a2), (a3, a4) = axes

    # 1. Feedback-type mix within each customer status
    statuses = [s for s in ("NO FLAG", "EARLY WARNING", "FINANCIAL DISTRESS") if (fb["status"] == s).any()]
    mix = pd.crosstab(fb["status"], fb[type_c]).reindex(statuses)[types]
    share = mix.div(mix.sum(axis=1), axis=0)
    width = 0.8 / len(types)
    for j, t in enumerate(types):
        xs = [i + (j - (len(types) - 1) / 2) * width for i in range(len(statuses))]
        a1.bar(xs, share[t].values, width=width, color=tcolor[t], label=str(t))
    a1.set_xticks(range(len(statuses)))
    a1.set_xticklabels([f"{STATUS_LABEL[s]}\nn={int(mix.loc[s].sum()):,}" for s in statuses])
    a1.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a1.set_ylabel("Share of that status's feedback")
    a1.set_title("Feedback type by customer status", loc="left", fontweight="bold")
    a1.set_ylim(0, share.to_numpy().max() * 1.2)
    a1.legend(frameon=False, fontsize=8, ncol=len(types), loc="upper center")
    tidy(a1)

    # 2. Resolution outcome by feedback type (100% stacked)
    res = pd.crosstab(fb[type_c], fb[res_c]).reindex(types)
    rshare = res.div(res.sum(axis=1).replace(0, 1), axis=0)
    left = pd.Series(0.0, index=rshare.index)
    for k, r in enumerate(rshare.columns):
        a2.barh([str(t) for t in rshare.index], rshare[r], left=left, color=PALETTE[(k + 2) % len(PALETTE)], label=str(r))
        left += rshare[r]
    a2.set_xlim(0, 1)
    a2.invert_yaxis()
    a2.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a2.set_xlabel("Share of feedback")
    a2.set_title("Resolution outcome by feedback type", loc="left", fontweight="bold")
    a2.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    tidy(a2)

    # 3. Days to resolve
    fb["days"] = (fb[rdate_c] - fb[fdate_c]).dt.days
    groups = [fb.loc[(fb[type_c] == t) & (fb["days"] >= 0), "days"].dropna() for t in types]
    if any(len(g) for g in groups):
        bp = a3.boxplot([g if len(g) else [float("nan")] for g in groups], patch_artist=True, showfliers=False,
                        medianprops=dict(color="#222222"))
        for patch, t in zip(bp["boxes"], types):
            patch.set_facecolor(tcolor[t])
        a3.set_xticks(range(1, len(types) + 1))
        a3.set_xticklabels([f"{t}\nn={len(g):,}" for t, g in zip(types, groups)])
        a3.set_ylabel("Days from feedback to resolution")
    else:
        a3.text(0.5, 0.5, "No resolved feedback with valid dates", ha="center", va="center", transform=a3.transAxes)
    a3.set_title("Time to resolve", loc="left", fontweight="bold")
    tidy(a3)

    # 4. Free-text terms if available, otherwise volume over time
    text_col = find_text_column(fb, args.text_col)
    terms = top_terms(fb, text_col) if text_col else None
    if terms is not None:
        tdf, n_f, n_u = terms
        y = range(len(tdf))
        a4.barh([i - 0.2 for i in y], tdf["flagged"], height=0.4, color=COLORS["FINANCIAL DISTRESS"],
                label=f"Flagged customers (n={n_f:,})")
        a4.barh([i + 0.2 for i in y], tdf["not flagged"], height=0.4, color=COLORS["NEUTRAL"],
                label=f"Not flagged (n={n_u:,})")
        a4.set_yticks(list(y))
        a4.set_yticklabels(tdf.index)
        a4.invert_yaxis()
        a4.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        a4.set_xlabel("Share of feedback mentioning the term")
        a4.set_title(f"Top terms in '{text_col}'", loc="left", fontweight="bold")
        a4.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    else:
        period = fb[fdate_c].dt.to_period(args.freq)
        vol = fb.groupby([period, type_c]).size().unstack(fill_value=0)
        vol = vol.reindex(pd.period_range(vol.index.min(), vol.index.max(), freq=args.freq), fill_value=0)
        idx = vol.index.to_timestamp()
        for t in types:
            a4.plot(idx, vol[t].values, color=tcolor[t], lw=2, label=str(t),
                    marker="o" if len(vol) <= 60 else None, markersize=4)
        a4.set_ylabel("Feedback items")
        a4.set_title(f"Feedback volume per {FREQ_NAMES[args.freq]}", loc="left", fontweight="bold")
        a4.legend(frameon=False, fontsize=8)
        for lbl in a4.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")
    tidy(a4)

    flagged = (fb["status"] != "NO FLAG").mean()
    text_note = (f"Free text analysed from '{text_col}'." if terms is not None
                 else "No free-text column found, so this shows categorical feedback only (use --text-col to add one).")
    titles(fig, "Customer feedback",
           f"{len(fb):,} feedback records from {csv_path.name} ({flagged:.0%} from flagged customers). {text_note}")
    finish(fig, args)


if __name__ == "__main__":
    main()
