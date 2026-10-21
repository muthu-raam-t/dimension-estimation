"""Produces every number and chart the results slides need.

    python evaluate.py

Reads  data/dataset.csv
Writes results/results_tables.txt   - all tables, ready to copy into the deck
       results/ablation.png         - three-stage error chart  (slide 13)
       results/scatter.png          - predicted vs true        (slide 14)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold

import config

FEATS = ["raw_long", "raw_short", "distance_cm", "long_px", "short_px",
         "area_px", "yolo_conf"]

# palette: categorical slots 1 and 2, light surface
C1, C2 = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load():
    if not os.path.exists(config.DATASET_CSV):
        sys.exit(f"{config.DATASET_CSV} not found - collect data first.")
    df = pd.read_csv(config.DATASET_CSV)
    if len(df) < 6 or df.object_id.nunique() < 3:
        sys.exit(f"Only {len(df)} rows over {df.object_id.nunique()} objects. "
                 "Collect at least 3 objects before evaluating.")
    # un-anchored estimate: measurement scales linearly with depth, so dividing
    # by the per-frame scale factor recovers what the raw depth model alone gave
    df["unanch_long"] = df.raw_long / df.scale_k.replace(0, np.nan)
    df["unanch_short"] = df.raw_short / df.scale_k.replace(0, np.nan)
    return df.dropna(subset=["unanch_long", "unanch_short"])


def oof_corrected(df):
    """Out-of-fold corrected predictions, split by object so nothing leaks."""
    import lightgbm as lgb
    out = {}
    groups = df.object_id.values
    n = min(5, df.object_id.nunique())
    for axis in ["long", "short"]:
        X = df[FEATS].values
        y = df[f"true_{axis}"].values
        base = df[f"raw_{axis}"].values
        pred = np.zeros_like(y, dtype=float)
        for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
            m = lgb.LGBMRegressor(num_leaves=7, min_child_samples=5,
                                  n_estimators=200, learning_rate=0.05,
                                  verbose=-1)
            m.fit(X[tr], y[tr] - base[tr])
            pred[te] = base[te] + m.predict(X[te])
        out[axis] = pred
    return out


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def rel(pred, true):
    return float(np.mean(np.abs(pred - true) / true) * 100)


def main():
    df = load()
    corr = oof_corrected(df)

    stages = [
        ("Raw depth, no anchoring", df.unanch_long, df.unanch_short),
        ("With marker anchoring", df.raw_long, df.raw_short),
        ("With learned correction", corr["long"], corr["short"]),
    ]

    lines = []
    def w(s=""):
        lines.append(s)
        print(s)

    w(f"Dataset: {len(df)} samples, {df.object_id.nunique()} objects, "
      f"{df['class'].nunique()} classes")
    w()
    w("TABLE 1 - Three-stage ablation   (slide 13)")
    w(f"{'Stage':<28}{'Length MAE':>12}{'Width MAE':>12}{'Rel. error':>12}")
    w("-" * 64)
    rows = []
    for name, pl, ps in stages:
        ml, ms = mae(pl, df.true_long), mae(ps, df.true_short)
        r = (rel(pl, df.true_long) + rel(ps, df.true_short)) / 2
        w(f"{name:<28}{ml:>10.2f} cm{ms:>10.2f} cm{r:>11.1f}%")
        rows.append((name, ml, ms))
    w()
    w("Base paper reference: 0.86 cm length, 0.76 cm width, 89% accuracy "
      "(two cameras)")
    w()

    w("TABLE 2 - Correction model comparison")
    import lightgbm as lgb
    makers = {
        "linear": lambda: LinearRegression(),
        "ridge": lambda: Ridge(alpha=1.0),
        "lightgbm": lambda: lgb.LGBMRegressor(num_leaves=7, min_child_samples=5,
                                              n_estimators=200,
                                              learning_rate=0.05, verbose=-1),
    }
    w(f"{'Model':<14}{'Length MAE':>12}{'Width MAE':>12}")
    w("-" * 38)
    w(f"{'none (raw)':<14}{mae(df.raw_long, df.true_long):>10.2f} cm"
      f"{mae(df.raw_short, df.true_short):>10.2f} cm")
    groups = df.object_id.values
    n = min(5, df.object_id.nunique())
    for mname, mk in makers.items():
        res = []
        for axis in ["long", "short"]:
            X, y = df[FEATS].values, df[f"true_{axis}"].values
            base = df[f"raw_{axis}"].values
            pred = np.zeros_like(y, dtype=float)
            for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
                m = mk()
                m.fit(X[tr], y[tr] - base[tr])
                pred[te] = base[te] + m.predict(X[te])
            res.append(mae(pred, y))
        w(f"{mname:<14}{res[0]:>10.2f} cm{res[1]:>10.2f} cm")
    w()

    w("TABLE 3 - Repeatability   (slide 14)")
    w(f"{'Object':<16}{'n':>4}{'Length SD':>12}{'Width SD':>12}")
    w("-" * 44)
    rep = df.groupby("object_id").agg(n=("raw_long", "size"),
                                      sd_l=("raw_long", "std"),
                                      sd_s=("raw_short", "std"))
    rep = rep[rep.n >= 2]
    for oid, r in rep.iterrows():
        w(f"{str(oid):<16}{int(r.n):>4}{r.sd_l:>10.2f} cm{r.sd_s:>10.2f} cm")
    if len(rep):
        w(f"{'MEAN':<16}{'':>4}{rep.sd_l.mean():>10.2f} cm"
          f"{rep.sd_s.mean():>10.2f} cm")
    w()

    w("TABLE 4 - Error by object class   (slide 14)")
    w(f"{'Class':<16}{'n':>4}{'Length MAE':>12}{'Width MAE':>12}")
    w("-" * 44)
    d2 = df.copy()
    d2["err_l"] = np.abs(corr["long"] - df.true_long)
    d2["err_s"] = np.abs(corr["short"] - df.true_short)
    for cls, g in d2.groupby("class"):
        w(f"{str(cls):<16}{len(g):>4}{g.err_l.mean():>10.2f} cm"
          f"{g.err_s.mean():>10.2f} cm")
    w()

    os.makedirs("results", exist_ok=True)
    with open("results/results_tables.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # ---------------------------------------------------- chart 1: ablation
    # two panels: the unanchored stage is an order of magnitude larger, so it
    # gets its own scale rather than flattening the comparison that matters
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(9.0, 4.4), gridspec_kw={"width_ratios": [1, 2]})

    def pair(ax, items, title):
        x = np.arange(len(items))
        bw = 0.36
        ln = [i[1] for i in items]
        wd = [i[2] for i in items]
        ax.bar(x - bw / 2 - 0.01, ln, bw, label="Length", color=C1,
               edgecolor=SURFACE, linewidth=2)
        ax.bar(x + bw / 2 + 0.01, wd, bw, label="Width", color=C2,
               edgecolor=SURFACE, linewidth=2)
        for xi, v in list(zip(x - bw / 2 - 0.01, ln)) + \
                     list(zip(x + bw / 2 + 0.01, wd)):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                    color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([i[0].replace("With ", "With\n")
                            .replace(", ", ",\n") for i in items], fontsize=9)
        ax.set_title(title, fontsize=10, color=INK2, pad=8)
        ax.yaxis.grid(True, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.margins(y=0.18)

    pair(axa, rows[:1], "Without scale anchoring")
    pair(axb, rows[1:], "Anchored, then corrected")
    axa.set_ylabel("Mean absolute error (cm)")
    axb.legend(frameon=False, loc="upper right")
    fig.suptitle("Each stage of the pipeline reduces measurement error",
                 fontsize=12.5, color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("results/ablation.png", dpi=200, facecolor=SURFACE)

    # ---------------------------------------------------- chart 2: scatter
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    lim = max(df.true_long.max(), corr["long"].max()) * 1.1
    ax.plot([0, lim], [0, lim], color=AXIS, linewidth=1.5, zorder=1)
    ax.scatter(df.true_long, corr["long"], s=46, color=C1, label="Length",
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.scatter(df.true_short, corr["short"], s=46, color=C2, label="Width",
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Tape-measured dimension (cm)")
    ax.set_ylabel("Predicted dimension (cm)")
    ax.set_title("Predicted against true, after correction", fontsize=12,
                 color=INK, pad=14)
    ax.grid(True, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig("results/scatter.png", dpi=200, facecolor=SURFACE)

    print("wrote results/results_tables.txt, results/ablation.png, "
          "results/scatter.png")


if __name__ == "__main__":
    main()
