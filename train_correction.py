"""Trains and compares correction models. Run with --demo to test it before you have data.

    python train_correction.py --demo      # synthetic rows, sanity check only
    python train_correction.py             # your real data/dataset.csv
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
import config

FEATS = ["raw_long", "raw_short", "distance_cm", "long_px", "short_px", "area_px", "yolo_conf"]


def synthetic(n_obj=12, per=3, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_obj):
        tl, ts = rng.uniform(8, 35), rng.uniform(5, 25)
        for c in range(per):
            d = rng.uniform(40, 100)
            bias = 1 + 0.12 - 0.0012 * d
            rl = tl * bias + rng.normal(0, 0.8)
            rs = ts * bias + rng.normal(0, 0.8)
            rows.append(dict(object_id=f"obj{i:02d}", capture=c, true_long=tl, true_short=ts,
                             raw_long=rl, raw_short=rs, distance_cm=d,
                             long_px=rl * 30, short_px=rs * 30, area_px=rl * rs * 700,
                             yolo_conf=rng.uniform(0.5, 0.95)))
    return pd.DataFrame(rows)


def cv_mae(df, make_model, target, k=5):
    X, y = df[FEATS].values, df[target].values
    groups = df["object_id"].values
    n = min(k, len(np.unique(groups)))
    preds = np.zeros_like(y, dtype=float)
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        m = make_model()
        m.fit(X[tr], y[tr] - X[tr][:, FEATS.index(target.replace("true", "raw"))])
        base = X[te][:, FEATS.index(target.replace("true", "raw"))]
        preds[te] = base + m.predict(X[te])
    return float(np.mean(np.abs(preds - y))), float(np.mean(np.abs(preds - y) / y) * 100)


def main(demo):
    if demo:
        df = synthetic()
        print("using SYNTHETIC data - numbers below are a pipeline check, not a result\n")
    else:
        df = pd.read_csv(config.DATASET_CSV)
        print(f"loaded {len(df)} rows over {df.object_id.nunique()} objects\n")

    import lightgbm as lgb
    makers = {
        "linear": lambda: LinearRegression(),
        "ridge": lambda: Ridge(alpha=1.0),
        "lightgbm": lambda: lgb.LGBMRegressor(num_leaves=7, min_child_samples=5,
                                              n_estimators=200, learning_rate=0.05, verbose=-1),
    }

    print(f"{'model':<12}{'long MAE':>10}{'long %':>9}{'short MAE':>11}{'short %':>9}")
    print("-" * 51)
    raw_l = float(np.mean(np.abs(df.raw_long - df.true_long)))
    raw_s = float(np.mean(np.abs(df.raw_short - df.true_short)))
    pl = float(np.mean(np.abs(df.raw_long - df.true_long) / df.true_long) * 100)
    ps = float(np.mean(np.abs(df.raw_short - df.true_short) / df.true_short) * 100)
    print(f"{'none (raw)':<12}{raw_l:>10.2f}{pl:>9.1f}{raw_s:>11.2f}{ps:>9.1f}")

    best, best_mae = None, 1e9
    for name, mk in makers.items():
        ml, pl_ = cv_mae(df, mk, "true_long")
        ms, ps_ = cv_mae(df, mk, "true_short")
        print(f"{name:<12}{ml:>10.2f}{pl_:>9.1f}{ms:>11.2f}{ps_:>9.1f}")
        if ml + ms < best_mae:
            best, best_mae = name, ml + ms

    print(f"\nbest: {best}")
    if not demo:
        os.makedirs(os.path.dirname(config.CORRECTION_MODEL), exist_ok=True)
        final = {}
        for target in ["true_long", "true_short"]:
            m = makers[best]()
            base = df[target.replace("true", "raw")].values
            m.fit(df[FEATS].values, df[target].values - base)
            final[target] = m
        joblib.dump({"models": final, "feats": FEATS, "kind": best}, config.CORRECTION_MODEL)
        print("saved", config.CORRECTION_MODEL)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    main(ap.parse_args().demo)
