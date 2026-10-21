"""
sv_eval.py -- measure dimension accuracy against tape-measure ground truth,
and print the comparison table against the base paper.

1. Photograph each parcel with the marker in frame.
2. Measure each one with a tape or calipers.
3. Write results/ground_truth.csv:

       image,length_cm,width_cm,height_cm
       box01.jpg,24.0,16.2,11.0
       box02.jpg,31.5,22.0,8.5

4. python3 sv_eval.py --csv results/ground_truth.csv --images results/shots

Reports MAE, RMSE and mean relative error per axis -- exactly the metrics the
base paper reports, so the numbers are directly comparable.

ON THE PAPER'S "89% ACCURACY": the paper does not define it explicitly. The
definition that reproduces their height numbers is
    accuracy = 1 - mean(|error| / true_value)
i.e. 1 minus the mean relative error. We print that too, under that definition,
and say so -- do not quote a single accuracy figure without stating the formula.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

import sv_config as cfg
from sv_pipeline import measure_image, load_camera

PAPER = {  # Chu et al. 2026, "new weighted", 14 test samples
    "length": dict(mae=0.86, rel=5.29),
    "width":  dict(mae=0.76, rel=8.18),
    "height": dict(mae=1.43, rel=16.62),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--method", default="auto")
    ap.add_argument("--out", default=str(cfg.RESULTS_DIR / "evaluation.csv"))
    args = ap.parse_args()

    img_dir = Path(args.images)
    rows = list(csv.DictReader(open(args.csv)))
    if not rows:
        raise SystemExit("ground truth CSV is empty")

    recs, failures = [], []
    for row in rows:
        p = img_dir / row["image"]
        if not p.exists():
            failures.append((row["image"], "image file missing"))
            continue
        img = cv2.imread(str(p))
        if img is None:
            failures.append((row["image"], "unreadable"))
            continue
        r = measure_image(img, cam=load_camera(img.shape), seg_method=args.method)
        if not r.ok:
            failures.append((row["image"], r.message))
            continue
        eL, eW, eH = r.cuboid.sorted_dims()
        tL, tW, tH = sorted([float(row["length_cm"]), float(row["width_cm"])],
                            reverse=True) + [float(row["height_cm"])]
        recs.append(dict(image=row["image"],
                         true_L=tL, true_W=tW, true_H=tH,
                         est_L=eL, est_W=eW, est_H=eH,
                         iou=r.iou, tilt=r.tilt_deg, trusted=r.trusted))

    n_total = len(rows)
    n_ok = len(recs)
    print(f"\ndetection/measurement success: {n_ok}/{n_total} "
          f"({100.0*n_ok/max(n_total,1):.1f}%)")
    print("base paper test-phase success after their brightness step: 93.3%")
    for name, why in failures:
        print(f"   FAILED {name}: {why}")
    if not recs:
        raise SystemExit("\nNo successful measurements; nothing to score.")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    print(f"\n{'axis':8s} {'MAE cm':>8s} {'RMSE cm':>9s} {'rel %':>8s}"
          f" {'paper MAE':>10s} {'paper rel%':>11s}")
    print("-" * 60)
    summary = {}
    for axis, tk, ek in (("length", "true_L", "est_L"),
                         ("width", "true_W", "est_W"),
                         ("height", "true_H", "est_H")):
        t = np.array([r[tk] for r in recs], float)
        e = np.array([r[ek] for r in recs], float)
        err = e - t
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rel = float(np.mean(np.abs(err) / np.maximum(t, 1e-9)) * 100)
        summary[axis] = dict(mae=mae, rmse=rmse, rel=rel)
        print(f"{axis:8s} {mae:8.2f} {rmse:9.2f} {rel:8.2f}"
              f" {PAPER[axis]['mae']:10.2f} {PAPER[axis]['rel']:11.2f}")

    acc = 100.0 - summary["height"]["rel"]
    print(f"\nheight 'accuracy' = 1 - mean relative error = {acc:.1f}%")
    print("base paper reports 89% for height under the same definition.")
    print(f"\nper-image results -> {args.out}")

    better = [a for a in summary if summary[a]["mae"] < PAPER[a]["mae"]]
    print(f"\nbeats the base paper on: {', '.join(better) if better else 'nothing yet'}")
    if n_ok < 10:
        print("NOTE: fewer than 10 samples. Report the sample count alongside "
              "every number; the paper's own test set was only 14-15 samples, "
              "which is why its figures are so noisy.")


if __name__ == "__main__":
    main()
