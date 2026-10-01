"""Captures one dataset row per keypress, with your tape measurement as ground truth.

    python collect_dataset.py --id bottle01 --true 7 7 22

Keys:  c capture one sample    q quit
"""
import argparse
import csv
import os
import cv2
import numpy as np
import config
import models
import geometry

COLS = ["object_id", "class", "capture", "true_long", "true_short",
        "raw_long", "raw_short", "distance_cm", "long_px", "short_px",
        "area_px", "yolo_conf", "scale_k", "n_markers"]


def main(oid, true_dims):
    os.makedirs(os.path.dirname(config.DATASET_CSV), exist_ok=True)
    new = not os.path.exists(config.DATASET_CSV)
    fh = open(config.DATASET_CSV, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(COLS)

    t_long, t_short = max(true_dims), min(true_dims)
    K, dist = geometry.load_intrinsics()
    cap = cv2.VideoCapture(config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if K is not None:
            frame = cv2.undistort(frame, K, dist)
        vis = frame.copy()
        cv2.putText(vis, f"{oid}  true {t_long} x {t_short} cm   captured {count}   [c] capture",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("collect", cv2.resize(vis, None, fx=0.6, fy=0.6))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            f = geometry.focal_px(frame.shape[1], K)
            det = models.segment(frame)
            if det is None:
                print("no object detected, try again")
                continue
            depth = models.depth_map_cm(frame)
            markers = geometry.detect_markers(frame)
            k, _ = geometry.anchor_scale(depth, markers, f)
            m = geometry.measure(det["mask"], depth * k, f)
            if m is None:
                print("mask too small, try again")
                continue
            count += 1
            wr.writerow([oid, det["label"], count, t_long, t_short,
                         round(m["long_cm"], 2), round(m["short_cm"], 2),
                         round(m["distance_cm"], 1), round(m["long_px"], 1),
                         round(m["short_px"], 1), m["area_px"],
                         round(det["conf"], 3), round(k, 3), len(markers)])
            fh.flush()
            print(f"row {count}: raw {m['long_cm']:.1f} x {m['short_cm']:.1f} cm "
                  f"vs true {t_long} x {t_short} cm  (d={m['distance_cm']:.0f} cm)")
            cv2.imwrite(f"data/{oid}_{count}.jpg", frame)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    fh.close()
    print("appended to", config.DATASET_CSV)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="object id, e.g. bottle01")
    ap.add_argument("--true", nargs=2, type=float, required=True,
                    metavar=("DIM_A", "DIM_B"), help="the two dimensions facing the camera, in cm")
    a = ap.parse_args()
    main(a.id, a.true)
