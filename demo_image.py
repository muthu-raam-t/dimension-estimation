"""Runs the whole pipeline on ONE image file. No camera, no calibration, no dataset needed.

    python demo_image.py photo.jpg
    python demo_image.py photo.jpg --distance 60      # you measured the distance yourself
"""
import argparse
import cv2
import numpy as np
import config
import models
import geometry


def run(path, manual_distance=None, save="results/demo_out.jpg"):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"could not read {path}")
    h, w = frame.shape[:2]
    K, _ = geometry.load_intrinsics()
    f = geometry.focal_px(w, K)
    print(f"image {w}x{h}, focal {f:.1f} px ({'calibrated' if K is not None else 'assumed FOV'})")

    det = models.segment(frame)
    if det is None:
        raise SystemExit("no object detected - try a plainer background or move closer")
    print(f"object: {det['label']} (confidence {det['conf']:.2f})")

    depth = models.depth_map_cm(frame)
    markers = geometry.detect_markers(frame)
    k, spread = geometry.anchor_scale(depth, markers, f)
    print(f"markers found: {len(markers)}  scale correction: x{k:.3f}")
    depth = depth * k

    if manual_distance:
        raw = float(np.median(depth[det["mask"]]))
        depth = depth * (manual_distance / raw)
        print(f"overridden with your measured distance {manual_distance} cm")

    m = geometry.measure(det["mask"], depth, f)
    if m is None:
        raise SystemExit("mask too small to measure")
    print(f"\ndistance      : {m['distance_cm']:.1f} cm")
    print(f"longer side   : {m['long_cm']:.1f} cm")
    print(f"shorter side  : {m['short_cm']:.1f} cm")
    print("(third dimension comes from a second view - see README)")

    vis = frame.copy()
    vis[det["mask"]] = (0.6 * vis[det["mask"]] + 0.4 * np.array([0, 180, 120])).astype(np.uint8)
    cv2.drawContours(vis, [cv2.boxPoints(m["rect"]).astype(int)], 0, (0, 255, 255), 2)
    cv2.putText(vis, f"{det['label']} {m['long_cm']:.1f} x {m['short_cm']:.1f} cm",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imwrite(save, vis)
    dn = cv2.applyColorMap(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                           cv2.COLORMAP_INFERNO)
    cv2.imwrite(save.replace(".jpg", "_depth.jpg"), dn)
    print(f"\nsaved {save} and the depth map beside it")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--distance", type=float, default=None)
    a = ap.parse_args()
    run(a.image, a.distance)
