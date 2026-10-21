"""
sv_measure.py -- measure a parcel in a still image.

    python3 sv_measure.py --image path/to/photo.jpg
    python3 sv_measure.py --image photo.jpg --save results/out.jpg
    python3 sv_measure.py --image photo.jpg --method grabcut
"""
import argparse
import json

import cv2

import sv_config as cfg
from sv_pipeline import measure_image, annotate, load_camera


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--method", default="auto",
                    choices=["auto", "yolo_seg", "yolo_det", "grabcut"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--save", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")

    if not cfg.INTRINSICS_FILE.exists():
        print("WARNING: no calibration found; using a guessed field of view. "
              "Run sv_calibrate_intrinsics.py for roughly 2x better accuracy.\n")

    cam = load_camera(img.shape)
    r = measure_image(img, cam=cam, seg_method=args.method, weights=args.weights)
    print(r.summary())
    if r.ok:
        print(f"segmentation: {r.seg_method}")

    out = annotate(img, r, cam=cam)
    save = args.save or str(cfg.RESULTS_DIR / "measured.jpg")
    cv2.imwrite(save, out)
    print(f"\nannotated image -> {save}")

    if args.json and r.ok and r.cuboid is not None:
        L, W, H = r.cuboid.sorted_dims()
        with open(args.json, "w") as f:
            json.dump({"length_cm": L, "width_cm": W, "height_cm": H,
                       "volume_cm3": r.cuboid.volume_cm3(),
                       "iou": r.iou, "tilt_deg": r.tilt_deg,
                       "trusted": r.trusted}, f, indent=2)
        print(f"json -> {args.json}")


if __name__ == "__main__":
    main()
