"""
sv_train.py -- fine-tune YOLO11n as a parcel detector on the prepared dataset.

    python3 sv_train.py                  # sensible defaults
    python3 sv_train.py --quick          # fast sanity run (~5 min CPU)
    python3 sv_train.py --epochs 150 --imgsz 640 --device 0

NOTE ON SCOPE: the demo does NOT need this. The GrabCut fallback in
sv_segment.py already produces a usable mask, so you can run the full
measurement pipeline before any model is trained. Train this when you want the
detector table that compares against the base paper.

WHY DETECTION AND NOT SEGMENTATION: the Kaggle labels are bounding boxes, not
polygons, so a *-seg model cannot be trained on them. We train a detector and
convert its box to a mask with GrabCut. If you later hand-label even 20 images
with polygons, switching to yolo11n-seg will improve the mask and therefore the
dimensions.

The augmentation defaults below are deliberately NOT the Ultralytics defaults:
  * mosaic is reduced -- parcels are large, centred and few, so mosaic mostly
    manufactures unrealistic crops here
  * vertical flip is off and rotation is small -- deployment is a fixed camera,
    so training on upside-down parcels spends capacity on poses you will never see
  * HSV jitter is raised -- lighting is the thing that actually varies, and the
    base paper's single biggest win came from normalising brightness
"""
import argparse
from pathlib import Path

import sv_config as cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(cfg.DATA_DIR / "processed" / "box.yaml"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu' otherwise")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--name", default="box_yolo11n")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.imgsz, args.batch = 25, 416, 8

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(f"{data} not found. Run sv_prepare_dataset.py first.")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("pip install ultralytics")

    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(cfg.RESULTS_DIR / "train"),
        name=args.name,
        exist_ok=True,
        seed=0,
        deterministic=True,
        patience=30,
        # --- augmentation, tuned for a fixed-camera parcel scene ---
        mosaic=0.3,
        close_mosaic=10,
        fliplr=0.5,
        flipud=0.0,
        degrees=8.0,
        scale=0.4,
        translate=0.1,
        shear=2.0,
        perspective=0.0005,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.5,
        erasing=0.2,
    )

    metrics = model.val(data=str(data), device=args.device)
    best = cfg.RESULTS_DIR / "train" / args.name / "weights" / "best.pt"
    if best.exists():
        cfg.MODELS_DIR.mkdir(exist_ok=True, parents=True)
        import shutil
        shutil.copy2(best, cfg.YOLO_WEIGHTS)
        print(f"\nbest weights -> {cfg.YOLO_WEIGHTS}")

    b = metrics.box
    print("\n================ DETECTOR RESULTS ================")
    print(f"  precision     {b.mp:.4f}")
    print(f"  recall        {b.mr:.4f}")
    print(f"  mAP@0.5       {b.map50:.4f}")
    print(f"  mAP@0.5:0.95  {b.map:.4f}")
    print("\n  base paper (Chu et al. 2026, YOLOv11n):")
    print("    precision ~0.72  recall ~0.20  mAP@0.5 0.28  mAP@0.5:0.95 0.19")
    print("==================================================")
    print("Paste these into RESULTS.md with today's date and the exact command.")


if __name__ == "__main__":
    main()
