"""Loads the two pretrained models and exposes two simple functions."""
import cv2
import numpy as np
import config

_yolo = None
_depth = None


def get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        print("[models] loading", config.YOLO_MODEL)
        _yolo = YOLO(config.YOLO_MODEL)
    return _yolo


def get_depth():
    global _depth
    if _depth is None:
        import torch
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        print("[models] loading", config.DEPTH_MODEL, "on", "gpu" if dev == 0 else "cpu")
        _depth = pipeline(task="depth-estimation", model=config.DEPTH_MODEL, device=dev)
    return _depth


def segment(frame_bgr):
    """Largest non-background object. Returns dict(mask, label, conf) or None."""
    model = get_yolo()
    res = model(frame_bgr, verbose=False)[0]
    if res.masks is None or len(res.masks) == 0:
        return None
    h, w = frame_bgr.shape[:2]
    best = None
    for i in range(len(res.masks)):
        label = model.names[int(res.boxes.cls[i])]
        conf = float(res.boxes.conf[i])
        if label in config.SKIP_CLASSES or conf < config.MIN_CONF:
            continue
        m = res.masks.data[i].cpu().numpy()
        m = cv2.resize(m.astype(np.float32), (w, h)) > 0.5
        area = int(m.sum())
        if best is None or area > best["area"]:
            best = {"mask": m, "label": label, "conf": conf, "area": area}
    return best


def depth_map_cm(frame_bgr):
    """Metric depth for every pixel, in centimetres."""
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    out = get_depth()(img)
    d = out["predicted_depth"]
    d = d.detach().cpu().numpy() if hasattr(d, "detach") else np.asarray(d)
    d = np.squeeze(d).astype(np.float32)
    h, w = frame_bgr.shape[:2]
    return cv2.resize(d, (w, h)) * 100.0
