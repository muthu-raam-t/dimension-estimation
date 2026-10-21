"""
sv_segment.py -- get a binary mask of the parcel.

Three strategies, tried in this order by `segment_object(..., method="auto")`:

  1. "yolo_seg"   a YOLO *-seg model gives a polygon mask directly. Best quality.
  2. "yolo_det"   a YOLO detector gives a box; GrabCut turns it into a mask.
                  This is the path for a model trained on the Kaggle Box set,
                  whose labels are bounding boxes, not polygons -- so you cannot
                  train a -seg model on it directly.
  3. "grabcut"    no model at all: GrabCut seeded from a central rectangle.
                  Always available, so the demo never hard-fails.

Whatever the source, the ArUco marker is erased from the mask -- it lies on the
same plane and would otherwise be swallowed into the parcel's silhouette.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

import sv_config as cfg

_MODEL_CACHE: dict = {}


def _load_yolo(weights: str):
    if weights in _MODEL_CACHE:
        return _MODEL_CACHE[weights]
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    try:
        m = YOLO(weights)
    except Exception as e:                                  # noqa: BLE001
        print(f"[segment] could not load {weights}: {e}")
        return None
    _MODEL_CACHE[weights] = m
    return m


def marker_mask(image: np.ndarray, pad: int = 14) -> np.ndarray:
    """Binary mask covering the ArUco marker(s), so we can exclude them."""
    h, w = image.shape[:2]
    out = np.zeros((h, w), np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.ARUCO_DICT))
    det = cv2.aruco.ArucoDetector(adict, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return out
    for c in corners:
        pts = c.reshape(-1, 2).astype(np.int32)
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 255)
    if pad > 0:
        out = cv2.dilate(out, np.ones((pad, pad), np.uint8))
    return out


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest blob; drops GrabCut speckle."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((lab == idx).astype(np.uint8)) * 255


def _clean(mask: np.ndarray, exclude: Optional[np.ndarray]) -> np.ndarray:
    if exclude is not None:
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return _largest_component(mask)


def _grabcut(image: np.ndarray, rect: Tuple[int, int, int, int],
             iters: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, rw, rh = rect
    x = max(1, min(x, w - 3)); y = max(1, min(y, h - 3))
    rw = max(3, min(rw, w - x - 1)); rh = max(3, min(rh, h - y - 1))
    gc = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, gc, (x, y, rw, rh), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        out = np.zeros((h, w), np.uint8)
        out[y:y + rh, x:x + rw] = 255
        return out
    return np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def segment_object(
    image: np.ndarray,
    method: str = "auto",
    weights: Optional[str] = None,
    exclude_marker: bool = True,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Return (mask, method_used). mask is None when nothing was found.
    """
    excl = marker_mask(image) if exclude_marker else None
    h, w = image.shape[:2]
    order = ([method] if method != "auto"
             else ["yolo_seg", "yolo_det", "grabcut"])

    for m in order:
        if m == "yolo_seg":
            wt = weights or str(cfg.YOLO_SEG_FALLBACK)
            model = _load_yolo(wt)
            if model is None:
                continue
            r = model.predict(image, conf=cfg.YOLO_CONF, verbose=False)[0]
            if r.masks is None or len(r.masks) == 0:
                continue
            best, best_area = None, 0
            for poly in r.masks.xy:
                cand = np.zeros((h, w), np.uint8)
                cv2.fillPoly(cand, [poly.astype(np.int32)], 255)
                cand = _clean(cand, excl)
                a = int(np.count_nonzero(cand))
                if a > best_area:
                    best, best_area = cand, a
            if best is not None and best_area > 0.002 * h * w:
                return best, "yolo_seg"

        elif m == "yolo_det":
            wt = weights or (str(cfg.YOLO_WEIGHTS) if cfg.YOLO_WEIGHTS.exists()
                             else str(cfg.YOLO_SEG_FALLBACK))
            model = _load_yolo(wt)
            if model is None:
                continue
            r = model.predict(image, conf=cfg.YOLO_CONF, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            x1, y1, x2, y2 = xyxy[int(np.argmax(areas))].astype(int)
            pad = 6
            rect = (x1 - pad, y1 - pad, (x2 - x1) + 2 * pad, (y2 - y1) + 2 * pad)
            mask = _clean(_grabcut(image, rect), excl)
            if np.count_nonzero(mask) > 0.002 * h * w:
                return mask, "yolo_det+grabcut"

        elif m == "grabcut":
            # Seed with the central 60% of the frame, minus the marker.
            rect = (int(w * 0.20), int(h * 0.20), int(w * 0.60), int(h * 0.60))
            mask = _clean(_grabcut(image, rect), excl)
            if np.count_nonzero(mask) > 0.002 * h * w:
                return mask, "grabcut"

    return None, "none"
