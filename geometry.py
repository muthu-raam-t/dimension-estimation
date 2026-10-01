"""Camera maths: focal length, marker anchoring, pixels to centimetres."""
import os
import cv2
import numpy as np
import config


def load_intrinsics():
    if os.path.exists(config.CALIB_FILE):
        z = np.load(config.CALIB_FILE)
        return z["K"], z["dist"]
    return None, None


def focal_px(frame_w, K=None):
    """Focal length in pixels: from calibration if available, else from assumed field of view."""
    if K is not None:
        return float((K[0, 0] + K[1, 1]) / 2.0)
    return (frame_w / 2.0) / np.tan(np.radians(config.ASSUMED_HFOV_DEG) / 2.0)


_dict = None
_params = None


def detect_markers(frame_bgr):
    """Returns list of (centre_xy, side_px)."""
    global _dict, _params
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if _dict is None:
        _dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.MARKER_DICT))
        _params = cv2.aruco.DetectorParameters()
    try:
        corners, ids, _ = cv2.aruco.ArucoDetector(_dict, _params).detectMarkers(gray)
    except AttributeError:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, _dict, parameters=_params)
    out = []
    for c in corners or []:
        p = c.reshape(4, 2)
        side = float(np.mean(np.linalg.norm(p - np.roll(p, 1, axis=0), axis=1)))
        out.append((p.mean(axis=0), side))
    return out


def marker_distance_cm(side_px, f_px, marker_cm=None):
    """Pinhole distance to a marker of known printed size."""
    marker_cm = marker_cm or config.MARKER_CM
    return marker_cm * f_px / max(side_px, 1e-6)


def anchor_scale(depth_cm, markers, f_px):
    """One scale factor that pulls the model's depth onto the marker's true depth."""
    ratios = []
    for centre, side in markers:
        x, y = int(round(centre[0])), int(round(centre[1]))
        h, w = depth_cm.shape
        if not (0 <= x < w and 0 <= y < h):
            continue
        patch = depth_cm[max(0, y - 4):y + 5, max(0, x - 4):x + 5]
        pred = float(np.median(patch))
        if pred > 1:
            ratios.append(marker_distance_cm(side, f_px) / pred)
    if not ratios:
        return 1.0, None
    k = float(np.median(ratios))
    err = float(np.std(ratios)) if len(ratios) > 1 else 0.0
    return k, err


def measure(mask, depth_cm, f_px):
    """Two visible dimensions in cm, plus distance and the rotated rectangle."""
    m8 = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    (w_px, h_px) = rect[1]
    vals = depth_cm[mask]
    vals = vals[np.isfinite(vals) & (vals > 1)]
    if vals.size < 50:
        return None
    Z = float(np.median(vals))
    long_cm = max(w_px, h_px) * Z / f_px
    short_cm = min(w_px, h_px) * Z / f_px
    return {
        "long_cm": long_cm,
        "short_cm": short_cm,
        "distance_cm": Z,
        "rect": rect,
        "long_px": max(w_px, h_px),
        "short_px": min(w_px, h_px),
        "area_px": int(mask.sum()),
    }
