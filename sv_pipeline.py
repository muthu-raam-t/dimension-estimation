"""
sv_pipeline.py -- the end-to-end measurement, in one function.

    from sv_pipeline import measure_image
    result = measure_image(bgr_frame)

Everything else (CLI, live window, web UI) is a thin wrapper around this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import sv_config as cfg
from sv_geometry import (
    CameraParams, PlanePose, Cuboid, estimate_plane_pose, fit_cuboid,
    check_viewing_geometry, draw_overlay,
)
from sv_segment import segment_object


@dataclass
class Measurement:
    ok: bool
    message: str
    cuboid: Optional[Cuboid] = None
    iou: float = 0.0
    tilt_deg: float = 0.0
    tilt_message: str = ""
    seg_method: str = ""
    mask: Optional[np.ndarray] = None
    pose: Optional[PlanePose] = None
    trusted: bool = False

    def summary(self) -> str:
        if not self.ok or self.cuboid is None:
            return f"FAILED: {self.message}"
        L, W, H = self.cuboid.sorted_dims()
        flag = "" if self.trusted else "   [LOW CONFIDENCE]"
        return (f"L = {L:6.2f} cm\n"
                f"W = {W:6.2f} cm\n"
                f"H = {H:6.2f} cm\n"
                f"volume = {self.cuboid.volume_cm3():.0f} cm3\n"
                f"volumetric weight = "
                f"{self.cuboid.volumetric_weight_kg(cfg.VOLUMETRIC_DIVISOR):.2f} kg\n"
                f"fit IoU = {self.iou:.3f}{flag}\n"
                f"camera tilt = {self.tilt_deg:.0f} deg ({self.tilt_message})")


def load_camera(image_shape) -> CameraParams:
    """Load calibrated intrinsics, or fall back to a guessed FOV with a warning."""
    h, w = image_shape[:2]
    if cfg.INTRINSICS_FILE.exists():
        d = np.load(cfg.INTRINSICS_FILE)
        K, dist = d["K"].astype(np.float64), d["dist"].astype(np.float64)
        calib_w = int(d["image_size"][0]) if "image_size" in d else w
        if calib_w != w:
            # Reusing a calibration at a different capture resolution is fine
            # as long as the aspect ratio matches -- just rescale K.
            s = w / float(calib_w)
            K = K.copy()
            K[0, 0] *= s; K[1, 1] *= s
            K[0, 2] = (K[0, 2] + 0.5) * s - 0.5
            K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
        return CameraParams(K=K, dist=dist)
    return CameraParams.approximate(w, h, cfg.FALLBACK_HFOV_DEG)


def measure_image(
    image: np.ndarray,
    cam: Optional[CameraParams] = None,
    seg_method: str = "auto",
    weights: Optional[str] = None,
    fast: bool = False,
) -> Measurement:
    if image is None or image.size == 0:
        return Measurement(False, "empty image")

    cam = cam or load_camera(image.shape)

    pose = estimate_plane_pose(image, cam, cfg.MARKER_LENGTH_CM,
                              getattr(cv2.aruco, cfg.ARUCO_DICT))
    if pose is None:
        return Measurement(False,
                           "ArUco marker not found - it must be flat on the "
                           "same surface as the parcel and fully visible")

    tilt, tilt_msg = check_viewing_geometry(pose)

    mask, used = segment_object(image, method=seg_method, weights=weights)
    if mask is None or np.count_nonzero(mask) == 0:
        return Measurement(False, "could not segment the parcel",
                           tilt_deg=tilt, tilt_message=tilt_msg, pose=pose)

    try:
        cub, iou = fit_cuboid(mask, cam, pose,
                              work_width=cfg.FIT_WORK_WIDTH,
                              max_height_cm=cfg.MAX_HEIGHT_CM,
                              fast=fast)
    except Exception as e:                                   # noqa: BLE001
        return Measurement(False, f"cuboid fit failed: {e}",
                           tilt_deg=tilt, tilt_message=tilt_msg,
                           mask=mask, pose=pose)

    trusted = iou >= cfg.MIN_TRUSTED_IOU and tilt >= 20.0
    return Measurement(True, "ok", cuboid=cub, iou=iou, tilt_deg=tilt,
                       tilt_message=tilt_msg, seg_method=used, mask=mask,
                       pose=pose, trusted=trusted)


def annotate(image: np.ndarray, r: Measurement,
             cam: Optional[CameraParams] = None) -> np.ndarray:
    """Draw the wireframe and a readout panel onto a copy of the frame."""
    cam = cam or load_camera(image.shape)
    out = image.copy()

    if r.mask is not None:
        tint = np.zeros_like(out); tint[:, :, 2] = 255
        m = (r.mask > 0)[:, :, None]
        out = np.where(m, (0.75 * out + 0.25 * tint).astype(np.uint8), out)

    if r.ok and r.cuboid is not None and r.pose is not None:
        color = (0, 255, 0) if r.trusted else (0, 180, 255)
        out = draw_overlay(out, r.cuboid, cam, r.pose, color=color, thickness=2)
        L, W, H = r.cuboid.sorted_dims()
        lines = [f"L {L:.1f} cm", f"W {W:.1f} cm", f"H {H:.1f} cm",
                 f"vol {r.cuboid.volume_cm3():.0f} cm3",
                 f"IoU {r.iou:.2f}  tilt {r.tilt_deg:.0f}d"]
        if not r.trusted:
            lines.append("LOW CONFIDENCE")
    else:
        color = (0, 0, 255)
        lines = ["NO MEASUREMENT", r.message[:38]]

    pad, lh = 12, 30
    bw = 310
    bh = pad * 2 + lh * len(lines)
    panel = out[10:10 + bh, 10:10 + bw].copy()
    cv2.rectangle(panel, (0, 0), (bw, bh), (0, 0, 0), -1)
    out[10:10 + bh, 10:10 + bw] = cv2.addWeighted(
        panel, 0.65, out[10:10 + bh, 10:10 + bw], 0.35, 0)
    for i, ln in enumerate(lines):
        cv2.putText(out, ln, (10 + pad, 10 + pad + lh * (i + 1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    return out
