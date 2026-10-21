"""
sv_geometry.py -- Single-view (one webcam) 3-D cuboid dimension estimation.

Core idea
---------
A printed ArUco marker lying on the same flat surface as the parcel gives us
BOTH the camera pose relative to that surface AND a metric scale. With the
ground plane known in metric units, a cuboid standing on it has exactly 6
unknowns:

    cx, cy    footprint centre on the plane (cm)
    theta     yaw of the footprint (rad)
    L, W      footprint side lengths (cm)
    H         height above the plane (cm)

We recover all 6 by *silhouette fitting*: render the 8 corners of a candidate
cuboid, take their convex hull, and compare that predicted silhouette with the
observed object mask (IoU). The best-IoU cuboid is the answer.

This replaces the base paper's stereo disparity entirely -- no second camera.

IMPORTANT geometric caveat (read this):
    If the camera looks straight DOWN (nadir), the mapping from the top face to
    the ground plane is a pure scaling and height becomes weakly observable --
    many (H, L, W) combinations explain the same silhouette. Tilt the camera
    30-60 degrees off vertical so at least one vertical side face is visible.
    `check_viewing_geometry()` below warns you when the pose is degenerate.

Author: Muthu Raam
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import minimize, differential_evolution


# --------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------

@dataclass
class CameraParams:
    """Pinhole intrinsics."""
    K: np.ndarray            # 3x3
    dist: np.ndarray         # 1xN distortion coefficients

    @staticmethod
    def approximate(image_w: int, image_h: int, hfov_deg: float = 60.0) -> "CameraParams":
        """
        Fallback intrinsics from a guessed horizontal field of view.

        Use ONLY to get a demo running. A real chessboard calibration
        (sv_calibrate_intrinsics.py) typically halves the dimension error.
        """
        f = (image_w / 2.0) / np.tan(np.deg2rad(hfov_deg) / 2.0)
        K = np.array([[f, 0, image_w / 2.0],
                      [0, f, image_h / 2.0],
                      [0, 0, 1.0]], dtype=np.float64)
        return CameraParams(K=K, dist=np.zeros((5, 1), dtype=np.float64))


@dataclass
class PlanePose:
    """Pose of the ground plane (the marker's plane) in camera coordinates."""
    rvec: np.ndarray         # 3x1 Rodrigues
    tvec: np.ndarray         # 3x1 translation, same unit as marker size (cm)

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]


@dataclass
class Cuboid:
    """A box standing on the plane, expressed in marker/plane coordinates (cm)."""
    cx: float
    cy: float
    theta: float
    L: float
    W: float
    H: float

    def as_dict(self) -> dict:
        return asdict(self)

    def sorted_dims(self) -> Tuple[float, float, float]:
        """Return (length, width, height) with length >= width, as the paper reports."""
        a, b = max(self.L, self.W), min(self.L, self.W)
        return a, b, self.H

    def volume_cm3(self) -> float:
        return self.L * self.W * self.H

    def volumetric_weight_kg(self, divisor: float = 5000.0) -> float:
        """Courier volumetric weight. divisor=5000 for cm/kg (standard air freight)."""
        return self.volume_cm3() / divisor


# --------------------------------------------------------------------------
# Marker detection -> plane pose
# --------------------------------------------------------------------------

def _aruco_detector(dict_id: int = cv2.aruco.DICT_4X4_50):
    """Build an ArUco detector that works on both OpenCV 4.7+ and 5.x."""
    adict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    # Sub-pixel corner refinement matters a lot here: a 1 px corner error at
    # 1 m distance is roughly 1 mm of scale error.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(adict, params)


def estimate_plane_pose(
    image: np.ndarray,
    cam: CameraParams,
    marker_length_cm: float,
    dict_id: int = cv2.aruco.DICT_4X4_50,
) -> Optional[PlanePose]:
    """
    Detect the ArUco marker and return the plane pose, or None if not found.

    The marker defines the world frame: its centre is the origin, its plane is
    Z=0, and Z grows upward (away from the table).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, _ = _aruco_detector(dict_id).detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None

    # If several markers are visible, use the largest (closest / most reliable).
    areas = [cv2.contourArea(c.reshape(-1, 2).astype(np.float32)) for c in corners]
    c = corners[int(np.argmax(areas))].reshape(-1, 2).astype(np.float64)

    h = marker_length_cm / 2.0
    # ArUco corner order is TL, TR, BR, BL looking at the marker.
    obj = np.array([[-h,  h, 0.0],
                    [ h,  h, 0.0],
                    [ h, -h, 0.0],
                    [-h, -h, 0.0]], dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        obj, c, cam.K, cam.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok:
        return None
    return PlanePose(rvec=rvec.reshape(3, 1), tvec=tvec.reshape(3, 1))


def check_viewing_geometry(pose: PlanePose) -> Tuple[float, str]:
    """
    Return (tilt_deg, message). tilt_deg is the angle between the camera's
    optical axis and the plane normal.

    ~0 deg  = straight down (nadir)  -> height is poorly observable, BAD
    30-60   = good
    >75     = very oblique, the footprint becomes poorly observable
    """
    n_cam = pose.R[:, 2]                     # plane normal in camera coords
    axis = np.array([0.0, 0.0, 1.0])         # camera optical axis
    cos = abs(float(np.dot(n_cam, axis)) / (np.linalg.norm(n_cam) + 1e-12))
    tilt = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))

    if tilt < 15:
        msg = (f"DEGENERATE ({tilt:.0f}deg from vertical): camera is nearly straight "
               f"down. Height will be unreliable. Tilt the webcam to 30-60deg.")
    elif tilt < 25:
        msg = f"MARGINAL ({tilt:.0f}deg). Tilt a bit more for a stable height."
    elif tilt <= 70:
        msg = f"GOOD ({tilt:.0f}deg from vertical)."
    else:
        msg = f"VERY OBLIQUE ({tilt:.0f}deg). Footprint may be distorted; reduce tilt."
    return tilt, msg


# --------------------------------------------------------------------------
# Ray / plane intersection
# --------------------------------------------------------------------------

def pixels_to_plane(
    pts_px: np.ndarray,
    cam: CameraParams,
    pose: PlanePose,
    height: float = 0.0,
) -> np.ndarray:
    """
    Back-project pixels onto the plane Z = `height` (in marker coordinates).

    Parameters
    ----------
    pts_px : (N,2) pixel coordinates
    height : plane offset in cm above the marker plane

    Returns
    -------
    (N,2) array of (X, Y) in marker coordinates, cm.
    """
    pts = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)

    # Undistort to normalised camera coordinates, then form rays.
    norm = cv2.undistortPoints(pts, cam.K, cam.dist).reshape(-1, 2)
    rays = np.hstack([norm, np.ones((norm.shape[0], 1))])      # (N,3) directions

    R, t = pose.R, pose.tvec.reshape(3)
    n = R[:, 2]                                                 # plane normal
    p0 = R @ np.array([0.0, 0.0, height]) + t                   # point on plane

    denom = rays @ n
    denom = np.where(np.abs(denom) < 1e-9, 1e-9, denom)
    s = (p0 @ n) / denom                                        # ray parameter
    P_cam = rays * s[:, None]                                   # (N,3) in camera frame

    P_world = (R.T @ (P_cam - t).T).T                           # back to marker frame
    return P_world[:, :2]


def cuboid_corners(c: Cuboid) -> np.ndarray:
    """Return the 8 corners of the cuboid in marker coordinates, (8,3)."""
    ct, st = np.cos(c.theta), np.sin(c.theta)
    hx, hy = c.L / 2.0, c.W / 2.0
    base = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    rot = np.array([[ct, -st], [st, ct]])
    base = base @ rot.T + np.array([c.cx, c.cy])

    bottom = np.hstack([base, np.zeros((4, 1))])
    top = np.hstack([base, np.full((4, 1), c.H)])
    return np.vstack([bottom, top])


def project_cuboid(c: Cuboid, cam: CameraParams, pose: PlanePose) -> np.ndarray:
    """Project the cuboid's 8 corners to pixels, (8,2)."""
    pts, _ = cv2.projectPoints(
        cuboid_corners(c).astype(np.float64), pose.rvec, pose.tvec, cam.K, cam.dist
    )
    return pts.reshape(-1, 2)


def render_silhouette(
    c: Cuboid, cam: CameraParams, pose: PlanePose, shape: Tuple[int, int]
) -> np.ndarray:
    """Render the cuboid's silhouette (convex hull of the 8 projected corners)."""
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    px = project_cuboid(c, cam, pose)
    if not np.all(np.isfinite(px)):
        return canvas
    # Guard against wild projections blowing up fillConvexPoly.
    px = np.clip(px, -1e4, 1e4)
    hull = cv2.convexHull(px.astype(np.float32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(canvas, hull.astype(np.int32), 255)
    return canvas


# --------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b)) / float(union)


def initial_guess(
    mask: np.ndarray, cam: CameraParams, pose: PlanePose
) -> Cuboid:
    """
    Seed the optimiser by projecting the mask outline onto the ground plane.

    This over-estimates the footprint (the top face casts outward), so we shrink
    it and start the height at a plausible fraction. The optimiser fixes both.
    """
    cnts, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not cnts:
        raise ValueError("initial_guess: the mask is empty.")
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)

    ground = pixels_to_plane(cnt, cam, pose, height=0.0)
    ground = ground[np.all(np.isfinite(ground), axis=1)]
    if len(ground) < 3:
        raise ValueError("initial_guess: back-projection failed (check the pose).")

    rect = cv2.minAreaRect(ground.astype(np.float32))
    (cx, cy), (w, h), ang = rect
    L, W = max(w, h), min(w, h)
    theta = np.deg2rad(ang if w >= h else ang + 90.0)

    # NOTE: L and W here are the RAW ground-plane footprint of the whole
    # silhouette. For a tall box this over-estimates the true footprint (the top
    # face casts outward past the base). For a flat box it can slightly
    # UNDER-estimate. Callers must therefore bracket it on both sides rather
    # than assuming it is an upper bound -- getting this wrong silently excludes
    # the true solution from the search.
    return Cuboid(cx=float(cx), cy=float(cy), theta=float(theta),
                  L=float(L), W=float(W), H=float(W) * 0.40)


def _scaled_camera(cam: CameraParams, scale: float) -> CameraParams:
    """
    Rescale intrinsics for a resized image.

    cv2.resize maps pixel centres as x' = (x + 0.5) * s - 0.5, so the principal
    point needs that same half-pixel correction. Skipping it costs about half a
    pixel, which is small but free to fix.
    """
    K = cam.K.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] = (K[0, 2] + 0.5) * scale - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * scale - 0.5
    K[0, 1] *= scale
    return CameraParams(K=K, dist=cam.dist)


def fit_cuboid(
    mask: np.ndarray,
    cam: CameraParams,
    pose: PlanePose,
    work_width: int = 480,
    max_height_cm: float = 60.0,
    seed: int = 0,
    fast: bool = False,
) -> Tuple[Cuboid, float]:
    """
    Fit a cuboid to an object mask by maximising silhouette IoU.

    Strategy
    --------
    The objective is well peaked at the true parameters but riddled with local
    maxima -- a tall narrow box and a flat wide box cast confusingly similar
    silhouettes, and the plane-projection seed over-estimates the footprint by
    as much as 50%. A local optimiser started from that seed lands in the wrong
    basin. So we run a bounded GLOBAL search (differential evolution) and then
    polish the winner with Nelder-Mead.

    Bounds come from the seed: back-projecting the silhouette onto the ground
    plane always OVER-estimates the footprint, so the true L and W lie below the
    seed values -- a genuinely useful one-sided constraint.

    Set `fast=True` for live video (roughly 4x quicker, slightly noisier).

    Returns (cuboid, iou). An IoU below ~0.85 means the fit is not trustworthy --
    usually the mask is wrong, or the camera is too close to nadir.
    """
    mask_bin = (mask > 0).astype(np.uint8) * 255
    H_img, W_img = mask_bin.shape[:2]

    scale = min(1.0, work_width / float(W_img))
    small_shape = (max(8, int(round(H_img * scale))), max(8, int(round(W_img * scale))))
    # INTER_AREA + threshold antialiases the target, which makes the IoU surface
    # smoother than nearest-neighbour downsampling does.
    target = cv2.resize(mask_bin, (small_shape[1], small_shape[0]),
                        interpolation=cv2.INTER_AREA) > 127
    cam_s = _scaled_camera(cam, scale)

    init = initial_guess(mask_bin, cam, pose)
    tgt_count = int(np.count_nonzero(target))
    if tgt_count == 0:
        raise ValueError("fit_cuboid: the mask is empty after resizing.")

    def iou_of(c: Cuboid) -> float:
        pred = render_silhouette(c, cam_s, pose, small_shape) > 0
        inter = int(np.count_nonzero(pred & target))
        if inter == 0:
            return 0.0
        union = int(np.count_nonzero(pred)) + tgt_count - inter
        return inter / float(union)

    def cost(x: np.ndarray) -> float:
        return 1.0 - iou_of(Cuboid(cx=x[0], cy=x[1], theta=x[2],
                                   L=abs(x[3]), W=abs(x[4]), H=abs(x[5])))

    span = max(init.L, init.W, 1.0)
    hi = max(max_height_cm, 2.0)
    bounds = [
        (init.cx - 0.6 * span, init.cx + 0.6 * span),   # cx
        (init.cy - 0.6 * span, init.cy + 0.6 * span),   # cy
        (-np.pi / 2, np.pi / 2),                        # theta
        (0.12 * init.L, 1.35 * init.L),                 # L
        (0.12 * init.W, 1.35 * init.W),                 # W
        (0.3, hi),                                      # H
    ]

    de = differential_evolution(
        cost, bounds,
        strategy="best1bin",
        maxiter=40 if fast else 90,
        popsize=12 if fast else 20,
        tol=1e-6, mutation=(0.4, 1.0), recombination=0.85,
        seed=seed, polish=False, init="sobol",
    )

    res = minimize(cost, de.x, method="Nelder-Mead",
                   options=dict(maxiter=4000, xatol=1e-4, fatol=1e-8))
    best_x = res.x if res.fun < de.fun else de.x

    # ---- high-resolution polish -------------------------------------------
    # Downsampling blurs the silhouette, which flattens the objective; thin or
    # flat parcels suffer most because their whole signal is a few pixels of
    # side face. Re-polishing at full resolution sharpens those cases.
    if not fast:
        hi_scale = min(1.0, (work_width * 2.0) / float(W_img))
        hi_shape = (max(8, int(round(H_img * hi_scale))),
                    max(8, int(round(W_img * hi_scale))))
        hi_target = cv2.resize(mask_bin, (hi_shape[1], hi_shape[0]),
                               interpolation=cv2.INTER_AREA) > 127
        hi_cam = _scaled_camera(cam, hi_scale)
        hi_count = int(np.count_nonzero(hi_target))

        def cost_hi(x: np.ndarray) -> float:
            c = Cuboid(cx=x[0], cy=x[1], theta=x[2],
                       L=abs(x[3]), W=abs(x[4]), H=abs(x[5]))
            pred = render_silhouette(c, hi_cam, pose, hi_shape) > 0
            inter = int(np.count_nonzero(pred & hi_target))
            if inter == 0:
                return 1.0
            union = int(np.count_nonzero(pred)) + hi_count - inter
            return 1.0 - inter / float(union)

        res_hi = minimize(cost_hi, best_x, method="Nelder-Mead",
                          options=dict(maxiter=4000, xatol=1e-4, fatol=1e-9))
        if res_hi.fun < cost_hi(best_x):
            best_x = res_hi.x
        final_iou = 1.0 - float(cost_hi(best_x))
    else:
        final_iou = 1.0 - float(cost(best_x))

    c = Cuboid(cx=float(best_x[0]), cy=float(best_x[1]),
               theta=float(best_x[2]), L=abs(float(best_x[3])),
               W=abs(float(best_x[4])), H=abs(float(best_x[5])))
    c.theta = float(np.arctan2(np.sin(c.theta), np.cos(c.theta)))
    return c, final_iou


def draw_overlay(
    image: np.ndarray, c: Cuboid, cam: CameraParams, pose: PlanePose,
    color=(0, 255, 0), thickness: int = 2
) -> np.ndarray:
    """Draw the fitted wireframe cuboid onto a copy of the image."""
    out = image.copy()
    px = project_cuboid(c, cam, pose)
    if not np.all(np.isfinite(px)):
        return out
    p = np.clip(px, -1e4, 1e4).astype(int)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),          # bottom
             (4, 5), (5, 6), (6, 7), (7, 4),          # top
             (0, 4), (1, 5), (2, 6), (3, 7)]          # verticals
    for i, j in edges:
        cv2.line(out, tuple(p[i]), tuple(p[j]), color, thickness, cv2.LINE_AA)
    cv2.drawFrameAxes(out, cam.K, cam.dist, pose.rvec, pose.tvec, 5.0, 2)
    return out
