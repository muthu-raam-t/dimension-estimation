"""
Synthetic validation of sv_geometry.

Builds a virtual scene with a KNOWN camera, a KNOWN marker pose and a KNOWN
cuboid, renders it, then runs the full pipeline and checks we recover the
true dimensions. Run:  python3 -I test_sv_geometry.py
"""
import numpy as np
import cv2

from sv_geometry import (
    CameraParams, PlanePose, Cuboid, pixels_to_plane, render_silhouette,
    project_cuboid, fit_cuboid, estimate_plane_pose, check_viewing_geometry,
)

W, H = 1280, 720
cam = CameraParams.approximate(W, H, hfov_deg=60.0)


def make_pose(tilt_deg, dist_cm=70.0):
    """Camera looking at the marker origin from `tilt_deg` off vertical."""
    t = np.deg2rad(tilt_deg)
    # Camera position in marker coords.
    cam_pos = np.array([0.0, -dist_cm * np.sin(t), dist_cm * np.cos(t)])
    fwd = -cam_pos / np.linalg.norm(cam_pos)          # look at origin
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R_wc = np.vstack([right, down, fwd])              # world -> camera
    tvec = (-R_wc @ cam_pos).reshape(3, 1)
    return PlanePose(rvec=cv2.Rodrigues(R_wc)[0], tvec=tvec)


def test_backprojection():
    """A point placed on the plane must come back at the same coordinates."""
    pose = make_pose(45)
    truth = np.array([[0., 0., 0.], [10., 5., 0.], [-7., 12., 0.], [3., -9., 0.]])
    px, _ = cv2.projectPoints(truth, pose.rvec, pose.tvec, cam.K, cam.dist)
    back = pixels_to_plane(px.reshape(-1, 2), cam, pose, height=0.0)
    err = np.abs(back - truth[:, :2]).max()
    print(f"  round-trip max error: {err:.6f} cm")
    assert err < 1e-6, "back-projection is wrong"

    # Same at a raised plane.
    truth_h = truth.copy(); truth_h[:, 2] = 11.0
    px, _ = cv2.projectPoints(truth_h, pose.rvec, pose.tvec, cam.K, cam.dist)
    back = pixels_to_plane(px.reshape(-1, 2), cam, pose, height=11.0)
    err = np.abs(back - truth_h[:, :2]).max()
    print(f"  raised-plane max error: {err:.6f} cm")
    assert err < 1e-6
    print("  PASS back-projection")


def test_fit(tilt_deg, truth: Cuboid, noise_px=0.0, tag=""):
    pose = make_pose(tilt_deg)
    tilt, msg = check_viewing_geometry(pose)
    mask = render_silhouette(truth, cam, pose, (H, W))

    if noise_px > 0:
        # Roughen the mask edge to imitate an imperfect segmentation.
        k = int(noise_px)
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        mask = cv2.erode(mask, np.ones((k, k), np.uint8))

    est, iou = fit_cuboid(mask, cam, pose)
    tL, tW, tH = truth.sorted_dims()
    eL, eW, eH = est.sorted_dims()
    errs = (abs(eL - tL), abs(eW - tW), abs(eH - tH))
    print(f"  [{tag}] tilt={tilt:.0f}deg iou={iou:.3f}")
    print(f"       true  L={tL:6.2f} W={tW:6.2f} H={tH:6.2f}")
    print(f"       est   L={eL:6.2f} W={eW:6.2f} H={eH:6.2f}")
    print(f"       err   L={errs[0]:6.2f} W={errs[1]:6.2f} H={errs[2]:6.2f}  ({msg})")
    return errs, iou


def test_marker_pipeline():
    """Render a real ArUco marker in perspective and recover its pose."""
    marker_cm = 10.0
    pose_true = make_pose(45)
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img_m = cv2.aruco.generateImageMarker(adict, 0, 400)
    img_m = cv2.copyMakeBorder(img_m, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)

    # Marker spans +/- half its size, plus the white quiet zone.
    s = marker_cm / 2.0 * (500 / 400)
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)
    dst, _ = cv2.projectPoints(obj, pose_true.rvec, pose_true.tvec, cam.K, cam.dist)
    src = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], np.float32)
    Hm = cv2.getPerspectiveTransform(src, dst.reshape(-1, 2).astype(np.float32))

    scene = np.full((H, W), 200, np.uint8)
    warped = cv2.warpPerspective(img_m, Hm, (W, H), borderValue=0)
    scene = np.where(warped > 0, warped, scene).astype(np.uint8)
    scene = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)

    pose_est = estimate_plane_pose(scene, cam, marker_cm)
    assert pose_est is not None, "marker not detected"
    d = np.linalg.norm(pose_est.tvec - pose_true.tvec)
    print(f"  marker detected; translation error {d:.3f} cm")
    assert d < 1.0, "marker pose is off"
    print("  PASS marker pipeline")


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    print("\n[1] back-projection")
    test_backprojection()

    print("\n[2] marker detection -> pose")
    test_marker_pipeline()

    print("\n[3] cuboid fit, clean masks")
    box = Cuboid(cx=2.0, cy=1.0, theta=np.deg2rad(20), L=24.0, W=16.0, H=11.0)
    r45, _ = test_fit(45, box, tag="tilt 45")
    r35, _ = test_fit(35, box, tag="tilt 35")
    r60, _ = test_fit(60, box, tag="tilt 60")

    print("\n[4] cuboid fit, noisy mask (imperfect segmentation)")
    rn, _ = test_fit(45, box, noise_px=9, tag="noisy")

    print("\n[5] near-nadir (expected to be the weak case)")
    rnad, _ = test_fit(5, box, tag="tilt 5")

    print("\n[6] a flat/thin parcel")
    flat = Cuboid(cx=0, cy=0, theta=0.3, L=30.0, W=22.0, H=4.0)
    rf, _ = test_fit(45, flat, tag="flat")

    worst = max(max(r45), max(r35), max(r60))
    print(f"\nworst clean-mask error across tilts 35/45/60: {worst:.3f} cm")
    assert worst < 0.5, "clean-mask fit should be near-exact"
    print("ALL CORE ASSERTIONS PASSED")
