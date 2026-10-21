"""
sv_calibrate_intrinsics.py -- one-off camera calibration with a chessboard.

Two modes:

  Capture frames from the webcam (press SPACE to grab, Q to finish):
      python3 sv_calibrate_intrinsics.py --capture

  Calibrate from already-captured images:
      python3 sv_calibrate_intrinsics.py --images "calibration/shots/*.jpg"

Writes calibration/intrinsics.npz.

Print a 9x6 chessboard (inner corners), stick it on something rigid and flat,
and take 15-25 shots covering the frame: centre, all four corners, and several
strong tilts. Tilt variety is what actually pins down the focal length.

SKIPPING THIS: the pipeline falls back to a guessed field of view. It will run,
but expect roughly double the dimension error. 10 minutes here is the single
highest-value thing you can do for accuracy.
"""
import argparse
import glob

import cv2
import numpy as np

import sv_config as cfg


def object_points() -> np.ndarray:
    objp = np.zeros((cfg.CHESSBOARD_ROWS * cfg.CHESSBOARD_COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cfg.CHESSBOARD_COLS,
                           0:cfg.CHESSBOARD_ROWS].T.reshape(-1, 2)
    return objp * cfg.CHESSBOARD_SQUARE_CM


def find_corners(gray: np.ndarray):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(
        gray, (cfg.CHESSBOARD_COLS, cfg.CHESSBOARD_ROWS), flags)
    if not ok:
        return False, None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
    return True, corners


def capture() -> list:
    cap = cv2.VideoCapture(cfg.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {cfg.CAMERA_INDEX}. "
                         f"Try a different CAMERA_INDEX in sv_config.py.")

    shots_dir = cfg.CALIB_DIR / "shots"
    shots_dir.mkdir(exist_ok=True)
    grabbed, n = [], 0
    print("SPACE = capture when the board is detected,  Q = done")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(
                disp, (cfg.CHESSBOARD_COLS, cfg.CHESSBOARD_ROWS), corners, found)
        cv2.putText(disp, f"captured: {n}  {'BOARD OK' if found else 'no board'}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibration", disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' ') and found:
            n += 1
            cv2.imwrite(str(shots_dir / f"calib_{n:03d}.jpg"), frame)
            grabbed.append((gray.shape[::-1], corners))
            print(f"  captured {n}")
        elif k in (ord('q'), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    return grabbed


def from_images(pattern: str) -> list:
    out = []
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No images matched {pattern!r}")
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray)
        print(f"  {'OK ' if found else 'miss'} {f}")
        if found:
            out.append((gray.shape[::-1], corners))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--images", type=str, default=None)
    args = ap.parse_args()

    if args.capture:
        grabbed = capture()
    elif args.images:
        grabbed = from_images(args.images)
    else:
        raise SystemExit("Pass --capture or --images <glob>")

    if len(grabbed) < 8:
        raise SystemExit(f"Only {len(grabbed)} usable views. Need at least 8, "
                         f"ideally 15-25 with varied tilt.")

    size = grabbed[0][0]
    objp = object_points()
    obj_pts = [objp for _ in grabbed]
    img_pts = [c for _, c in grabbed]

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, size, None, None)

    # Reprojection error is the honest quality number; print it so you know
    # whether to trust the result.
    total, count = 0.0, 0
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        total += cv2.norm(img_pts[i], proj, cv2.NORM_L2) ** 2
        count += len(proj)
    reproj = float(np.sqrt(total / count))

    np.savez(cfg.INTRINSICS_FILE, K=K, dist=dist,
             image_size=np.array(size), reproj_error=reproj)
    print(f"\nviews used      : {len(grabbed)}")
    print(f"RMS             : {rms:.4f}")
    print(f"reproj error    : {reproj:.4f} px   "
          f"({'good' if reproj < 0.5 else 'high - retake with more tilt variety'})")
    print(f"fx, fy          : {K[0,0]:.1f}, {K[1,1]:.1f}")
    print(f"saved           : {cfg.INTRINSICS_FILE}")


if __name__ == "__main__":
    main()
