"""
sv_live.py -- live webcam measurement in an OpenCV window. Zero extra deps.

    python3 sv_live.py
    python3 sv_live.py --camera 2 --method grabcut

Keys
    SPACE   measure the current frame (full accuracy, ~1-3 s)
    C       toggle continuous mode (fast fit, runs in a background thread)
    S       save the annotated frame to results/
    Q/ESC   quit

The marker-detection overlay updates every frame, so you can position the
camera and see immediately whether the plane is being picked up and whether the
tilt is in the good range -- before you measure anything.
"""
import argparse
import threading
import time
from datetime import datetime

import cv2
import numpy as np

import sv_config as cfg
from sv_geometry import estimate_plane_pose, check_viewing_geometry
from sv_pipeline import measure_image, annotate, load_camera


class Worker:
    """Runs fits off the UI thread so the preview never freezes."""

    def __init__(self, cam, method, weights):
        self.cam, self.method, self.weights = cam, method, weights
        self.lock = threading.Lock()
        self.busy = False
        self.result = None
        self.elapsed = 0.0

    def submit(self, frame, fast):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        threading.Thread(target=self._run, args=(frame.copy(), fast),
                         daemon=True).start()
        return True

    def _run(self, frame, fast):
        t0 = time.time()
        try:
            r = measure_image(frame, cam=self.cam, seg_method=self.method,
                              weights=self.weights, fast=fast)
        except Exception as e:                                # noqa: BLE001
            from sv_pipeline import Measurement
            r = Measurement(False, f"error: {e}")
        with self.lock:
            self.result = (frame, r)
            self.elapsed = time.time() - t0
            self.busy = False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=cfg.CAMERA_INDEX)
    ap.add_argument("--method", default="auto",
                    choices=["auto", "yolo_seg", "yolo_det", "grabcut"])
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}. Try --camera 1 or 2, "
                         f"and check: ls /dev/video*")

    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Camera opened but returned no frame.")
    cam = load_camera(frame.shape)

    if not cfg.INTRINSICS_FILE.exists():
        print("WARNING: no calibration; using a guessed field of view.")
    print(__doc__)

    worker = Worker(cam, args.method, args.weights)
    continuous = False
    last_disp = None
    aruco_dict = getattr(cv2.aruco, cfg.ARUCO_DICT)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Cheap per-frame feedback: is the marker visible, is the tilt sane?
        pose = estimate_plane_pose(frame, cam, cfg.MARKER_LENGTH_CM, aruco_dict)
        live = frame.copy()
        if pose is None:
            cv2.putText(live, "MARKER NOT VISIBLE", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        else:
            tilt, msg = check_viewing_geometry(pose)
            col = (0, 255, 0) if 20 <= tilt <= 70 else (0, 165, 255)
            cv2.drawFrameAxes(live, cam.K, cam.dist, pose.rvec, pose.tvec, 5.0, 2)
            cv2.putText(live, f"marker OK  tilt {tilt:.0f}deg", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
            if tilt < 20:
                cv2.putText(live, "TILT THE CAMERA MORE (aim 30-60deg)", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        if continuous and pose is not None:
            worker.submit(frame, fast=True)

        with worker.lock:
            res = worker.result
            busy = worker.busy
            el = worker.elapsed

        if res is not None:
            shown_frame, r = res
            last_disp = annotate(shown_frame, r, cam=cam)

        disp = last_disp if (last_disp is not None and not continuous) else live
        if continuous and last_disp is not None:
            disp = last_disp

        bar = (f"[SPACE] measure   [C] continuous: {'ON' if continuous else 'off'}   "
               f"[S] save   [Q] quit   {'...fitting' if busy else f'{el*1000:.0f} ms'}")
        cv2.putText(disp, bar, (20, disp.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Single-camera parcel dimensioning", disp)

        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        elif k == ord(' '):
            if pose is None:
                print("No marker visible - cannot measure.")
            elif not worker.submit(frame, fast=False):
                print("Still fitting the previous frame...")
        elif k == ord('c'):
            continuous = not continuous
        elif k == ord('s') and last_disp is not None:
            p = cfg.RESULTS_DIR / f"live_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(str(p), last_disp)
            print(f"saved {p}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
