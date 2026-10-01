"""Camera calibration. Print a checkerboard, hold it at many angles.

Keys:  c capture a view    d done    q quit
"""
import os
import cv2
import numpy as np
import config

PAT = (config.CHECKER_COLS, config.CHECKER_ROWS)


def main():
    objp = np.zeros((PAT[0] * PAT[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PAT[0], 0:PAT[1]].T.reshape(-1, 2) * config.SQUARE_CM
    objpoints, imgpoints, shape = [], [], None

    cap = cv2.VideoCapture(config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, PAT, None)
        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, PAT, corners, found)
        cv2.putText(vis, f"captured {len(objpoints)}  [c] capture  [d] done",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("calibration", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c") and found:
            corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            shape = gray.shape[::-1]
            print("captured", len(objpoints))
        if key in (ord("d"), ord("q")):
            break
    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 8:
        raise SystemExit("need at least 8 good views")
    err, K, dist, rv, tv = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
    os.makedirs(os.path.dirname(config.CALIB_FILE), exist_ok=True)
    np.savez(config.CALIB_FILE, K=K, dist=dist)
    print(f"\nreprojection error {err:.3f} px (under 1.0 is good)")
    print(f"fx={K[0,0]:.1f}  fy={K[1,1]:.1f}")
    print("saved", config.CALIB_FILE)


if __name__ == "__main__":
    main()
