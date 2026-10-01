"""Live three-panel demo. Works with the laptop camera.

Keys:  q quit    space force a fresh depth pass
"""
import time
import cv2
import numpy as np
import config
import models
import geometry


def main():
    cap = cv2.VideoCapture(config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    if not cap.isOpened():
        raise SystemExit("camera not opening - try a different CAM_INDEX in config.py")

    K, dist = geometry.load_intrinsics()
    depth, scale, n, history = None, 1.0, 0, []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if K is not None:
            frame = cv2.undistort(frame, K, dist)
        h, w = frame.shape[:2]
        f = geometry.focal_px(w, K)

        det = models.segment(frame)
        if n % config.DEPTH_EVERY == 0 or depth is None:
            depth = models.depth_map_cm(frame)
            scale, _ = geometry.anchor_scale(depth, geometry.detect_markers(frame), f)
        n += 1

        vis = frame.copy()
        text = "show an object"
        if det is not None:
            m = geometry.measure(det["mask"], depth * scale, f)
            if m is not None:
                history.append((m["long_cm"], m["short_cm"]))
                history = history[-config.STABLE_FRAMES:]
                arr = np.array(history)
                stable = len(history) == config.STABLE_FRAMES and arr.std(0).max() < config.STABLE_TOL_CM
                L, W = arr.mean(0)
                colour = (0, 255, 0) if stable else (0, 255, 255)
                vis[det["mask"]] = (0.6 * vis[det["mask"]] + 0.4 * np.array([0, 160, 120])).astype(np.uint8)
                cv2.drawContours(vis, [cv2.boxPoints(m["rect"]).astype(int)], 0, colour, 2)
                text = (f"{det['label']}  {L:.1f} x {W:.1f} cm   d={m['distance_cm']:.0f} cm"
                        + ("  LOCKED" if stable else ""))
        else:
            history.clear()

        cv2.putText(vis, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(vis, f"scale x{scale:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        dn = cv2.applyColorMap(cv2.normalize(depth * scale, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                               cv2.COLORMAP_INFERNO)
        panel = np.hstack([vis, cv2.resize(dn, (vis.shape[1], vis.shape[0]))])
        cv2.imshow("measure  |  depth", cv2.resize(panel, None, fx=0.6, fy=0.6))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            depth = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
