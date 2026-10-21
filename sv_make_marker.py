"""
sv_make_marker.py -- generate the ArUco marker to print.

    python3 sv_make_marker.py

Writes calibration/marker_to_print.png at 300 DPI with a white quiet zone and
the required physical size printed on it.

PRINTING RULES (these matter more than the code):
  * Print at 100% / "Actual size". Do NOT use "Fit to page" -- it rescales and
    every measurement you take afterwards will be wrong by that factor.
  * Measure the printed black square edge-to-edge with a ruler and put that
    number in sv_config.MARKER_LENGTH_CM.
  * Glue or tape it flat. A curled marker tilts the estimated ground plane.
"""
import cv2
import numpy as np

import sv_config as cfg


def main() -> None:
    dict_id = getattr(cv2.aruco, cfg.ARUCO_DICT)
    adict = cv2.aruco.getPredefinedDictionary(dict_id)

    dpi = 300
    side_px = int(round(cfg.MARKER_LENGTH_CM / 2.54 * dpi))
    marker = cv2.aruco.generateImageMarker(adict, cfg.ARUCO_ID, side_px)

    quiet = int(side_px * 0.25)
    canvas = cv2.copyMakeBorder(marker, quiet, quiet + 110, quiet, quiet,
                                cv2.BORDER_CONSTANT, value=255)
    cv2.putText(canvas,
                f"{cfg.ARUCO_DICT} id={cfg.ARUCO_ID}  "
                f"PRINT AT 100%  side must measure {cfg.MARKER_LENGTH_CM} cm",
                (quiet, canvas.shape[0] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA)

    out = cfg.CALIB_DIR / "marker_to_print.png"
    cv2.imwrite(str(out), canvas)
    print(f"Wrote {out}")
    print(f"Print at 100% scale, then verify the black square measures "
          f"{cfg.MARKER_LENGTH_CM} cm and update sv_config.py if it does not.")


if __name__ == "__main__":
    main()
