"""Generates the ArUco marker PNG to print. Needs no camera."""
import cv2
import numpy as np
import config

DPI = 300
px = int(round(config.MARKER_CM / 2.54 * DPI))
d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.MARKER_DICT))
img = cv2.aruco.generateImageMarker(d, 0, px)
pad = px // 5
sheet = np.full((px + 2 * pad, px + 2 * pad), 255, np.uint8)
sheet[pad:pad + px, pad:pad + px] = img
cv2.imwrite("marker.png", sheet)
print(f"marker.png written. Print at 100% scale, then check the black square measures "
      f"{config.MARKER_CM} cm. If it does not, set MARKER_CM in config.py to what you measured.")
