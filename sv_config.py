"""
sv_config.py -- every physical / tunable constant in one place.

MEASURE THE MARKER. Everything downstream is scaled by MARKER_LENGTH_CM, so a
5% error here is a 5% error in every dimension you report.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
CALIB_DIR = ROOT / "calibration"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
for _d in (CALIB_DIR, RESULTS_DIR, MODELS_DIR, DATA_DIR):
    _d.mkdir(exist_ok=True, parents=True)

INTRINSICS_FILE = CALIB_DIR / "intrinsics.npz"

# ---------------------------------------------------------------- marker
# <<< MEASURE THIS WITH A RULER, EDGE TO EDGE OF THE BLACK SQUARE >>>
MARKER_LENGTH_CM = 10.0
ARUCO_DICT = "DICT_4X4_50"
ARUCO_ID = 0

# ---------------------------------------------------------------- camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# Only used when no calibration file exists. Most laptop/USB webcams are 55-70.
FALLBACK_HFOV_DEG = 60.0

# ---------------------------------------------------------------- chessboard
CHESSBOARD_COLS = 9          # INNER corners across
CHESSBOARD_ROWS = 6          # INNER corners down
CHESSBOARD_SQUARE_CM = 2.5

# ---------------------------------------------------------------- fitting
MAX_HEIGHT_CM = 60.0
FIT_WORK_WIDTH = 480
MIN_TRUSTED_IOU = 0.85

# ---------------------------------------------------------------- detector
YOLO_WEIGHTS = MODELS_DIR / "box_detector.pt"   # produced by sv_train.py
YOLO_SEG_FALLBACK = ROOT / "yolo11n-seg.pt"     # COCO-pretrained, already present
YOLO_CONF = 0.25

# ---------------------------------------------------------------- output
VOLUMETRIC_DIVISOR = 5000.0   # cm^3 per kg, standard air-freight convention
