"""Central settings. Edit these, not the other files."""

CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

YOLO_MODEL = "yolo11n-seg.pt"
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"

SKIP_CLASSES = {"person", "dining table", "tv", "chair", "couch", "bed"}
MIN_CONF = 0.30

MARKER_DICT = "DICT_4X4_50"
MARKER_CM = 10.0

ASSUMED_HFOV_DEG = 60.0
CALIB_FILE = "calibration/intrinsics.npz"

CHECKER_COLS, CHECKER_ROWS = 9, 6
SQUARE_CM = 2.5

DATASET_CSV = "data/dataset.csv"
CORRECTION_MODEL = "models/correction.pkl"

STABLE_FRAMES = 15
STABLE_TOL_CM = 0.6
DEPTH_EVERY = 5
