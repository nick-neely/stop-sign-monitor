import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VIDEO_PATH = os.path.join(PROJECT_ROOT, "data", "videos", "traffic_test.mp4")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "output")

STOP_SPEED_THRESHOLD = 3.0

LICENSE_PLATE_MODEL_PATH = (
    "hf:morsetechlab/yolov11-license-plate-detection/license-plate-finetune-v1x.pt"
)

BASE_PIXELS_PER_METER = 30
PERSPECTIVE_FACTOR = 1.5
REFERENCE_Y_RATIO = 0.75

SPEED_INTERVALS = (5, 7, 10)
SPEED_SMOOTHING_ALPHA = 0.3
MIN_HISTORY_FRAMES = 10

HORIZONTAL_WEIGHT = 0.85
VERTICAL_WEIGHT = 0.15
