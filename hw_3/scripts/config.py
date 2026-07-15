from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DATASET_FILE = DATA_DIR / "data.yaml"
YOLO_DIR = PROJECT_ROOT / "yolov5"
TRAIN_SCRIPT = YOLO_DIR / "train.py"
DETECT_SCRIPT = YOLO_DIR / "detect.py"
BEST_MODEL_PATH = PROJECT_ROOT / "runs/train/dog_breeds/weights/best.pt"
WEIGHTS = PROJECT_ROOT / "runs/train/exp/weights/best.pt"
SOURCE = PROJECT_ROOT / "test_images"
OUTPUT = PROJECT_ROOT / "predictions"
