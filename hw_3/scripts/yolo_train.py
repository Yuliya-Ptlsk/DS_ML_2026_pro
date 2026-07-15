import subprocess
import torch
from scripts.config import PROJECT_ROOT, YOLO_DIR, DATASET_FILE, TRAIN_SCRIPT, BEST_MODEL_PATH
import sys


def train(batch_size=16, epochs=100):
    # ---------- checks ----------
    if not YOLO_DIR.exists():
        raise FileNotFoundError(
            "YOLOv5 directory not found. Run clone_yolo.py first."
        )

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(
            "train.py was not found inside yolov5."
        )

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            "Dataset was not found. Run download_dataset.py first."
        )

    device = "0" if torch.cuda.is_available() else "cpu"

    print(f"PyTorch: {torch.__version__}")
    print(f"Using device: {'GPU (CUDA)' if device == '0' else device.upper()}")

    if device == "0":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Training model...")

    process = subprocess.Popen(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--img", "640",
            "--batch", str(batch_size),
            "--epochs", str(epochs),
            "--data", str(DATASET_FILE),
            "--weights", "yolov5s.pt",
            "--device", device,
            "--project", "runs/train",
            "--name", "dog_breeds",
            "--exist-ok",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for line in process.stdout:
        print(line, end="")

    print("✅ Training completed!")

    print("Best model saved to: ../runs/train/dog_breeds/weights/best.pt")

if __name__ == "__main__":
    train()