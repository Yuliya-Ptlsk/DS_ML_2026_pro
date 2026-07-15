import subprocess
from scripts.config import PROJECT_ROOT, YOLO_DIR


# clone yolo repository and install dependencies if it is absent
def clone_yolo():
    if YOLO_DIR.exists():
        print("✅ YOLOv5 is already installed.")
        return

    print("Cloning YOLOv5...")

    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/ultralytics/yolov5.git",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print("Installing dependencies...")

    result = subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "-r",
            str(YOLO_DIR / "requirements.txt"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to install YOLOv5 dependencies: \n{result.stderr}")

    print("✅ YOLOv5 installed successfully!")

if __name__ == "__main__":
    clone_yolo()