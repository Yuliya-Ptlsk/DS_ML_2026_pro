from pathlib import Path
import subprocess
import os
from scripts.config import PROJECT_ROOT, DATA_DIR, DATASET_FILE

required_dirs = [
    DATA_DIR / "train/images",
    DATA_DIR / "valid/images",
]

def download_dataset():
    try:
        from roboflow import Roboflow
    except ImportError:
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "roboflow",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

        from roboflow import Roboflow

        print("✅ Roboflow was installed successfully!")


    if DATASET_FILE.exists() and all(path.exists() for path in required_dirs):
        print("Dataset already exists. Skipping download.")
        return

    api_key = os.getenv("ROBOFLOW_API_KEY")

    if api_key is None:
        raise ValueError("ROBOFLOW_API_KEY environment variable is not set.")

    print("Downloading dataset from Roboflow...")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("yuliya-piatlitskaya-gmail-com").project("long-eared-dog-breeds-mpvf7")
    version = project.version(2)
    version.download(model_format="yolov5", location=str(DATA_DIR))

    print("✅ Dataset downloaded.")

if __name__ == "__main__":
    download_dataset()
