import subprocess
from scripts.config import PROJECT_ROOT, BEST_MODEL_PATH, SOURCE, OUTPUT, DETECT_SCRIPT
import sys


def predict():
    OUTPUT.mkdir(exist_ok=True)

    print("Running object detection on test images...")

    subprocess.run(
        [
            sys.executable,
            str(DETECT_SCRIPT),
            "--weights", str(BEST_MODEL_PATH),
            "--source", str(SOURCE),
            "--project", str(OUTPUT),
            "--name", "results",
            "--exist-ok"
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    print("✅ Detection completed!")
    print("Predictions saved to: ../predictions/results")

if __name__ == "__main__":
    predict()