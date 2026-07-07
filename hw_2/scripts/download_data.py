from pathlib import Path
from zipfile import ZipFile
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "mateuszbuda/lgg-mri-segmentation"

DATA_DIR = Path("../data")
DATASET_DIR = DATA_DIR / "kaggle_3m"
ZIP_FILE = DATA_DIR / "lgg-mri-segmentation.zip"

if DATASET_DIR.exists():
    print("Dataset already exists")
    raise SystemExit

DATA_DIR.mkdir(exist_ok=True)

api = KaggleApi()
api.authenticate()

api.dataset_download_files(
    DATASET,
    path=DATA_DIR,
    unzip=False,
)

with ZipFile(ZIP_FILE) as zip_ref:
    for member in zip_ref.infolist():
        if member.filename.startswith("kaggle_3m"):
            zip_ref.extract(member, DATA_DIR)


ZIP_FILE.unlink()

print("Dataset downloaded successfully.")
