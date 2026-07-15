# Long-Eared Dog Breed Detection with YOLOv5

## Project Overview

This project demonstrates how to train a custom **YOLOv5** object detection model to recognize different dog breeds.

The model was trained on a custom dataset created in **Roboflow** and is capable of detecting the following breeds:

- Basset Hound
- Beagle
- Cocker Spaniel
- Vizsla
- Weimaraner

The project includes automatic setup of YOLOv5, dataset downloading, model training, and prediction on new images.

---

## Dataset

The dataset was created in **Roboflow**.

Dataset split:

- **Train:** 80%
- **Validation:** 20%

The dataset is automatically downloaded in YOLOv5 format.

---

## Project Structure

```
hw_3/
│
├── data/
│   └── train/
│       ├── images/
│       └── labels/
│   └── valid/
│       ├── images/
│       └── labels/
│   └── data.yaml
├── notebooks/
│   └── dog_breed_detection.ipynb
├── predictions/
│       └── runs/
├── scripts/
│   ├── clone_yolo.py
│   ├── download_dataset.py
│   ├── yolo_train.py
│   ├── yolo_predict.py
│   └── config.py
├── test_images/
└── yolov5/
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/Yuliya-Ptlsk/DS_ML_2026_pro.git
cd DS_ML_2026_pro/hw_3
```

Create a Python 3.10 virtual environment using **uv**.

```bash
uv python install 3.10
uv venv --python 3.10
```

Activate the virtual environment.

**Windows**

```bash
.venv\Scripts\activate
```

Install PyTorch with CUDA support.

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

The project automatically:

- clones the YOLOv5 repository;
- installs YOLOv5 dependencies;
- downloads the dataset from Roboflow.

---

## Workflow

The notebook executes the entire pipeline:

```python
clone_yolo()
download_dataset()
train()
predict()
```

Pipeline steps:

1. Clone YOLOv5 repository (only once)
2. Install YOLOv5 dependencies
3. Download dataset from Roboflow (only once)
4. Train the model
5. Run inference on new images

---

## Scripts

### clone_yolo.py

- clones the official YOLOv5 repository;
- installs required dependencies;
- skips cloning if the repository already exists.

### download_dataset.py

- installs Roboflow if necessary;
- downloads the dataset;
- skips downloading if the dataset already exists.

### yolo_train.py

- checks project structure;
- automatically detects GPU or CPU;
- starts model training;
- saves the best model to

```
runs/train/dog_breeds/weights/best.pt
```

### yolo_predict.py

Runs object detection on images located in

```
test_images/
```

Predictions are saved to

```
predictions/
```

---

## Results

After training, YOLOv5 automatically saves:

- training curves;
- loss plots;
- precision/recall metrics;
- mAP metrics;
- confusion matrix;
- the best trained model.

---

## Technologies

- Python 3.10
- PyTorch
- YOLOv5
- Roboflow
- UV package manager

---

## Notes

The project automatically avoids unnecessary downloads:

- YOLOv5 is cloned only if it is not already present.
- The dataset is downloaded only if it does not already exist.
- Predictions are generated using the best trained model.