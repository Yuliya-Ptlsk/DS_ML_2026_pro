# Brain MRI Segmentation using U-Net

PyTorch implementation of a U-Net model for automatic brain tumor segmentation on MRI images using the **LGG MRI Segmentation** dataset from Kaggle.

> This project was created as part of a Machine Learning coursework.

---

# Project Overview

The goal of this project is to develop a semantic segmentation model capable of identifying brain tumors on MRI scans.

The project includes:

- downloading the dataset from Kaggle using the Kaggle CLI;
- preparing MRI–mask pairs;
- splitting the dataset by patients to prevent data leakage;
- implementing a custom PyTorch `Dataset` and `DataLoader`;
- applying image augmentation;
- building a U-Net model;
- training and evaluating the model;
- visualizing learning curves and segmentation predictions.

---

# Dataset

**Brain MRI Segmentation**

Dataset:

https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

The dataset contains:

- MRI images (`.tif`)
- binary tumor masks (`*_mask.tif`)
- metadata (`data.csv`)

Each MRI slice has a corresponding binary mask describing the tumor region.

---

# Project Structure

```text
project/
│
├── data/
│   └── kaggle_3m/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── model.ipynb
│
├── scripts/
│   ├── dataset.py
│   ├── download_data.py
│   ├── evaluate.py
│   ├── train.py
│   ├── unet.py
│   └── utils.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/Yuliya-Ptlsk/DS_ML_2026_pro.git
cd DS_ML_2026_pro
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install project dependencies

```bash
pip install -r requirements.txt
```

---

# Kaggle Authentication

The dataset is **not included** in this repository.

This project uses **Kaggle CLI 2.2.3** with **OAuth authentication**.

## Install Kaggle CLI

```bash
pip install kaggle==2.2.3
```

## Authenticate

Run

```bash
kaggle auth login
```

A browser window will open asking you to sign in to your Kaggle account and authorize the Kaggle CLI.

After successful authentication, your credentials are stored locally and reused automatically in future sessions.

> Authentication only needs to be completed once.

---

# Download the Dataset

Run

```bash
python scripts/download_data.py
```

The script automatically:

- authenticates with Kaggle;
- downloads the dataset archive;
- extracts only the `kaggle_3m` directory;
- removes the downloaded archive.

The resulting directory structure will be

```text
data/
└── kaggle_3m/
```

If the dataset has already been downloaded, the script exits without downloading it again.

---

# Train the Model

Open

```text
notebooks/model.ipynb
```

The notebook prepares the dataset, creates the `DataLoader`s, initializes the model, and calls the training function

```python
train_UNet(...)
```

implemented in

```text
scripts/train.py
```

During training, the following metrics are calculated for every epoch:

- Training Loss
- Test Loss
- Dice Coefficient
- Intersection over Union (IoU)

The model with the lowest test loss is automatically saved as

```text
best_model.pth
```

---

# Evaluation

The dataset is split **by patients**, ensuring that MRI slices from the same patient never appear in both the training and test sets.

The model is evaluated using:

- Binary Cross Entropy Loss
- Soft Dice Loss
- Dice Coefficient
- Intersection over Union (IoU)

---

# Model

The project implements a U-Net architecture consisting of:

- encoder–decoder architecture;
- skip connections;
- Batch Normalization;
- ReLU activation;
- binary segmentation output.

---

# Data Preparation

MRI images are automatically paired with their corresponding masks.

Each sample has the following structure:

```python
{
    "patient": "...",
    "img": "...",
    "mask": "..."
}
```

The patient identifier is extracted from the directory name and is used to split the dataset correctly.

---

# Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- scikit-learn
- tqdm
- Kaggle CLI 2.2.3

---

# Results

After training, the project provides:

- training and test loss curves;
- Dice coefficient curve;
- IoU curve;
- qualitative segmentation examples;
- the best trained model (`best_model.pth`).

