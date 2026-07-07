from pathlib import Path
import torch

# create list of dictionaries with image and mask pairs by patients
# [{"patient": id, "img": path, "mask": path}...]
def create_image_mask_pairs(data_dir):
    data_dir = Path(data_dir)
    samples = []

    for patient_dir in sorted(data_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        for img_path in sorted(patient_dir.glob("*.tif")):
            if img_path.stem.endswith("_mask"):
                continue

            mask_path = img_path.with_name(img_path.stem + "_mask.tif")

            if mask_path.exists():
                samples.append({
                    "patient": patient_id,
                    "img": img_path,
                    "mask": mask_path,
                })

    return samples

# Restore image after torchvision Normalize(mean=0.5, std=0.5)
def denormalize(image):
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)

    image = image * std + mean

    return image.clamp(0, 1)