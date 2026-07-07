import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF, InterpolationMode


class LGGDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["img"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def __len__(self):
        return len(self.samples)


# Transformations for image segmentation
# Applies identical geometric transformations to image and mask
class SegmentationTransform:
    def __init__(
        self,
        img_size=(256, 256),
        train=True,
        hflip_prob=0.5,
        vflip_prob=0.5,
        max_rotation=20,
    ):
        self.img_size = img_size
        self.train = train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.max_rotation = max_rotation

    def __call__(self, image, mask):
        # Resize image and mask
        image = TF.resize(
            image,
            self.img_size,
            interpolation=InterpolationMode.BILINEAR,
        )

        mask = TF.resize(
            mask,
            self.img_size,
            interpolation=InterpolationMode.NEAREST,
        )

        # Augmentation
        if self.train:
            # Horizontal flip
            if random.random() < self.hflip_prob:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() < self.vflip_prob:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Rotation
            angle = random.uniform(
                -self.max_rotation,
                self.max_rotation,
            )

            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
            )

            mask = TF.rotate(
                mask,
                angle,
                interpolation=InterpolationMode.NEAREST,
            )

        # Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        # Normalization
        image = TF.normalize(
            image,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        )

        mask = (mask > 0).float()

        return image, mask
