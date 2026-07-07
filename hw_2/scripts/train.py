import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scripts.utils import create_image_mask_pairs
from scripts.dataset import LGGDataset, SegmentationTransform
from torch.utils.data import DataLoader
from scripts.evaluate import SoftDiceLoss, evaluate
from scripts.unet import UNetModel
import platform


def train_UNet(num_epochs):
    isCudaAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if isCudaAvailable else "cpu")

    #  get all MRI images and masks pairs by patients
    # type [{"patient": id, "img": path, "mask": path}...]
    samples = create_image_mask_pairs("../data/kaggle_3m")

    # get list of patients
    patients = sorted({
        sample["patient"] for sample in samples
    })

    # split data by patients to form train, test sets correctly and avoid data leakage
    train_patients, test_patients = train_test_split(
        patients,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    train_samples = [
        sample for sample in samples
        if sample["patient"] in train_patients
    ]

    test_samples = [
        sample for sample in samples
        if sample["patient"] in test_patients
    ]

    train_transform = SegmentationTransform(img_size=(256, 256), train=True)
    test_transform = SegmentationTransform(img_size=(256, 256), train=False)

    train_dataset = LGGDataset(
        samples=train_samples,
        transform=train_transform,
    )
    train_data = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0 if platform.system() == "Windows" else 4,
        pin_memory=isCudaAvailable,
    )

    test_dataset = LGGDataset(
        samples=test_samples,
        transform=test_transform,
    )
    test_data = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0 if platform.system() == "Windows" else 4,
        pin_memory=isCudaAvailable,
    )

    # Check DataLoder
    images, masks = next(iter(train_data))
    print(images.shape, images.dtype)
    print(masks.shape, masks.dtype)

    # create model
    model = UNetModel()
    model = model.to(device)

    # Loss functions
    loss_1 = torch.nn.BCEWithLogitsLoss()
    loss_2 = SoftDiceLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training history
    train_losses = []
    test_losses = []
    dice_scores = []
    iou_scores = []

    # Best model
    best_loss = float("inf")

    # Model training
    print("\nStart training...\n")

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0

        train_bar = tqdm(
            train_data,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for imgs, masks in train_bar:
            # move batch to GPU/CPU
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Forward
            prediction = model(imgs)
            loss = loss_1(prediction, masks) + loss_2(prediction, masks)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item()
            train_bar.set_postfix(
                avg_loss=f"{running_loss / (train_bar.n + 1):.4f}"
            )

        train_loss = running_loss / len(train_data)
        train_losses.append(train_loss)

        # Test
        test_loss, dice, iou = evaluate(
            model=model,
            data_loader = test_data,
            loss_bce=loss_1,
            loss_dice=loss_2,
            device=device,
        )

        test_losses.append(test_loss)
        dice_scores.append(dice)
        iou_scores.append(iou)

        # Save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(
                model.state_dict(),
                "best_model.pth",
            )

        # Epoch summary
        tqdm.write(
            f"Epoch {epoch + 1:2d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Dice: {dice:.4f} | "
            f"IoU: {iou:.4f}"
        )

    print("\nTraining finished.")

    # Load best model
    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device,
        )
    )

    model.eval()

    print("\nBest model loaded.")


    return (
        model,
        test_data,
        device,
        train_losses,
        test_losses,
        dice_scores,
        iou_scores,
    )
