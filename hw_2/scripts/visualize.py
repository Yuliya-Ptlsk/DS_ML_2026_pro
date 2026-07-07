import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from scripts.utils import denormalize


# Count MRI slices with tumor and without
def count_samples_balance(samples):
    positive_samples = []
    negative_samples = []

    samples_count = len(samples)

    for sample in samples:
        if np.array(Image.open(sample["mask"])).max() > 0:
            positive_samples.append(sample)
        else:
            negative_samples.append(sample)

    print(f"Total samples: {samples_count}")
    print(f"MRI with tumor: {len(positive_samples)} / {100 * len(positive_samples) / samples_count:.2f}%")
    print(f"MRI without tumor: {len(negative_samples)} / {100 * len(negative_samples) / samples_count:.2f}%")

    return positive_samples, negative_samples


def visualize_raw_data(rnd_samples, title):
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(12, 18))
    fig.suptitle(title, fontsize=14, y=1.01)

    for row, sample in enumerate(rnd_samples):
        image = np.array(Image.open(sample["img"]).convert("RGB"))
        mask = np.array(Image.open(sample["mask"]).convert("L"))

        axes[row, 0].imshow(image)
        axes[row, 0].set_title("MRI", fontsize=9)

        axes[row, 1].imshow(mask, cmap="hot")
        axes[row, 1].set_title("Mask", fontsize=9)

        axes[row, 2].imshow(image)
        axes[row, 2].imshow(mask, cmap="Reds", alpha=0.6)
        axes[row, 2].set_title("Overlay", fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def visualize_training_history(num_epochs, train_losses, test_losses, dice_scores, iou_scores):
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(14, 4))

    # Loss
    plt.subplot(1, 3, 1)

    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Train",
    )

    plt.plot(
        epochs,
        test_losses,
        marker="o",
        label="Test",
    )

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Dice
    plt.subplot(1, 3, 2)

    plt.plot(
        epochs,
        dice_scores,
        marker="o",
    )

    plt.title("Dice Score")

    plt.xlabel("Epoch")

    plt.ylabel("Dice")

    plt.grid(True)

    # IoU
    plt.subplot(1, 3, 3)

    plt.plot(
        epochs,
        iou_scores,
        marker="o",
    )

    plt.title("IoU")

    plt.xlabel("Epoch")

    plt.ylabel("IoU")

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def visualize_prediction(model, test_data, device):
    model.eval()

    images, masks = next(iter(test_data))

    images = images.to(device)

    with torch.no_grad():
        predictions = model(images)

        predictions = torch.sigmoid(predictions)

        predictions = (predictions > 0.5).float()

    images = images.cpu()

    masks = masks.cpu()

    predictions = predictions.cpu()

    num_examples = min(4, len(images))

    plt.figure(figsize=(20, 4 * num_examples))

    for i in range(num_examples):
        image = denormalize(images[i])
        image = image.permute(1, 2, 0).numpy()

        gt = masks[i, 0].numpy()

        pred = predictions[i, 0].numpy()

        # MRI
        plt.subplot(num_examples, 5, i * 5 + 1)

        plt.imshow(image)

        plt.title("MRI")

        plt.axis("off")

        # Ground Truth
        plt.subplot(num_examples, 5, i * 5 + 2)

        plt.imshow(gt, cmap="gray")

        plt.title("Ground Truth")

        plt.axis("off")

        # Prediction
        plt.subplot(num_examples, 5, i * 5 + 3)

        plt.imshow(pred, cmap="gray")

        plt.title("Prediction")

        plt.axis("off")

        # Ground Truth Overlay
        plt.subplot(num_examples, 5, i * 5 + 4)

        plt.imshow(image)

        plt.imshow(
            gt,
            cmap="Reds",
            alpha=0.45,
        )

        plt.title("GT Overlay")

        plt.axis("off")

        # Prediction Overlay
        plt.subplot(num_examples, 5, i * 5 + 5)

        plt.imshow(image)

        plt.imshow(
            pred,
            cmap="Reds",
            alpha=0.45,
        )

        plt.title("Prediction Overlay")

        plt.axis("off")

    plt.tight_layout()

    plt.show()