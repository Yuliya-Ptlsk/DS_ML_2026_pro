import torch
from torch import nn as nn


# Evaluates model on test dataset
def evaluate(model, data_loader, loss_bce, loss_dice, device):
    model.eval()

    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for imgs, masks in data_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            pred = model(imgs)

            loss = loss_bce(pred, masks) + loss_dice(pred, masks)
            total_loss += loss.item()

            pred = torch.sigmoid(pred)

            total_dice += dice_score(pred, masks)
            total_iou += iou_score(pred, masks)

    mean_loss = total_loss / len(data_loader)
    mean_dice = total_dice / len(data_loader)
    mean_iou = total_iou / len(data_loader)

    return mean_loss, mean_dice, mean_iou


#  compute dice coef
# pred: Tensor [B, 1, H, W], target: Tensor [B, 1, H, W]
def dice_score(pred, target, eps=1e-6):
    prediction = (pred > 0.5).float()

    prediction = prediction.view(prediction.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (prediction * target).sum(dim=1)

    dice = (
                   2 * intersection + eps
           ) / (
                   prediction.sum(dim=1)
                   + target.sum(dim=1)
                   + eps
           )

    return dice.mean().item()


#  compute intersection over union
# pred: Tensor [B, 1, H, W], target: Tensor [B, 1, H, W]
def iou_score(pred, target, eps=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + eps) / (union + eps)

    return iou.item()


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)

        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        loss = 1 - dice

        return loss.mean()
