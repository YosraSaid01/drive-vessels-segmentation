import torch
import torch.nn as nn

# -----------------------------
# 🔹 Dice Loss
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice


# -----------------------------
# 🔹 Combined BCE + Dice Loss
# -----------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha = weight for BCE, (1-alpha) = weight for Dice.
        """
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, preds, targets):
        loss_bce = self.bce(preds, targets)
        loss_dice = self.dice(preds, targets)
        return self.alpha * loss_bce + (1 - self.alpha) * loss_dice

import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss
    ------------------
    This loss is designed for highly imbalanced segmentation tasks such as
    retinal vessel segmentation where small structures are easily ignored.

    References:
    - Abraham & Khan, "A Novel Focal Tversky Loss Function With Improved Attention U-Net
      for Lesion Segmentation", IEEE ISBI 2019.
      DOI: 10.1109/ISBI.2019.8759329

    Args:
        alpha (float): weight for false positives (default 0.7)
        beta (float): weight for false negatives (default 0.3)
        gamma (float): focal parameter to emphasize hard examples (default 0.75)
        smooth (float): small constant to avoid division by zero
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        y_pred: (N, 1, H, W) raw logits from the model
        y_true: (N, 1, H, W) binary ground truth mask
        """
        # Sigmoid activation
        y_pred = torch.sigmoid(y_pred)

        # Flatten tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # True positives, false positives & false negatives
        tp = (y_true * y_pred).sum()
        fp = ((1 - y_true) * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()

        # Tversky index
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Focal Tversky loss
        loss = (1 - tversky_index) ** self.gamma
        return loss
