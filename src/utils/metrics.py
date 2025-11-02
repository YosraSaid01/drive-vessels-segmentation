import torch

def dice_coefficient(preds, targets, eps=1e-6, threshold=0.5):
    """
    Computes the mean Dice coefficient over a batch.
    Both preds and targets should be tensors of shape (B, 1, H, W).
    """
    # Apply sigmoid to logits to get probabilities
    preds = torch.sigmoid(preds)

    # Threshold predictions to binary mask
    preds = (preds > threshold).float()
    targets = (targets > 0.5).float()

    # Compute Dice per sample, then average
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)

    # Return mean Dice across batch (not detached)
    return dice.mean()


def iou_score(preds, targets, eps=1e-6, threshold=0.5):
    """
    Computes the mean IoU (Jaccard index) over a batch.
    """
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    targets = (targets > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)

    return iou.mean()
