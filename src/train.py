"""
train.py — Training script for retinal vessel segmentation using U-Net.

This script:
1. Loads parameters and paths from configs/config.yaml.
2. Loads preprocessed DRIVE data (train / val folders with images & masks).
3. Trains a U-Net model on binary vessel masks.
4. Evaluates on validation set to monitor overfitting.
5. Saves the best model checkpoint.

Author: Yosra Said
"""

import os
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import RetinalDataset
from src.model.unet_model import UNet
from src.utils.metrics import dice_coefficient, iou_score

# -----------------------------
# 1. Load Configuration
# -----------------------------
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

DATA_DIR = cfg["data_dir"]
TRAIN_IMG_DIR = cfg["train_images"]
TRAIN_MASK_DIR = cfg["train_masks"]
VAL_IMG_DIR = cfg["val_images"]
VAL_MASK_DIR = cfg["val_masks"]
CHECKPOINT_DIR = cfg["save_dir"]
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

EPOCHS = cfg["epochs"]
BATCH_SIZE = cfg["batch_size"]
LEARNING_RATE = cfg["learning_rate"]
N_CHANNELS = cfg["n_channels"]
N_CLASSES = cfg["n_classes"]
BILINEAR = cfg["bilinear"]
NUM_WORKERS = cfg["num_workers"]
DEVICE = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")

# -----------------------------
# 2. Datasets and Loaders
# -----------------------------
train_dataset = RetinalDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, augment=True)
val_dataset = RetinalDataset(VAL_IMG_DIR, VAL_MASK_DIR, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# -----------------------------
# 3. Model, Loss, Optimizer
# -----------------------------
model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=BILINEAR).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
bce_loss = nn.BCEWithLogitsLoss()

# -----------------------------
# 4. Training Loop
# -----------------------------
best_val_dice = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

    for images, masks in pbar:
        images = images.to(DEVICE, dtype=torch.float32)
        masks = masks.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(images)
        loss = bce_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / len(train_loader)

    # -----------------------------
    # 5. Validation Step
    # -----------------------------
    model.eval()
    val_loss, val_dice, val_iou = 0.0, 0.0, 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.float32)
            outputs = model(images)
            loss = bce_loss(outputs, masks)

            val_loss += loss.item()
            val_dice += dice_coefficient(outputs, masks)
            val_iou += iou_score(outputs, masks)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Dice: {avg_val_dice:.4f} | IoU: {avg_val_iou:.4f}")

    # -----------------------------
    # 6. Save best checkpoint
    # -----------------------------
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Saved new best model to {checkpoint_path} with Dice {best_val_dice:.4f}")

print("Training complete.")
