"""
evaluate.py — Evaluation script for retinal vessel segmentation using U-Net.

This script:
1. Loads the trained U-Net model from a checkpoint.
2. Evaluates it on the test dataset (images & binary masks).
3. Computes Dice and IoU for each image.
4. Saves per-image and mean results into an Excel file.
5. Saves the best and worst Dice predictions as concatenated visualizations:
   Input | Ground Truth | Prediction

Author: Yosra Said
"""

import os
import torch
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from src.dataset import RetinalDataset
from src.model.unet_model import UNet
from src.utils.metrics import dice_coefficient, iou_score
import yaml


# -----------------------------
# 1. Load configuration
# -----------------------------
with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

TEST_IMG_DIR = os.path.join(cfg["data_dir"], "test/images")
TEST_MASK_DIR = os.path.join(cfg["data_dir"], "test/masks")
CHECKPOINT_PATH = os.path.join(cfg["save_dir"], "best_model.pth")
OUTPUT_EXCEL = "outputs/evaluation_results.xlsx"
OUTPUT_VIS_DIR = "outputs/visualizations"

DEVICE = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
N_CHANNELS = cfg["n_channels"]
N_CLASSES = cfg["n_classes"]
BILINEAR = cfg["bilinear"]

# -----------------------------
# 2. Load model
# -----------------------------
print("🔹 Loading trained model from checkpoint...")
model = UNet(n_channels=N_CHANNELS, n_classes=N_CLASSES, bilinear=BILINEAR)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"✅ Model loaded successfully from {CHECKPOINT_PATH}")

# -----------------------------
# 3. Load test dataset
# -----------------------------
print("🔹 Loading test dataset...")
test_dataset = RetinalDataset(TEST_IMG_DIR, TEST_MASK_DIR, augment=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
print(f"✅ Loaded {len(test_dataset)} test images")

# -----------------------------
# 4. Evaluation
# -----------------------------
results = []
dice_scores = []
iou_scores = []

best_dice = -1
worst_dice = 999
best_data = None
worst_data = None

with torch.no_grad():
    for idx, (image, mask) in enumerate(tqdm(test_loader, desc="Evaluating")):
        image = image.to(DEVICE, dtype=torch.float32)
        mask = mask.to(DEVICE, dtype=torch.float32)

        output = model(image)
        dice = dice_coefficient(output, mask)
        iou = iou_score(output, mask)

        dice_scores.append(dice)
        iou_scores.append(iou)

        results.append({
            "Image_Index": idx + 1,
            "Dice": round(dice, 4),
            "IoU": round(iou, 4)
        })

        # Track best and worst Dice
        if dice > best_dice:
            best_dice = dice
            best_data = (image.cpu(), mask.cpu(), output.cpu())
        if dice < worst_dice:
            worst_dice = dice
            worst_data = (image.cpu(), mask.cpu(), output.cpu())

# -----------------------------
# 5. Compute mean scores
# -----------------------------
mean_dice = sum(dice_scores) / len(dice_scores)
mean_iou = sum(iou_scores) / len(iou_scores)
results.append({
    "Image_Index": "Mean",
    "Dice": round(mean_dice, 4),
    "IoU": round(mean_iou, 4)
})

print("\n📊 Evaluation complete:")
print(f"  Mean Dice:  {mean_dice:.4f}")
print(f"  Mean IoU:   {mean_iou:.4f}")
print(f"  Best Dice:  {best_dice:.4f}")
print(f"  Worst Dice: {worst_dice:.4f}")

# -----------------------------
# 6. Save results to Excel
# -----------------------------
os.makedirs("outputs", exist_ok=True)
df = pd.DataFrame(results)
df.to_excel(OUTPUT_EXCEL, index=False)
print(f"✅ Results saved to {OUTPUT_EXCEL}")

# -----------------------------
# 7. Save visualizations
# -----------------------------
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)


def save_concat_visual(image, mask, output, filename):
    """Save concatenated (Input | Ground Truth | Prediction) visualization with titles."""
    img_np = image.squeeze().permute(1, 2, 0).cpu().numpy()  # [H, W, C] RGB
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = torch.sigmoid(output).squeeze().cpu().numpy()
    pred_np = (pred_np > 0.5).astype(np.uint8)

    # --- 🔹 Reverse normalization if applied ---
    if img_np.min() < 0:  # e.g., in [-1,1]
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    elif img_np.max() <= 1.0 and np.mean(img_np) < 0.4:  # possibly ImageNet norm
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)

    # Scale to [0,255]
    img_vis = (img_np * 255).astype(np.uint8)

    # Convert RGB → BGR for OpenCV saving
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # Convert masks to RGB for concatenation
    mask_vis = (mask_np * 255).astype(np.uint8)
    pred_vis = (pred_np * 255).astype(np.uint8)
    mask_vis_rgb = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)
    pred_vis_rgb = cv2.cvtColor(pred_vis, cv2.COLOR_GRAY2BGR)

    # Resize all to the same height
    h = min(img_vis.shape[0], mask_vis_rgb.shape[0], pred_vis_rgb.shape[0])
    img_vis = cv2.resize(img_vis, (h, h))
    mask_vis_rgb = cv2.resize(mask_vis_rgb, (h, h))
    pred_vis_rgb = cv2.resize(pred_vis_rgb, (h, h))

    # Concatenate horizontally
    concat = np.concatenate((img_vis, mask_vis_rgb, pred_vis_rgb), axis=1)

    # --- 🏷️ Add labels on top ---
    titles = ["Input", "Ground Truth", "Prediction"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    color = (255, 255, 255)  # white text

    # Create a black header strip
    header_height = 50
    header = np.zeros((header_height, concat.shape[1], 3), dtype=np.uint8)

    # Compute section width for each panel
    section_w = concat.shape[1] // 3
    for i, title in enumerate(titles):
        text_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        x = int(section_w * i + (section_w - text_size[0]) / 2)
        y = int((header_height + text_size[1]) / 2)
        cv2.putText(header, title, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # Stack header and visualization
    labeled_concat = np.vstack((header, concat))

    # Save image
    save_path = os.path.join(OUTPUT_VIS_DIR, filename)
    cv2.imwrite(save_path, labeled_concat)
    print(f"🖼️  Saved {filename} with titles")







# Save best and worst Dice visualizations
if best_data is not None:
    save_concat_visual(*best_data, filename="best_dice_visualization.png")
if worst_data is not None:
    save_concat_visual(*worst_data, filename="worst_dice_visualization.png")

print(f"✅ Best and worst visualizations saved in {OUTPUT_VIS_DIR}")
