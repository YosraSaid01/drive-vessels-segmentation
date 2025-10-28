


r"""
Preprocess DRIVE training and test images for retinal vessel segmentation.
This script:
1. Loads DRIVE training images (21–40) and splits them into 80% train / 20% val.
2. Loads DRIVE test images (1–20) — without manual vessel annotations.
3. Converts all images to RGB, normalizes to [0, 1], resizes to 512×512,
   applies FOV masks, and saves preprocessed PNGs.
Output structure:
C:\Users\yosra\projects\data\preprocessed_data\
    ├── train\
    │    ├── images\
    │    └── masks\
    ├── val\
    │    ├── images\
    │    └── masks\
    └── test\
         └── images\
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import random


def load_image_and_masks(base_dir, idx, split="training", load_mask=True):
    """Load an image, optional vessel mask, and FOV mask."""
    img_path = os.path.join(base_dir, "images", f"{idx:02d}_{split}.tif")
    fov_path = os.path.join(base_dir, "mask", f"{idx:02d}_{split}_mask.gif")

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fov_mask = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)

    vessel_mask = None
    if load_mask:
        vessel_path = os.path.join(base_dir, "1st_manual", f"{idx:02d}_manual1.gif")
        vessel_mask = cv2.imread(vessel_path, cv2.IMREAD_GRAYSCALE)
        if vessel_mask is None:
            raise FileNotFoundError(f"Could not read {vessel_path}")

    return image_rgb, vessel_mask, fov_mask


def preprocess_image(image_rgb, vessel_mask, fov_mask, target_size=(512, 512)):
    """Normalize, resize, and apply FOV mask."""
    image_float = image_rgb.astype(np.float32) / 255.0
    if fov_mask is not None:
        image_float[fov_mask == 0] = 0

    image_resized = cv2.resize(image_float, target_size, interpolation=cv2.INTER_AREA)

    mask_resized = None
    if vessel_mask is not None:
        mask_resized = cv2.resize(vessel_mask, target_size, interpolation=cv2.INTER_NEAREST)
        mask_resized = (mask_resized > 0).astype(np.uint8) * 255

    return image_resized, mask_resized


def save_preprocessed(image, mask, output_dir, idx, save_mask=True):
    """Save preprocessed image (and mask if available)."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    img_path = os.path.join(output_dir, "images", f"{idx:02d}_image.png")
    cv2.imwrite(img_path, cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    if save_mask and mask is not None:
        os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
        mask_path = os.path.join(output_dir, "masks", f"{idx:02d}_mask.png")
        cv2.imwrite(mask_path, mask)


def preprocess_and_save(indices, base_dir, output_dir, split_name, load_mask=True, target_size=(512, 512)):
    """Preprocess and save all images for a given split."""
    print(f"🔹 Preprocessing {split_name} set...")
    for idx in tqdm(indices, desc=f"{split_name}"):
        try:
            image_rgb, vessel_mask, fov_mask = load_image_and_masks(base_dir, idx, split=split_name, load_mask=load_mask)
            image_processed, mask_processed = preprocess_image(image_rgb, vessel_mask, fov_mask, target_size)
            save_preprocessed(image_processed, mask_processed, output_dir, idx, save_mask=load_mask)
        except Exception as e:
            print(f"❌ Error processing image {idx:02d}: {e}")
            continue


def main():
    base_train_dir = r"C:\Users\yosra\projects\data\training"
    base_test_dir = r"C:\Users\yosra\projects\data\test"
    output_root = r"C:\Users\yosra\projects\data\preprocessed_data"
    target_size = (512, 512)

    # Split training indices (21–40) into 80/20
    all_train_indices = list(range(21, 41))
    random.seed(42)
    random.shuffle(all_train_indices)
    split_point = int(len(all_train_indices) * 0.8)
    train_indices = all_train_indices[:split_point]
    val_indices = all_train_indices[split_point:]

    os.makedirs(output_root, exist_ok=True)

    # Process train, val, and test sets
    preprocess_and_save(train_indices, base_train_dir, os.path.join(output_root, "train"), "training", load_mask=True, target_size=target_size)
    preprocess_and_save(val_indices, base_train_dir, os.path.join(output_root, "val"), "training", load_mask=True, target_size=target_size)
    preprocess_and_save(range(1, 21), base_test_dir, os.path.join(output_root, "test"), "test", load_mask=False, target_size=target_size)

    print(f"\n✅ All preprocessed data saved in: {output_root}")


if __name__ == "__main__":
    main()
