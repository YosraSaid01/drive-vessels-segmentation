"""
Preprocess all DRIVE training images for retinal vessel segmentation.

This script:
1. Loads all 20 training images and their vessel masks (indices 21–40).
2. Converts images to RGB, normalizes them to [0, 1].
3. Resizes them to a consistent size (512x512).
4. Applies the field-of-view (FOV) mask to remove black borders.
5. Saves the preprocessed images as PNGs in 'preprocessed_images/'.

"""

import os
import cv2
import numpy as np
from tqdm import tqdm


def load_image_and_masks(base_dir, idx):
    """
    Load a training image, its corresponding vessel mask, and field-of-view mask.

    Parameters
    ----------
    base_dir : str
        Path to the DRIVE 'training' directory.
    idx : int
        Image index (21 to 40).

    Returns
    -------
    image_rgb : np.ndarray
        RGB image as float32 array.
    vessel_mask : np.ndarray
        Binary vessel segmentation mask.
    fov_mask : np.ndarray
        Binary field-of-view mask.
    """
    img_path = os.path.join(base_dir, "images", f"{idx:02d}_training.tif")
    vessel_path = os.path.join(base_dir, "1st_manual", f"{idx:02d}_manual1.gif")
    fov_path = os.path.join(base_dir, "mask", f"{idx:02d}_training_mask.gif")

    # Read image and masks
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vessel_mask = cv2.imread(vessel_path, cv2.IMREAD_GRAYSCALE)
    fov_mask = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)

    return image_rgb, vessel_mask, fov_mask


def preprocess_image(image_rgb, vessel_mask, fov_mask, target_size=(512, 512)):
    """
    Normalize, resize, and apply FOV mask to an image.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image.
    vessel_mask : np.ndarray
        Vessel segmentation mask.
    fov_mask : np.ndarray
        Field-of-view mask.
    target_size : tuple of int
        Target output size (width, height).

    Returns
    -------
    image_processed : np.ndarray
        Preprocessed RGB image (float32, scaled 0–1).
    mask_processed : np.ndarray
        Binary vessel mask (uint8).
    """
    # Convert to float and normalize
    image_float = image_rgb.astype(np.float32) / 255.0

    # Apply FOV mask
    if fov_mask is not None:
        image_float[fov_mask == 0] = 0

    # Resize image and mask
    image_resized = cv2.resize(image_float, target_size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(vessel_mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Binarize mask
    mask_resized = (mask_resized > 0).astype(np.uint8) * 255

    return image_resized, mask_resized


def save_preprocessed(image, mask, output_dir, idx):
    """
    Save preprocessed image and mask to disk.

    Parameters
    ----------
    image : np.ndarray
        RGB image in [0,1].
    mask : np.ndarray
        Binary mask in {0,255}.
    output_dir : str
        Folder where preprocessed data will be saved.
    idx : int
        Image index.
    """
    os.makedirs(output_dir, exist_ok=True)

    img_name = os.path.join(output_dir, f"{idx:02d}_image.png")
    mask_name = os.path.join(output_dir, f"{idx:02d}_mask.png")

    cv2.imwrite(img_name, cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_name, mask)


def main():
    """
    Main function: preprocess all DRIVE training images (indices 21–40).
    """
    base_dir = r"C:\Users\yosra\projects\data\training"  # adjust if needed
    output_dir = r"C:\Users\yosra\projects\data\preprocessed_images"
    target_size = (512, 512)

    print("Starting preprocessing of DRIVE training images (21–40)...")
    for idx in tqdm(range(21, 41), desc="Processing images"):
        try:
            image_rgb, vessel_mask, fov_mask = load_image_and_masks(base_dir, idx)
            image_processed, mask_processed = preprocess_image(
                image_rgb, vessel_mask, fov_mask, target_size=target_size
            )
            save_preprocessed(image_processed, mask_processed, output_dir, idx)
        except Exception as e:
            print(f"❌ Error processing image {idx:02d}: {e}")
            continue

    print(f"✅ All preprocessed images saved in: {output_dir}")


if __name__ == "__main__":
    main()
