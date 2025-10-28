import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Paths ---
DATA_DIR = R"C:\Users\yosra\projects\data\training"
IMG_PATH = os.path.join(DATA_DIR, "images", "21_training.tif")
MASK_PATH = os.path.join(DATA_DIR, "1st_manual", "21_manual1.gif")
FOV_PATH = os.path.join(DATA_DIR, "mask", "21_training_mask.gif")
print("Looking for image at:", IMG_PATH)
print("Exists:", os.path.exists(IMG_PATH))

# --- Load image and masks ---
image = cv2.imread(IMG_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
fov = cv2.imread(FOV_PATH, cv2.IMREAD_GRAYSCALE)

# --- Print basic characteristics ---
print("=== IMAGE CHARACTERISTICS ===")
print(f"File: {IMG_PATH}")
print(f"Shape (H, W, C): {image.shape}")
print(f"Data type: {image.dtype}")
print(f"Pixel range: {image.min()} → {image.max()}")
print(f"Mean: {image.mean():.2f}, Std: {image.std():.2f}")
print()
print("=== MASK CHARACTERISTICS ===")
print(f"Shape: {mask.shape}, unique values: {np.unique(mask)}")

# --- Visualize ---
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(image)
plt.title("Original Image")

plt.subplot(1,3,2)
plt.imshow(mask, cmap="gray")
plt.title("Vessel Ground Truth")

plt.subplot(1,3,3)
plt.imshow(fov, cmap="gray")
plt.title("Field of View (FOV)")
plt.show()

# --- Preprocessing steps ---
# 1. Normalize intensity
image_norm = image.astype(np.float32) / 255.0

# 2. Apply FOV mask
image_norm[fov == 0] = 0  # remove background outside FOV

# 3. Resize (optional for training)
IMG_SIZE = (512, 512)
image_resized = cv2.resize(image_norm, IMG_SIZE)
mask_resized = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
fov_resized = cv2.resize(fov, IMG_SIZE, interpolation=cv2.INTER_NEAREST)

# --- Visualize preprocessed sample ---
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(image_resized)
plt.title("Normalized + Resized")

plt.subplot(1,3,2)
plt.imshow(mask_resized, cmap="gray")
plt.title("Mask (Resized)")

plt.subplot(1,3,3)
plt.imshow(fov_resized, cmap="gray")
plt.title("FOV (Resized)")
plt.tight_layout()
plt.show()
