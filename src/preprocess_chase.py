import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def load_chase_image_and_mask(base_dir, name):
    """Load CHASE_DB1 image and its corresponding vessel mask."""
    img_path = os.path.join(base_dir, f"{name}.jpg")
    mask_path = os.path.join(base_dir, f"{name}_1stHO.png")

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read {img_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read {mask_path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), mask


def detect_fov_mask(image_rgb):
    """Detect and create a binary FOV mask from the largest bright contour."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray, dtype=np.uint8)

    largest_contour = max(contours, key=cv2.contourArea)
    fov_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(fov_mask, [largest_contour], -1, color=255, thickness=-1)
    return fov_mask


def preprocess_image(image_rgb, vessel_mask, target_size=(512, 512)):
    """Normalize, mask background, and resize."""
    fov_mask = detect_fov_mask(image_rgb)
    image_rgb[fov_mask == 0] = 0

    image_float = image_rgb.astype(np.float32) / 255.0
    image_resized = cv2.resize(image_float, target_size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(vessel_mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.uint8) * 255
    return image_resized, mask_resized


def save_preprocessed(image, mask, output_dir, split, name):
    """Save preprocessed image and mask into split folders."""
    img_dir = os.path.join(output_dir, split, "images")
    mask_dir = os.path.join(output_dir, split, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    img_path = os.path.join(img_dir, f"{name}.png")
    mask_path = os.path.join(mask_dir, f"{name}_mask.png")

    cv2.imwrite(img_path, cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask)


def main():
    base_dir = r"C:\Users\yosra\projects\data\Chase_DB1"
    output_dir = r"C:\Users\yosra\projects\data\preprocessed_chase_DB1"
    target_size = (512, 512)

    print("🔹 Listing CHASE_DB1 images...")
    all_images = [os.path.splitext(f)[0] for f in os.listdir(base_dir) if f.endswith(".jpg")]
    all_images = sorted(list(set(all_images)))  # ensure unique and ordered

    print(f"Found {len(all_images)} images.")

    # Split into train, val, test (60/20/20)
    train_names, temp_names = train_test_split(all_images, test_size=0.4, random_state=42)
    val_names, test_names = train_test_split(temp_names, test_size=0.5, random_state=42)

    print(f"Train: {len(train_names)} | Val: {len(val_names)} | Test: {len(test_names)}")

    for split, names in zip(["train", "val", "test"], [train_names, val_names, test_names]):
        print(f"\n🔹 Processing {split} set...")
        for name in tqdm(names):
            try:
                image_rgb, vessel_mask = load_chase_image_and_mask(base_dir, name)
                image_processed, mask_processed = preprocess_image(image_rgb, vessel_mask, target_size)
                save_preprocessed(image_processed, mask_processed, output_dir, split, name)
            except Exception as e:
                print(f"❌ Error processing {name}: {e}")
                continue

    print(f"\n✅ All preprocessed CHASE_DB1 data saved in: {output_dir}")


if __name__ == "__main__":
    main()
