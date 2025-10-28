import os
import cv2
import numpy as np
from tqdm import tqdm


def load_chase_image_and_mask(base_dir, name):
    """
    Load CHASE_DB1 image and its corresponding vessel mask.
    Uses the 1st human observer annotation (_1stHO.png).
    """
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
    """
    Automatically detect and create a binary FOV mask from the image
    by finding the largest bright contour (retina region).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray, dtype=np.uint8)

    # Select the largest contour (the retina)
    largest_contour = max(contours, key=cv2.contourArea)
    fov_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(fov_mask, [largest_contour], -1, color=255, thickness=-1)

    return fov_mask


def preprocess_image(image_rgb, vessel_mask, target_size=(512, 512)):
    """Normalize, detect retina region, mask background, and resize."""
    fov_mask = detect_fov_mask(image_rgb)

    # Apply FOV mask (remove background)
    image_rgb[fov_mask == 0] = 0

    # Normalize to [0, 1]
    image_float = image_rgb.astype(np.float32) / 255.0

    # Resize image and mask
    image_resized = cv2.resize(image_float, target_size, interpolation=cv2.INTER_AREA)
    mask_resized = cv2.resize(vessel_mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.uint8) * 255

    return image_resized, mask_resized


def save_preprocessed(image, mask, output_dir, name):
    """Save preprocessed image and mask as PNG."""
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)

    img_path = os.path.join(output_dir, "images", f"{name}.png")
    mask_path = os.path.join(output_dir, "masks", f"{name}_mask.png")

    cv2.imwrite(img_path, cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask)


def main():
    base_dir = r"C:\Users\yosra\projects\data\Chase_DB1"
    output_dir = r"C:\Users\yosra\projects\data\preprocessed_chase_DB1_test"
    target_size = (512, 512)

    print("🔹 Preprocessing CHASE_DB1 test set (with automatic FOV removal)...")
    for fname in tqdm(os.listdir(base_dir)):
        if fname.endswith(".jpg"):
            name = os.path.splitext(fname)[0]
            try:
                image_rgb, vessel_mask = load_chase_image_and_mask(base_dir, name)
                image_processed, mask_processed = preprocess_image(image_rgb, vessel_mask, target_size)
                save_preprocessed(image_processed, mask_processed, output_dir, name)
            except Exception as e:
                print(f"❌ Error processing {name}: {e}")
                continue

    print(f"\n✅ All preprocessed CHASE_DB1 images saved in: {output_dir}")


if __name__ == "__main__":
    main()
