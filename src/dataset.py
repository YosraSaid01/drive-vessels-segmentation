"""
dataset.py — Custom PyTorch Dataset for Retinal Vessel Segmentation

This script:
1. Loads preprocessed images and corresponding masks from folders.
2. Applies data augmentation and normalization using Albumentations.
3. Returns image and mask tensors ready for model input.
"""

import os
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RetinalDataset(Dataset):
    """
    Custom Dataset for Retinal Vessel Segmentation.

    Parameters
    ----------
    image_dir : str
        Path to the folder containing input images.
    mask_dir : str
        Path to the folder containing segmentation masks.
    augment : bool
        Whether to apply data augmentation (True for training).
    """

    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.augment = augment
        self.patch_size = 64
        self.stride = 64
        # Data augmentations for training
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=45, p=1),
            A.ElasticTransform(alpha=80, sigma=10, alpha_affine=20, p=0.4),
            A.RandomBrightnessContrast(p=0.6),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(var_limit=(5.0, 25.0), p=0.4),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        # Only normalization for validation/testing
        self.val_transform = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0  # normalize to [0, 1]

        # Apply transformation
        if self.augment:
            transformed = self.train_transform(image=image, mask=mask)
        else:
            transformed = self.val_transform(image=image, mask=mask)

        image = transformed["image"]
        mask = transformed["mask"].unsqueeze(0)  # shape (1, H, W)

        return image, mask
