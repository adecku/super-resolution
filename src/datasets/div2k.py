"""DIV2K dataset for image super-resolution."""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


class DIV2KDataset(Dataset):
    """Dataset for DIV2K super-resolution training."""
    
    def __init__(self, root, split="train", scale=2, patch_size=96, augment=True, full_image=False):
        """
        Initialize DIV2K dataset.
        
        Args:
            root: Path to DIV2K root directory (e.g., "data/raw/DIV2K")
            split: "train" or "val"
            scale: Super-resolution scale factor (2 or 4)
            patch_size: Size of LR patch in pixels
            augment: Whether to apply data augmentation (True for train, False for val)
            full_image: If True, return full images without cropping (works for both train and val)
        """
        self.root = Path(root)
        self.split = split
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.full_image = full_image
        
        if split not in ["train", "val"]:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        
        if scale not in [2, 4]:
            raise ValueError(f"scale must be 2 or 4, got {scale}")
        
        if split == "train":
            hr_folder = self.root / "DIV2K_train_HR"
            if scale == 2:
                lr_folder = self.root / "DIV2K_train_LR_bicubic/X2"
            else:  # scale == 4
                lr_folder = self.root / "DIV2K_train_LR_bicubic_X4/X4"
        else:  # split == "val"
            hr_folder = self.root / "DIV2K_valid_HR"
            if scale == 2:
                lr_folder = self.root / "DIV2K_valid_LR_bicubic/X2"
            else:
                lr_folder = self.root / "DIV2K_valid_LR_bicubic_X4/X4"
        
        if not hr_folder.exists():
            raise ValueError(f"HR folder does not exist: {hr_folder}")
        if not lr_folder.exists():
            raise ValueError(f"LR folder does not exist: {lr_folder}")
        
        hr_files = sorted(hr_folder.glob("*.png"))
        if not hr_files:
            raise ValueError(f"No PNG files found in HR folder: {hr_folder}")
        
        self.pairs = []
        for hr_file in hr_files:
            hr_stem = hr_file.stem
            hr_suffix = hr_file.suffix
            lr_filename = f"{hr_stem}x{self.scale}{hr_suffix}"
            lr_file = lr_folder / lr_filename
            if not lr_file.exists():
                raise ValueError(f"LR file not found for HR file {hr_file.name}: {lr_file}")
            self.pairs.append((lr_file, hr_file))
        
        if not self.pairs:
            raise ValueError(f"No valid LR/HR pairs found in {hr_folder} and {lr_folder}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        
        if self.full_image:
            lr_tensor = TF.to_tensor(lr_img)
            hr_tensor = TF.to_tensor(hr_img)
            return lr_tensor, hr_tensor
        
        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size
        
        hr_patch_size = self.patch_size * self.scale
        
        if lr_w < self.patch_size or lr_h < self.patch_size:
            raise ValueError(
                f"LR image {lr_path.name} is too small: {lr_w}x{lr_h}, "
                f"required at least {self.patch_size}x{self.patch_size}"
            )
        if hr_w < hr_patch_size or hr_h < hr_patch_size:
            raise ValueError(
                f"HR image {hr_path.name} is too small: {hr_w}x{hr_h}, "
                f"required at least {hr_patch_size}x{hr_patch_size}"
            )
        
        if self.split == "train" and self.augment:
            top = random.randint(0, lr_h - self.patch_size)
            left = random.randint(0, lr_w - self.patch_size)
        else:
            top = (lr_h - self.patch_size) // 2
            left = (lr_w - self.patch_size) // 2
        
        lr_patch = TF.crop(lr_img, top, left, self.patch_size, self.patch_size)
        
        hr_top = top * self.scale
        hr_left = left * self.scale
        hr_patch = TF.crop(hr_img, hr_top, hr_left, hr_patch_size, hr_patch_size)
        
        if self.augment:
            if random.random() > 0.5:
                lr_patch = TF.hflip(lr_patch)
                hr_patch = TF.hflip(hr_patch)
            
            if random.random() > 0.5:
                lr_patch = TF.vflip(lr_patch)
                hr_patch = TF.vflip(hr_patch)
            
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                lr_patch = TF.rotate(lr_patch, angle)
                hr_patch = TF.rotate(hr_patch, angle)
        
        lr_tensor = TF.to_tensor(lr_patch)
        hr_tensor = TF.to_tensor(hr_patch)
        
        return lr_tensor, hr_tensor


def make_div2k_loaders(cfg):
    """
    Create train and validation DataLoaders for DIV2K dataset.
    
    Args:
        cfg: Configuration dictionary from load_config
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    root = Path(cfg["paths"]["data_root"])
    scale = cfg["data"]["scale"]
    patch_size = cfg["data"]["patch_size"]
    batch_size = cfg["data"]["batch_size"]
    val_batch_size = cfg["data"]["val_batch_size"]
    num_workers = cfg["runtime"]["num_workers"]
    
    train_full_image = cfg.get("train", {}).get("full_image", False)
    val_full_image = cfg.get("eval", {}).get("full_image", False)
    
    if train_full_image:
        batch_size = 1
    if val_full_image:
        val_batch_size = 1
    
    train_dataset = DIV2KDataset(
        root=root,
        split="train",
        scale=scale,
        patch_size=patch_size,
        augment=not train_full_image,  # No augmentation in full-image mode
        full_image=train_full_image
    )
    
    val_dataset = DIV2KDataset(
        root=root,
        split="val",
        scale=scale,
        patch_size=patch_size,
        augment=False,
        full_image=val_full_image
    )
    
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader
