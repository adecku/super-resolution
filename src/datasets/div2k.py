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
        
        # Validate split
        if split not in ["train", "val"]:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        
        # Validate scale
        if scale not in [2, 4]:
            raise ValueError(f"scale must be 2 or 4, got {scale}")
        
        # Map folders
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
            else:  # scale == 4
                lr_folder = self.root / "DIV2K_valid_LR_bicubic_X4/X4"
        
        # Check if folders exist
        if not hr_folder.exists():
            raise ValueError(f"HR folder does not exist: {hr_folder}")
        if not lr_folder.exists():
            raise ValueError(f"LR folder does not exist: {lr_folder}")
        
        # Build list of image pairs
        hr_files = sorted(hr_folder.glob("*.png"))
        if not hr_files:
            raise ValueError(f"No PNG files found in HR folder: {hr_folder}")
        
        self.pairs = []
        for hr_file in hr_files:
            # LR files have format: {base_name}x{scale}.png
            # e.g., HR: 0001.png -> LR: 0001x2.png (for scale=2) or 0001x4.png (for scale=4)
            hr_stem = hr_file.stem  # e.g., "0001"
            hr_suffix = hr_file.suffix  # e.g., ".png"
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
        
        # Load images as RGB
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        
        # Full image mode (for both train and val)
        if self.full_image:
            # Return full images without cropping
            # For train: no augmentation in full-image mode
            # For val: no augmentation anyway
            lr_tensor = TF.to_tensor(lr_img)  # (C, H, W), [0, 1]
            hr_tensor = TF.to_tensor(hr_img)  # (C, H, W), [0, 1]
            return lr_tensor, hr_tensor
        
        # Patch-based mode (original behavior)
        # Get image dimensions
        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size
        
        # Calculate required HR size for patch
        hr_patch_size = self.patch_size * self.scale
        
        # Check if images are large enough
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
        
        # Patch sampling
        if self.split == "train" and self.augment:
            # Random crop for training
            top = random.randint(0, lr_h - self.patch_size)
            left = random.randint(0, lr_w - self.patch_size)
        else:
            # Center crop for validation
            top = (lr_h - self.patch_size) // 2
            left = (lr_w - self.patch_size) // 2
        
        # Crop LR patch
        lr_patch = TF.crop(lr_img, top, left, self.patch_size, self.patch_size)
        
        # Crop corresponding HR patch (coordinates scaled by scale factor)
        hr_top = top * self.scale
        hr_left = left * self.scale
        hr_patch = TF.crop(hr_img, hr_top, hr_left, hr_patch_size, hr_patch_size)
        
        # Apply augmentations (same for LR and HR)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                lr_patch = TF.hflip(lr_patch)
                hr_patch = TF.hflip(hr_patch)
            
            # Random vertical flip
            if random.random() > 0.5:
                lr_patch = TF.vflip(lr_patch)
                hr_patch = TF.vflip(hr_patch)
            
            # Random rotation (0, 90, 180, 270 degrees)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                lr_patch = TF.rotate(lr_patch, angle)
                hr_patch = TF.rotate(hr_patch, angle)
        
        # Convert to tensors: (C, H, W), float32, range [0, 1]
        lr_tensor = TF.to_tensor(lr_patch)  # Already in [0, 1] range
        hr_tensor = TF.to_tensor(hr_patch)  # Already in [0, 1] range
        
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
    
    # Check if full-image mode is enabled
    train_full_image = cfg.get("train", {}).get("full_image", False)
    val_full_image = cfg.get("eval", {}).get("full_image", False)
    
    # If full-image mode, force batch_size=1
    if train_full_image:
        batch_size = 1
    if val_full_image:
        val_batch_size = 1
    
    # Create datasets
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
    
    # Pin memory for CUDA
    pin_memory = torch.cuda.is_available()
    
    # Create loaders
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


if __name__ == "__main__":
    # Mini-test
    # Add project root to path so imports work when running directly
    import sys
    from pathlib import Path
    
    # Find project root (directory containing 'src' folder)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    from src.config import load_config
    
    # Load config
    cfg = load_config("configs/train_srcnn_x2.yaml")
    
    # Create loaders
    train_loader, val_loader = make_div2k_loaders(cfg)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Get one batch
    lr, hr = next(iter(train_loader))
    
    print(f"\nTrain batch:")
    print(f"  LR shape: {lr.shape}")
    print(f"  HR shape: {hr.shape}")
    print(f"  LR min/max: {lr.min().item():.4f} / {lr.max().item():.4f}")
    print(f"  HR min/max: {hr.min().item():.4f} / {hr.max().item():.4f}")
    
    # Verify scale relationship
    scale = cfg["data"]["scale"]
    assert hr.shape[2] == lr.shape[2] * scale, f"HR H should be LR H * {scale}"
    assert hr.shape[3] == lr.shape[3] * scale, f"HR W should be LR W * {scale}"
    print(f"\nScale verification passed: HR = LR * {scale}")
    
    # Test validation loader
    lr_val, hr_val = next(iter(val_loader))
    print(f"\nVal batch (patch-based):")
    print(f"  LR shape: {lr_val.shape}")
    print(f"  HR shape: {hr_val.shape}")
    print(f"  Validation loader works")
    
    # Test full-image eval
    print(f"\n" + "="*60)
    print("Testing full-image eval mode...")
    print("="*60)
    
    # Create config with full-image eval enabled
    cfg_full = cfg.copy()
    if "eval" not in cfg_full:
        cfg_full["eval"] = {}
    cfg_full["eval"]["full_image"] = True
    
    # Create loaders with full-image eval
    _, val_loader_full = make_div2k_loaders(cfg_full)
    
    print(f"Val batches (full-image): {len(val_loader_full)}")
    
    # Get one sample
    lr_full, hr_full = next(iter(val_loader_full))
    
    print(f"\nFull-image val sample:")
    print(f"  LR shape: {lr_full.shape}")
    print(f"  HR shape: {hr_full.shape}")
    print(f"  LR min/max: {lr_full.min().item():.4f} / {lr_full.max().item():.4f}")
    print(f"  HR min/max: {hr_full.min().item():.4f} / {hr_full.max().item():.4f}")
    
    # Verify scale relationship
    assert hr_full.shape[2] == lr_full.shape[2] * scale, f"HR H should be LR H * {scale}"
    assert hr_full.shape[3] == lr_full.shape[3] * scale, f"HR W should be LR W * {scale}"
    print(f"\nScale verification passed: HR = LR * {scale}")
    print("✓ Full-image eval works correctly!")
    
    # Test full-image train
    print(f"\n" + "="*60)
    print("Testing full-image train mode...")
    print("="*60)
    
    # Create config with full-image train enabled
    cfg_train_full = cfg.copy()
    if "train" not in cfg_train_full:
        cfg_train_full["train"] = {}
    cfg_train_full["train"]["full_image"] = True
    
    # Create loaders with full-image train
    train_loader_full, _ = make_div2k_loaders(cfg_train_full)
    
    print(f"Train batches (full-image): {len(train_loader_full)}")
    
    # Get one sample
    lr_train_full, hr_train_full = next(iter(train_loader_full))
    
    print(f"\nFull-image train sample:")
    print(f"  LR shape: {lr_train_full.shape}")
    print(f"  HR shape: {hr_train_full.shape}")
    print(f"  LR min/max: {lr_train_full.min().item():.4f} / {lr_train_full.max().item():.4f}")
    print(f"  HR min/max: {hr_train_full.min().item():.4f} / {hr_train_full.max().item():.4f}")
    
    # Verify scale relationship
    assert hr_train_full.shape[2] == lr_train_full.shape[2] * scale, f"HR H should be LR H * {scale}"
    assert hr_train_full.shape[3] == lr_train_full.shape[3] * scale, f"HR W should be LR W * {scale}"
    print(f"\nScale verification passed: HR = LR * {scale}")
    print("✓ Full-image train works correctly!")


