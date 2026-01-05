"""Metrics for super-resolution evaluation."""

import torch
import numpy as np
from skimage.metrics import structural_similarity


def psnr(pred, target):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted image tensor, shape (C, H, W) or (N, C, H, W)
        target: Target image tensor, shape (C, H, W) or (N, C, H, W)
        
    Returns:
        PSNR value in dB. Returns float("inf") if images are identical.
    """
    # Ensure tensors are on CPU and have same shape
    pred = pred.cpu()
    target = target.cpu()
    
    # Handle both (C, H, W) and (N, C, H, W)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Calculate MSE
    mse = torch.mean((pred - target) ** 2)
    
    # Handle perfect match
    if mse == 0:
        return float("inf")
    
    # Calculate PSNR: 10 * log10(1 / MSE)
    psnr_value = 10 * torch.log10(1.0 / mse)
    
    return psnr_value.item()


def ssim(pred, target):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted image tensor, shape (C, H, W) or (N, C, H, W)
        target: Target image tensor, shape (C, H, W) or (N, C, H, W)
        
    Returns:
        SSIM value in [0, 1]. Higher is better.
    """
    # Ensure tensors are on CPU
    pred = pred.cpu()
    target = target.cpu()
    
    # Handle both (C, H, W) and (N, C, H, W)
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    # Convert to numpy
    pred_np = pred.numpy()
    target_np = target.numpy()
    
    # Process each sample in batch
    ssim_values = []
    for i in range(pred_np.shape[0]):
        pred_img = pred_np[i]  # (C, H, W)
        target_img = target_np[i]  # (C, H, W)
        
        # Convert from (C, H, W) to (H, W, C) for skimage
        if pred_img.shape[0] == 3:  # RGB
            pred_img = np.transpose(pred_img, (1, 2, 0))  # (H, W, C)
            target_img = np.transpose(target_img, (1, 2, 0))  # (H, W, C)
            channel_axis = 2
        else:  # Grayscale
            pred_img = pred_img.squeeze(0)  # (H, W)
            target_img = target_img.squeeze(0)  # (H, W)
            channel_axis = None
        
        # Calculate SSIM
        ssim_val = structural_similarity(
            target_img,
            pred_img,
            data_range=1.0,  # Images are in [0, 1]
            channel_axis=channel_axis
        )
        ssim_values.append(ssim_val)
    
    # Average over batch
    return np.mean(ssim_values)


if __name__ == "__main__":
    # Mini-test: identical images
    # Test with (C, H, W)
    img1 = torch.rand(3, 64, 64)
    img2 = img1.clone()
    
    psnr_val = psnr(img1, img2)
    ssim_val = ssim(img1, img2)
    
    assert psnr_val == float("inf"), f"PSNR should be inf for identical images, got {psnr_val}"
    assert ssim_val == 1.0, f"SSIM should be 1.0 for identical images, got {ssim_val}"
    
    # Test with (N, C, H, W)
    batch1 = torch.rand(4, 3, 64, 64)
    batch2 = batch1.clone()
    
    psnr_batch = psnr(batch1, batch2)
    ssim_batch = ssim(batch1, batch2)
    
    assert psnr_batch == float("inf"), f"PSNR should be inf for identical batch, got {psnr_batch}"
    assert ssim_batch == 1.0, f"SSIM should be 1.0 for identical batch, got {ssim_batch}"
    
    print("All tests passed")

