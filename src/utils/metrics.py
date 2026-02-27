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
    pred = pred.detach().clamp(0.0, 1.0).cpu()
    target = target.detach().clamp(0.0, 1.0).cpu()
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    mse = torch.mean((pred - target) ** 2)
    
    if mse == 0:
        return float("inf")
    
    psnr_value = 10 * torch.log10(1.0 / mse)
    
    return psnr_value.item()


def ssim(pred, target):
    """
    Calculate Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted image tensor, shape (C, H, W) or (N, C, H, W)
        target: Target image tensor, shape (C, H, W) or (N, C, H, W)
        
    Returns:
        SSIM value (typically in [-1, 1], often close to [0, 1] for natural images). Higher is better.
    """
    pred = pred.detach().clamp(0.0, 1.0).cpu()
    target = target.detach().clamp(0.0, 1.0).cpu()
    
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    
    pred_np = pred.numpy()
    target_np = target.numpy()
    
    ssim_values = []
    for i in range(pred_np.shape[0]):
        pred_img = pred_np[i]
        target_img = target_np[i]
        
        if pred_img.shape[0] == 3:
            pred_img = np.transpose(pred_img, (1, 2, 0))
            target_img = np.transpose(target_img, (1, 2, 0))
            channel_axis = 2
        else:
            pred_img = pred_img.squeeze(0)
            target_img = target_img.squeeze(0)
            channel_axis = None
        
        ssim_val = structural_similarity(
            target_img,
            pred_img,
            data_range=1.0,  # Images are in [0, 1]
            channel_axis=channel_axis
        )
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)
