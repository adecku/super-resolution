"""SRCNN (Super-Resolution CNN) model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    """SRCNN model for image super-resolution.
    
    Architecture:
    1. Bicubic upsampling to target resolution
    2. Three convolutional layers for refinement
    """
    
    def __init__(self, scale=2, channels=64):
        """
        Initialize SRCNN model.
        
        Args:
            scale: Super-resolution scale factor (2 or 4)
            channels: Number of channels in hidden layers
        """
        super().__init__()
        self.scale = scale
        
        # Three convolutional layers
        self.conv1 = nn.Conv2d(3, channels, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(channels, channels // 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(channels // 2, 3, kernel_size=5, padding=2)
        
        # ReLU activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 3, H, W) in range [0, 1]
            
        Returns:
            Output tensor of shape (N, 3, H*scale, W*scale) in range [0, 1]
        """
        x = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        
        return x