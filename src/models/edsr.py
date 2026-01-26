"""EDSR (Enhanced Deep Super-Resolution) model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block for EDSR."""
    
    def __init__(self, num_feats, res_scale=0.1):
        """
        Initialize residual block.
        
        Args:
            num_feats: Number of feature channels
            res_scale: Residual scaling factor
        """
        super().__init__()
        self.res_scale = res_scale
        
        self.conv1 = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out * self.res_scale + residual
        return out


class EDSR(nn.Module):
    """EDSR (Enhanced Deep Super-Resolution) model.
    
    Architecture:
    1. Initial convolution
    2. Residual blocks
    3. Upsampling (pixel shuffle)
    4. Final convolution
    """
    
    def __init__(self, scale=2, num_feats=64, num_blocks=16, res_scale=0.1):
        """
        Initialize EDSR model.
        
        Args:
            scale: Super-resolution scale factor (2 or 4)
            num_feats: Number of feature channels
            num_blocks: Number of residual blocks
            res_scale: Residual scaling factor
        """
        super().__init__()
        self.scale = scale
        self.num_feats = num_feats
        
        self.conv_input = nn.Conv2d(3, num_feats, kernel_size=3, padding=1)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_feats, res_scale) for _ in range(num_blocks)
        ])
        
        self.conv_mid = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        
        if scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feats, num_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        elif scale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feats, num_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feats, num_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            raise ValueError(f"Scale must be 2 or 4, got {scale}")
        
        self.conv_output = nn.Conv2d(num_feats, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 3, H, W) in range [0, 1]
            
        Returns:
            Output tensor of shape (N, 3, H*scale, W*scale) in range [0, 1]
        """
        x = self.conv_input(x)
        residual = x
        
        for block in self.residual_blocks:
            x = block(x)
        
        x = self.conv_mid(x)
        x = x + residual
        
        x = self.upsample(x)
        x = self.conv_output(x)
        
        return x
