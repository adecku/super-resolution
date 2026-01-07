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
        """Forward pass."""
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
        
        # Initial convolution
        self.conv_input = nn.Conv2d(3, num_feats, kernel_size=3, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_feats, res_scale) for _ in range(num_blocks)
        ])
        
        # Convolution after residual blocks
        self.conv_mid = nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        
        # Upsampling
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
        
        # Final convolution
        self.conv_output = nn.Conv2d(num_feats, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, 3, H, W) in range [0, 1]
            
        Returns:
            Output tensor of shape (N, 3, H*scale, W*scale) in range [0, 1]
        """
        # Initial convolution
        x = self.conv_input(x)
        residual = x
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Mid convolution
        x = self.conv_mid(x)
        x = x + residual  # Long skip connection
        
        # Upsampling
        x = self.upsample(x)
        
        # Final convolution
        x = self.conv_output(x)
        
        return x


if __name__ == "__main__":
    # Mini-test
    print("Testing EDSR model...")
    
    # Test x2 scale
    model_x2 = EDSR(scale=2, num_feats=64, num_blocks=16, res_scale=0.1)
    x = torch.rand(1, 3, 96, 96)
    
    with torch.no_grad():
        output = model_x2(x)
    
    print(f"\nScale x2:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (1, 3, {96*2}, {96*2}) = (1, 3, 192, 192)")
    assert output.shape == (1, 3, 192, 192), f"Expected (1, 3, 192, 192), got {output.shape}"
    print("  ✓ Shape verification passed!")
    
    # Test x4 scale
    model_x4 = EDSR(scale=4, num_feats=64, num_blocks=16, res_scale=0.1)
    x = torch.rand(1, 3, 96, 96)
    
    with torch.no_grad():
        output = model_x4(x)
    
    print(f"\nScale x4:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (1, 3, {96*4}, {96*4}) = (1, 3, 384, 384)")
    assert output.shape == (1, 3, 384, 384), f"Expected (1, 3, 384, 384), got {output.shape}"
    print("  ✓ Shape verification passed!")
    
    # Test with different parameters
    model_custom = EDSR(scale=2, num_feats=128, num_blocks=8, res_scale=0.1)
    x = torch.rand(2, 3, 64, 64)
    
    with torch.no_grad():
        output = model_custom(x)
    
    print(f"\nCustom parameters (num_feats=128, num_blocks=8):")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (2, 3, {64*2}, {64*2}) = (2, 3, 128, 128)")
    assert output.shape == (2, 3, 128, 128), f"Expected (2, 3, 128, 128), got {output.shape}"
    print("  ✓ Shape verification passed!")
    
    print("\n✓ All tests passed!")

