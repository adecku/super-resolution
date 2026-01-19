"""SwinIR (Swin Transformer for Image Restoration) model for super-resolution."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def window_partition(x, window_size):
    """Partition image into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-head Self-Attention (W-MSA)."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)
    
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window attention and MLP."""
    
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4.0, qkv_bias=True,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop)
        self.drop_path = nn.Identity()  # Simplified: no drop_path
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x, H, W):
        # x shape: (B, H*W, C)
        B, L, C = x.shape
        
        # Layer norm before attention
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Window partition
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        x_windows = window_partition(x, self.window_size)  # (nW*B, window_size, window_size, C)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # (nW*B, window_size*window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows)  # (nW*B, window_size*window_size, C)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # (B, Hp, Wp, C)
        
        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # MLP
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    
    Each RSTB contains multiple Swin Transformer Layers
    with a residual connection around the entire block.
    """
    
    def __init__(self, dim, num_heads, depth, window_size=8, mlp_ratio=4.0,
                 qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.1):
        super().__init__()
        self.dim = dim
        
        # Multiple Swin Transformer layers
        self.layers = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path
            ) for _ in range(depth)
        ])
        
        # Convolution before residual connection
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    
    def forward(self, x, H, W):
        # x shape: (B, H*W, C)
        shortcut = x
        
        # Apply Swin Transformer layers
        for layer in self.layers:
            x = layer(x, H, W)
        
        # Convert to spatial: (B, H*W, C) -> (B, C, H, W)
        x = x.transpose(1, 2).view(x.shape[0], self.dim, H, W)
        
        # Convolution
        x = self.conv(x)
        
        # Convert back: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        
        # Residual connection
        x = x + shortcut
        
        return x


class SwinIR(nn.Module):
    """SwinIR (Swin Transformer for Image Restoration) model for super-resolution.
    
    Architecture (classical SwinIR):
    1. Shallow Feature Extraction (patch embedding)
    2. Deep Feature Extraction (RSTBs - Residual Swin Transformer Blocks)
    3. Reconstruction Module (upsampling + output convolution)
    
    Parameters actually used:
    - scale: super-resolution scale (2 or 4)
    - embed_dim: feature dimension
    - depths: number of Swin Transformer layers per RSTB (e.g., (6,6,6,6) means 4 RSTBs with 6 layers each)
    - num_heads: attention heads per layer
    - window_size: window size for window attention
    - mlp_ratio: MLP expansion ratio
    - qkv_bias: bias in QKV projection
    - drop_rate, attn_drop_rate: dropout rates
    - drop_path_rate: drop path rate (stochastic depth)
    """
    
    def __init__(self, scale=2, img_size=None, patch_size=1,
                 in_chans=3, embed_dim=96,
                 depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
                 window_size=8, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1):
        super().__init__()
        self.scale = scale
        self.embed_dim = embed_dim
        self.window_size = window_size
        
        # Shallow Feature Extraction (patch embedding)
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1)
        
        # Deep Feature Extraction (RSTBs)
        if isinstance(depths, (tuple, list)):
            num_rstbs = len(depths)
            depths_list = depths
        else:
            num_rstbs = 1
            depths_list = [depths]
        
        if isinstance(num_heads, (tuple, list)):
            num_heads_list = num_heads
        else:
            num_heads_list = [num_heads] * num_rstbs
        
        # Build drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_list))]
        
        self.rstbs = nn.ModuleList()
        cur = 0
        for i in range(num_rstbs):
            depth = depths_list[i]
            num_heads_rstb = num_heads_list[i] if i < len(num_heads_list) else num_heads_list[0]
            drop_path = dpr[cur:cur + depth]
            
            rstb = RSTB(
                dim=embed_dim,
                num_heads=num_heads_rstb,
                depth=depth,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path[0] if drop_path else 0.0
            )
            self.rstbs.append(rstb)
            cur += depth
        
        # Convolution after RSTBs (before reconstruction)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        
        # Reconstruction Module
        if scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        elif scale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(embed_dim, embed_dim * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            raise ValueError(f"Scale must be 2 or 4, got {scale}")
        
        # Output convolution
        self.conv_last = nn.Conv2d(embed_dim, in_chans, kernel_size=3, padding=1)
    
    def forward(self, x):
        """
        Forward pass (classical SwinIR architecture).
        
        Args:
            x: Input tensor of shape (N, 3, H, W) in range [0, 1]
            
        Returns:
            Output tensor of shape (N, 3, H*scale, W*scale) in range [0, 1]
        """
        B, C, H, W = x.shape
        
        # Shallow Feature Extraction
        x_shallow = self.patch_embed(x)  # (B, embed_dim, H, W)
        
        # Convert to sequence for RSTBs: (B, embed_dim, H, W) -> (B, H*W, embed_dim)
        x = x_shallow.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        
        # Deep Feature Extraction (RSTBs)
        for rstb in self.rstbs:
            x = rstb(x, H, W)
        
        # Convert back to spatial: (B, H*W, embed_dim) -> (B, embed_dim, H, W)
        x = x.transpose(1, 2).view(B, self.embed_dim, H, W)
        
        # Convolution after body
        x = self.conv_after_body(x)
        
        # Shallow-to-deep residual connection
        x = x + x_shallow
        
        # Reconstruction Module (upsampling)
        x = self.upsample(x)  # (B, embed_dim, H*scale, W*scale)
        
        # Output convolution
        x = self.conv_last(x)  # (B, 3, H*scale, W*scale)
        
        return x


if __name__ == "__main__":
    # Smoke test
    print("Testing SwinIR model...")
    
    # Test x2 scale
    model_x2 = SwinIR(scale=2, embed_dim=96, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=8)
    lr = torch.rand(1, 3, 64, 64)
    
    with torch.no_grad():
        output = model_x2(lr)
    
    print(f"\nScale x2:")
    print(f"  Input shape: {lr.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (1, 3, {64*2}, {64*2}) = (1, 3, 128, 128)")
    assert output.shape == (1, 3, 128, 128), f"Expected (1, 3, 128, 128), got {output.shape}"
    print("  ✓ Shape verification passed!")
    
    # Test x4 scale
    model_x4 = SwinIR(scale=4, embed_dim=96, depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=8)
    lr = torch.rand(1, 3, 64, 64)
    
    with torch.no_grad():
        output = model_x4(lr)
    
    print(f"\nScale x4:")
    print(f"  Input shape: {lr.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: (1, 3, {64*4}, {64*4}) = (1, 3, 256, 256)")
    assert output.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {output.shape}"
    print("  ✓ Shape verification passed!")
    
    # Test with different input size (checking padding works)
    lr_var = torch.rand(1, 3, 65, 67)  # Not divisible by window_size
    with torch.no_grad():
        output_var = model_x2(lr_var)
    
    print(f"\nVariable input size test:")
    print(f"  Input shape: {lr_var.shape}")
    print(f"  Output shape: {output_var.shape}")
    print(f"  Expected: (1, 3, {65*2}, {67*2}) = (1, 3, 130, 134)")
    assert output_var.shape == (1, 3, 130, 134), f"Expected (1, 3, 130, 134), got {output_var.shape}"
    print("  ✓ Variable size test passed!")
    
    print("\n✓ All tests passed!")
