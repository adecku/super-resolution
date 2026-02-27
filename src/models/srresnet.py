"""SRResNet generator used in SRGAN."""

import math

import torch.nn as nn


class ResidualBlock(nn.Module):
    """Residual block for SRResNet."""

    def __init__(self, num_feats: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_feats),
            nn.PReLU(num_parameters=num_feats),
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_feats),
        )

    def forward(self, x):
        return x + self.body(x)


class UpsampleBlock(nn.Module):
    """Pixel-shuffle upsample block."""

    def __init__(self, num_feats: int, scale: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_feats, num_feats * (scale**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(num_parameters=num_feats),
        )

    def forward(self, x):
        return self.block(x)


class SRResNet(nn.Module):
    """SRResNet generator for x2/x4 super-resolution."""

    def __init__(self, scale: int = 2, num_feats: int = 64, num_blocks: int = 16):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError(f"Scale must be 2 or 4, got {scale}")

        self.head = nn.Sequential(
            nn.Conv2d(3, num_feats, kernel_size=9, padding=4),
            nn.PReLU(num_parameters=num_feats),
        )

        self.body = nn.Sequential(*[ResidualBlock(num_feats) for _ in range(num_blocks)])
        self.body_tail = nn.Sequential(
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_feats),
        )

        upsample_steps = int(math.log2(scale))
        upsample_blocks = [UpsampleBlock(num_feats, scale=2) for _ in range(upsample_steps)]
        self.upsample = nn.Sequential(*upsample_blocks)
        self.tail = nn.Conv2d(num_feats, 3, kernel_size=9, padding=4)

    def forward(self, x):
        x = self.head(x)
        residual = x
        x = self.body(x)
        x = self.body_tail(x)
        x = x + residual
        x = self.upsample(x)
        x = self.tail(x)
        return x
