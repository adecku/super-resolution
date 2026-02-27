"""Losses used by SRGAN training."""

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class PerceptualLoss(nn.Module):
    """VGG-based perceptual content loss."""

    def __init__(self, feature_layer: int = 35, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = VGG19_Weights.IMAGENET1K_V1
        else:
            weights = None

        vgg = vgg19(weights=weights).features[:feature_layer].eval()
        for param in vgg.parameters():
            param.requires_grad = False

        self.feature_extractor = vgg
        self.criterion = nn.L1Loss()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_features = self.feature_extractor(self._normalize(pred))
        target_features = self.feature_extractor(self._normalize(target))
        return self.criterion(pred_features, target_features)


class GANLoss(nn.Module):
    """BCE-with-logits adversarial loss with optional one-sided label smoothing."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.label_smoothing = float(label_smoothing)
        self.criterion = nn.BCEWithLogitsLoss()

    def generator_loss(self, pred_fake: torch.Tensor) -> torch.Tensor:
        real_targets = torch.ones_like(pred_fake)
        if self.label_smoothing > 0.0:
            real_targets = real_targets * (1.0 - self.label_smoothing)
        return self.criterion(pred_fake, real_targets)

    def discriminator_loss(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        real_targets = torch.ones_like(pred_real)
        if self.label_smoothing > 0.0:
            real_targets = real_targets * (1.0 - self.label_smoothing)
        fake_targets = torch.zeros_like(pred_fake)

        loss_real = self.criterion(pred_real, real_targets)
        loss_fake = self.criterion(pred_fake, fake_targets)
        return 0.5 * (loss_real + loss_fake)
