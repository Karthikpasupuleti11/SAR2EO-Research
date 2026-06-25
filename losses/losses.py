from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models
from torchmetrics.image import StructuralSimilarityIndexMeasure


class GANLoss(nn.Module):
    """Least-squares GAN loss used in Pix2Pix."""

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss on relu1_2, relu2_2, relu3_3, relu4_3 features."""

    def __init__(self, layer_weights: List[float] | None = None) -> None:
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg[:4])).eval()
        self.slice2 = nn.Sequential(*list(vgg[4:9])).eval()
        self.slice3 = nn.Sequential(*list(vgg[9:16])).eval()
        self.slice4 = nn.Sequential(*list(vgg[16:23])).eval()

        for param in self.parameters():
            param.requires_grad = False

        self.layer_weights = layer_weights or [1.0, 1.0, 1.0, 1.0]
        self.l1 = nn.L1Loss()
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.0) / 2.0
        mean = self.imagenet_mean.to(dtype=x.dtype, device=x.device)
        std = self.imagenet_std.to(dtype=x.dtype, device=x.device)
        return (x - mean) / std

    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        h = x
        for block in (self.slice1, self.slice2, self.slice3, self.slice4):
            h = block(h)
            feats.append(h)
        return feats

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_in = self._preprocess(fake)
        real_in = self._preprocess(real)

        fake_feats = self._extract_features(fake_in)
        real_feats = self._extract_features(real_in)

        loss = torch.zeros((), device=fake.device, dtype=fake.dtype)
        for weight, fake_feat, real_feat in zip(self.layer_weights, fake_feats, real_feats):
            loss = loss + weight * self.l1(fake_feat, real_feat)
        return loss


class GeneratorLosses(nn.Module):
    def __init__(
        self,
        lambda_l1: float = 100.0,
        lambda_perceptual: float = 10.0,
        lambda_ssim: float = 1.0,
        use_gan: bool = True,
        use_perceptual: bool = True,
        use_ssim: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.use_gan = use_gan
        self.use_perceptual = use_perceptual
        self.use_ssim = use_ssim

        self.gan_loss = GANLoss()
        self.l1_loss = nn.L1Loss()

        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None

        if use_ssim:
            self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0)
            if device is not None:
                self.ssim_metric = self.ssim_metric.to(device)
        else:
            self.ssim_metric = None

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_gan:
            return torch.zeros((), device=real_pred.device)

        loss_real = self.gan_loss(real_pred, target_is_real=True)
        loss_fake = self.gan_loss(fake_pred, target_is_real=False)
        return 0.5 * (loss_real + loss_fake)

    def generator_loss(
        self,
        fake_pred: torch.Tensor,
        fake_optical: torch.Tensor,
        real_optical: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_l1 = self.l1_loss(fake_optical, real_optical) * self.lambda_l1

        if self.use_gan:
            loss_gan = self.gan_loss(fake_pred, target_is_real=True)
        else:
            loss_gan = torch.zeros((), device=fake_optical.device)

        if self.use_perceptual and self.perceptual_loss is not None:
            loss_perceptual = self.perceptual_loss(fake_optical, real_optical) * self.lambda_perceptual
        else:
            loss_perceptual = torch.zeros((), device=fake_optical.device)

        if self.use_ssim and self.ssim_metric is not None:
            ssim_value = self.ssim_metric(fake_optical, real_optical)
            loss_ssim = (1.0 - ssim_value) * self.lambda_ssim
        else:
            loss_ssim = torch.zeros((), device=fake_optical.device)

        loss_total = loss_l1 + loss_gan + loss_perceptual + loss_ssim

        return {
            "loss_total": loss_total,
            "loss_gan": loss_gan,
            "loss_l1": loss_l1,
            "loss_perceptual": loss_perceptual,
            "loss_ssim": loss_ssim,
        }
