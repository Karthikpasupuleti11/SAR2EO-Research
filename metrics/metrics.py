from __future__ import annotations

from typing import Dict, List

import lpips
import torch
import torch.nn.functional as F
from torchmetrics.image import FrechetInceptionDistance, PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure


class ImageMetrics:
    """Compute PSNR, SSIM, LPIPS, and FID for SAR-to-EO evaluation."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.lpips = lpips.LPIPS(net="alex").to(device)
        self.fid = FrechetInceptionDistance(normalize=True).to(device)

        self._lpips_values: List[float] = []

    @staticmethod
    def _to_fid_input(tensor: torch.Tensor) -> torch.Tensor:
        """Convert [-1,1] NCHW float tensor to [0,1] for FID."""
        images = tensor.detach().float().clamp(-1.0, 1.0)
        return (images + 1.0) / 2.0

    def update_batch(self, fake: torch.Tensor, real: torch.Tensor) -> None:
        fake = fake.to(self.device)
        real = real.to(self.device)

        self.psnr.update(fake, real)
        self.ssim.update(fake, real)
        self._lpips_values.append(float(self.lpips(fake, real).mean().item()))

        self.fid.update(self._to_fid_input(real), real=True)
        self.fid.update(self._to_fid_input(fake), real=False)

    def compute(self) -> Dict[str, float]:
        fid_value = float(self.fid.compute().item()) if self._lpips_values else float("nan")
        return {
            "psnr": float(self.psnr.compute().item()),
            "ssim": float(self.ssim.compute().item()),
            "lpips": float(sum(self._lpips_values) / len(self._lpips_values)),
            "fid": fid_value,
        }
