from __future__ import annotations

from typing import Dict, List, Optional

import lpips
import torch

try:
    from torchmetrics.image import (
        FrechetInceptionDistance,
        PeakSignalNoiseRatio,
        StructuralSimilarityIndexMeasure,
    )
except ImportError:
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.psnr import PeakSignalNoiseRatio
    from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


class ImageMetrics:
    """Compute PSNR, SSIM, LPIPS, and FID for SAR-to-EO evaluation."""

    def __init__(self, device: torch.device, compute_fid: bool = True) -> None:
        self.device = device
        self.psnr = PeakSignalNoiseRatio(data_range=2.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
        self.lpips = lpips.LPIPS(net="alex").to(device)

        self.fid: Optional[FrechetInceptionDistance] = None
        if compute_fid:
            try:
                self.fid = FrechetInceptionDistance(normalize=True).to(device)
            except ModuleNotFoundError:
                print(
                    "Warning: torch-fidelity is not installed; FID will be skipped. "
                    "Install with: pip install torch-fidelity"
                )

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

        if self.fid is not None:
            self.fid.update(self._to_fid_input(real), real=True)
            self.fid.update(self._to_fid_input(fake), real=False)

    def compute(self) -> Dict[str, float]:
        if self.fid is not None:
            fid_value = float(self.fid.compute().item())
        else:
            fid_value = float("nan")

        return {
            "psnr": float(self.psnr.compute().item()),
            "ssim": float(self.ssim.compute().item()),
            "lpips": float(sum(self._lpips_values) / len(self._lpips_values)),
            "fid": fid_value,
        }
