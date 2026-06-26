from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


def load_sar_png(path: str | Path, image_size: int = 256) -> torch.Tensor:
    """Load single-channel SAR PNG [0,255] and return tensor (1, H, W) in [-1, 1]."""
    sar = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if sar is None:
        raise ValueError(f"Failed to read SAR image: {path}")

    if sar.shape[:2] != (image_size, image_size):
        sar = cv2.resize(sar, (image_size, image_size), interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(sar.astype(np.float32) / 255.0)
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0)


def tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output (3,H,W) or (B,3,H,W) in [-1,1] to RGB uint8 HWC."""
    if tensor.ndim == 4:
        tensor = tensor[0]

    rgb = tensor.detach().float().cpu().clamp(-1.0, 1.0)
    rgb = ((rgb + 1.0) / 2.0 * 255.0).byte().numpy()
    rgb = np.transpose(rgb, (1, 2, 0))
    return rgb


def save_rgb_png(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = tensor_to_rgb_uint8(tensor)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
