from __future__ import annotations

from typing import Any, Dict, Optional

import albumentations as A
import numpy as np
import torch


def normalize_to_minus_one_one(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
    tensor = tensor * 2.0 - 1.0

    if tensor.ndim == 2:
        return tensor.unsqueeze(0)

    return tensor.permute(2, 0, 1)


def get_geometric_transforms(train: bool = True) -> Optional[A.BasicTransform]:
    if not train:
        return None

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ],
        additional_targets={"sar": "image"},
    )


def apply_geometric_transform(
    transform: Optional[A.BasicTransform],
    optical: np.ndarray,
    sar: np.ndarray,
) -> Dict[str, np.ndarray]:
    if transform is None:
        return {"optical": optical, "sar": sar}

    augmented = transform(image=optical, sar=sar)
    return {"optical": augmented["image"], "sar": augmented["sar"]}


def preprocess_pair(
    optical: np.ndarray,
    sar: np.ndarray,
    transform: Optional[A.BasicTransform] = None,
) -> Dict[str, torch.Tensor]:
    augmented = apply_geometric_transform(transform, optical=optical, sar=sar)
    return {
        "optical": normalize_to_minus_one_one(augmented["optical"]),
        "sar": normalize_to_minus_one_one(augmented["sar"]),
    }
