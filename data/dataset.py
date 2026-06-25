from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import get_geometric_transforms, preprocess_pair


class SAROpticalDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        train: bool = False,
        image_size: int = 256,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.train = train
        self.image_size = image_size
        self.transform = get_geometric_transforms(train=train)
        self.samples = self._load_manifest(self.manifest_path)

    def _load_manifest(self, manifest_path: Path) -> List[Dict[str, Any]]:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")

        with manifest_path.open("r", encoding="utf-8") as handle:
            samples = json.load(handle)

        if not samples:
            raise ValueError(f"Split manifest is empty: {manifest_path}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        sar_path = Path(sample["sar_path"])
        optical_path = Path(sample["optical_path"])

        if not sar_path.exists():
            raise FileNotFoundError(f"SAR image not found: {sar_path}")
        if not optical_path.exists():
            raise FileNotFoundError(f"Optical image not found: {optical_path}")

        sar = cv2.imread(str(sar_path), cv2.IMREAD_GRAYSCALE)
        optical = cv2.imread(str(optical_path), cv2.IMREAD_COLOR)

        if sar is None:
            raise ValueError(f"Failed to read SAR image: {sar_path}")
        if optical is None:
            raise ValueError(f"Failed to read optical image: {optical_path}")

        optical = cv2.cvtColor(optical, cv2.COLOR_BGR2RGB)

        if sar.shape[:2] != (self.image_size, self.image_size):
            sar = cv2.resize(sar, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if optical.shape[:2] != (self.image_size, self.image_size):
            optical = cv2.resize(
                optical,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )

        tensors = preprocess_pair(optical=optical, sar=sar, transform=self.transform)

        return {
            "sar": tensors["sar"],
            "optical": tensors["optical"],
            "terrain": sample["terrain"],
            "filename": sample["filename"],
        }


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sar": torch.stack([item["sar"] for item in batch], dim=0),
        "optical": torch.stack([item["optical"] for item in batch], dim=0),
        "terrain": [item["terrain"] for item in batch],
        "filename": [item["filename"] for item in batch],
    }


def _smoke_test() -> None:
    dataset = SAROpticalDataset("splits/train.json", train=True)
    sample = dataset[0]

    print("Dataset smoke test")
    print("-" * 40)
    print(f"Samples   : {len(dataset)}")
    print(f"SAR shape : {tuple(sample['sar'].shape)}")
    print(f"EO shape  : {tuple(sample['optical'].shape)}")
    print(f"SAR range : [{sample['sar'].min():.3f}, {sample['sar'].max():.3f}]")
    print(f"EO range  : [{sample['optical'].min():.3f}, {sample['optical'].max():.3f}]")
    print(f"Terrain   : {sample['terrain']}")
    print(f"Filename  : {sample['filename']}")


if __name__ == "__main__":
    _smoke_test()
