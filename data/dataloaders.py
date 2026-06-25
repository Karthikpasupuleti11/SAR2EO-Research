from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml
from torch.utils.data import DataLoader

from .dataset import SAROpticalDataset, collate_batch


def load_config(config_path: str | Path = "configs/config.yaml") -> Dict:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_dataloaders(
    config: Dict | None = None,
    config_path: str | Path = "configs/config.yaml",
) -> Dict[str, DataLoader]:
    if config is None:
        config = load_config(config_path)

    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    splits_dir = Path(dataset_cfg.get("splits_dir", "splits"))
    image_size = int(training_cfg.get("image_size", 256))
    batch_size = int(training_cfg.get("batch_size", 16))
    num_workers = int(training_cfg.get("num_workers", 4))

    loaders: Dict[str, DataLoader] = {}
    split_specs = {
        "train": (splits_dir / "train.json", True),
        "val": (splits_dir / "val.json", False),
        "test": (splits_dir / "test.json", False),
    }

    for split_name, (manifest_path, is_train) in split_specs.items():
        dataset = SAROpticalDataset(
            manifest_path=manifest_path,
            train=is_train,
            image_size=image_size,
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=is_train,
            collate_fn=collate_batch,
        )

    return loaders


def _smoke_test() -> None:
    loaders = build_dataloaders()
    batch = next(iter(loaders["train"]))

    print("Dataloader smoke test")
    print("-" * 40)
    for split_name, loader in loaders.items():
        print(f"{split_name:5} batches: {len(loader):5d} | samples: {len(loader.dataset):5d}")
    print(f"SAR batch shape    : {tuple(batch['sar'].shape)}")
    print(f"Optical batch shape: {tuple(batch['optical'].shape)}")


if __name__ == "__main__":
    _smoke_test()
