from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SAROpticalDataset, collate_batch
from metrics.metrics import ImageMetrics
from utils.checkpoint import build_generator_from_checkpoint, load_checkpoint
from utils.config import load_config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_split_loader(
    split: str,
    config: Dict,
    batch_size: int,
    num_workers: int = 0,
) -> DataLoader:
    splits_dir = Path(config.get("dataset", {}).get("splits_dir", "splits"))
    image_size = int(config.get("training", {}).get("image_size", 256))
    manifest = splits_dir / f"{split}.json"

    dataset = SAROpticalDataset(manifest_path=manifest, train=False, image_size=image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
    )


@torch.no_grad()
def evaluate_split(
    generator,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, float]:
    metrics = ImageMetrics(device=device)
    amp_enabled = use_amp and device.type == "cuda"

    for batch in tqdm(loader, desc="Evaluate", leave=False):
        sar = batch["sar"].to(device, non_blocking=True)
        real_optical = batch["optical"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            fake_optical = generator(sar)

        metrics.update_batch(fake_optical.float(), real_optical.float())

    return metrics.compute()


def save_qualitative_triplets(
    generator,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 5,
    use_amp: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    amp_enabled = use_amp and device.type == "cuda"
    saved = 0

    for batch in loader:
        if saved >= num_samples:
            break

        sar = batch["sar"].to(device, non_blocking=True)
        real_optical = batch["optical"].to(device, non_blocking=True)
        filenames = batch["filename"]

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            fake_optical = generator(sar)

        batch_size = sar.shape[0]
        for idx in range(batch_size):
            if saved >= num_samples:
                break

            sar_vis = sar[idx].detach().float().cpu().repeat(3, 1, 1)
            sar_vis = (sar_vis + 1.0) / 2.0
            real_vis = (real_optical[idx].detach().float().cpu() + 1.0) / 2.0
            fake_vis = (fake_optical[idx].detach().float().cpu() + 1.0) / 2.0
            grid = torch.stack([sar_vis, real_vis, fake_vis], dim=0)

            stem = Path(filenames[idx]).stem
            vutils.save_image(grid, output_dir / f"triplet_{saved + 1:02d}_{stem}.png", nrow=3)
            saved += 1

    print(f"Saved {saved} qualitative triplets to {output_dir}")


def evaluate_checkpoint(
    weights_path: str | Path,
    splits: List[str],
    output_dir: str | Path,
    config_path: str = "configs/config.yaml",
    batch_size: Optional[int] = None,
    num_qualitative: int = 5,
) -> Dict[str, Dict[str, float]]:
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(weights_path, device)
    generator, ckpt_config = build_generator_from_checkpoint(checkpoint, device)

    config = ckpt_config or load_config(config_path)
    batch_size = batch_size or int(config.get("training", {}).get("batch_size", 8))
    num_workers = int(config.get("training", {}).get("num_workers", 0))
    use_amp = bool(config.get("training", {}).get("use_amp", True))

    experiment_name = config.get("experiment", {}).get("name", Path(weights_path).stem)
    results: Dict[str, Dict[str, float]] = {}

    print(f"Device     : {device}")
    print(f"Experiment : {experiment_name}")
    print(f"Weights    : {weights_path}")

    for split in splits:
        loader = build_split_loader(split, config, batch_size=batch_size, num_workers=num_workers)
        print(f"\nEvaluating split: {split} ({len(loader.dataset)} samples)")
        results[split] = evaluate_split(generator, loader, device=device, use_amp=use_amp)
        print(
            f"{split:5} | PSNR: {results[split]['psnr']:.4f} | "
            f"SSIM: {results[split]['ssim']:.4f} | "
            f"LPIPS: {results[split]['lpips']:.4f} | "
            f"FID: {results[split]['fid']:.4f}"
        )

        if split == "test":
            qual_dir = output_path / experiment_name / "qualitative"
            save_qualitative_triplets(
                generator,
                loader,
                device=device,
                output_dir=qual_dir,
                num_samples=num_qualitative,
                use_amp=use_amp,
            )

    metrics_path = output_path / f"{experiment_name}_metrics.json"
    payload = {
        "experiment": experiment_name,
        "weights": str(weights_path),
        "metrics": results,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nSaved metrics: {metrics_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAR2EO checkpoints on val/test splits.")
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
    )
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_qualitative", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        weights_path=args.weights,
        splits=args.splits,
        output_dir=args.output_dir,
        config_path=args.config,
        batch_size=args.batch_size,
        num_qualitative=args.num_qualitative,
    )
