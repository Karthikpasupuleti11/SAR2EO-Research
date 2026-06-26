from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from utils.checkpoint import build_generator_from_checkpoint, load_checkpoint
from utils.io import load_sar_png, save_rgb_png


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_input_images(input_dir: Path) -> list[Path]:
    images = sorted(input_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No PNG files found in: {input_dir}")
    return images


@torch.no_grad()
def run_inference(
    weights_path: str | Path,
    input_dir: str | Path,
    output_dir: str | Path,
    image_size: int = 256,
    device: torch.device | None = None,
) -> None:
    device = device or get_device()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(weights_path, device)
    generator, config = build_generator_from_checkpoint(checkpoint, device)
    image_size = int(config.get("training", {}).get("image_size", image_size))

    image_paths = collect_input_images(input_path)
    print(f"Device : {device}")
    print(f"Input  : {input_path} ({len(image_paths)} images)")
    print(f"Output : {output_path}")

    for image_path in tqdm(image_paths, desc="Inference"):
        sar = load_sar_png(image_path, image_size=image_size).unsqueeze(0).to(device)
        fake_optical = generator(sar)
        save_rgb_png(fake_optical[0], output_path / image_path.name)

    print(f"Saved {len(image_paths)} RGB images to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAR-to-EO inference on a directory of SAR PNG patches."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        weights_path=args.weights,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
    )
