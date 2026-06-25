from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_DATASET_ROOT = Path("datasets/sentinel12/v_2")

VALID_TERRAINS = {
    "agri",
    "barrenland",
    "grassland",
    "urban",
}


def build_metadata(dataset_root: Path) -> pd.DataFrame:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    records = []

    for terrain_dir in sorted(dataset_root.iterdir()):
        if not terrain_dir.is_dir():
            continue

        terrain = terrain_dir.name
        if terrain not in VALID_TERRAINS:
            continue

        s1_dir = terrain_dir / "s1"
        s2_dir = terrain_dir / "s2"

        if not s1_dir.is_dir() or not s2_dir.is_dir():
            raise FileNotFoundError(
                f"Expected s1/ and s2/ under {terrain_dir}, but one or both are missing."
            )

        for sar_path in sorted(s1_dir.glob("*.png")):
            optical_path = s2_dir / sar_path.name.replace("_s1_", "_s2_")
            if not optical_path.exists():
                continue

            parts = sar_path.stem.split("_")
            if len(parts) < 5:
                continue

            roi = parts[0]
            season = parts[1]
            tile = parts[3]
            patch = parts[4]

            records.append(
                {
                    "terrain": terrain,
                    "roi": roi,
                    "season": season,
                    "tile": tile,
                    "patch": patch,
                    "sar_path": str(sar_path),
                    "optical_path": str(optical_path),
                    "filename": sar_path.name,
                }
            )

    if not records:
        raise ValueError(f"No paired SAR/optical samples found under {dataset_root}")

    return pd.DataFrame(records)


def save_metadata(df: pd.DataFrame, output_dir: Path = Path(".")) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "metadata.csv", index=False)
    df.to_parquet(output_dir / "metadata.parquet", index=False)


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 70)
    print(df.head())
    print("=" * 70)
    print(f"Total Samples : {len(df)}")
    print(f"Terrains      : {df['terrain'].nunique()} -> {sorted(df['terrain'].unique())}")
    print(f"ROIs          : {df['roi'].nunique()} -> {sorted(df['roi'].unique())}")
    print(f"Tiles         : {df['tile'].nunique()}")
    print(f"Seasons       : {sorted(df['season'].unique())}")
    print("\nSamples per terrain:")
    print(df.groupby("terrain").size().sort_index().to_string())
    print("\nTiles per terrain:")
    print(df.groupby("terrain")["tile"].nunique().sort_index().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build metadata.csv and metadata.parquet for SAR2EO."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the Sentinel-1/2 paired dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where metadata files will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)

    df = build_metadata(dataset_root)
    save_metadata(df, output_dir=output_dir)
    print_summary(df)

    print("\nSaved:")
    print(output_dir / "metadata.csv")
    print(output_dir / "metadata.parquet")


if __name__ == "__main__":
    main()
