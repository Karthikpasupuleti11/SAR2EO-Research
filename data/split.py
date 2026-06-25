"""
data/split.py

Tile-aware train/val/test splitting with optimal per-terrain tile assignment.
Whole tiles never appear in more than one split (no spatial leakage).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_METADATA_PATHS = (
    Path("metadata.parquet"),
    Path("metadata.csv"),
)


@dataclass
class SplitArtifacts:
    train: List[Dict]
    val: List[Dict]
    test: List[Dict]
    summary: Dict


class DatasetSplitter:
    def __init__(
        self,
        metadata_path: str | Path | None = None,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        self.metadata_path = self._resolve_metadata_path(metadata_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self._validate_ratios()
        self.df = self._load_metadata()

    def _resolve_metadata_path(self, metadata_path: str | Path | None) -> Path:
        if metadata_path is not None:
            path = Path(metadata_path)
            if not path.exists():
                raise FileNotFoundError(f"Metadata file not found: {path}")
            return path

        for candidate in DEFAULT_METADATA_PATHS:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Could not find metadata file. Expected one of: "
            f"{', '.join(str(path) for path in DEFAULT_METADATA_PATHS)}"
        )

    def _validate_ratios(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "Split ratios must sum to 1.0, "
                f"got {self.train_ratio:.3f}/{self.val_ratio:.3f}/{self.test_ratio:.3f}"
            )

        if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0:
            raise ValueError("All split ratios must be greater than zero.")

    def _load_metadata(self) -> pd.DataFrame:
        if self.metadata_path.suffix == ".parquet":
            df = pd.read_parquet(self.metadata_path)
        elif self.metadata_path.suffix == ".csv":
            df = pd.read_csv(self.metadata_path)
        else:
            raise ValueError(
                f"Unsupported metadata format: {self.metadata_path.suffix}"
            )

        required_columns = {
            "terrain",
            "roi",
            "season",
            "tile",
            "patch",
            "sar_path",
            "optical_path",
            "filename",
        }
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            raise ValueError(
                "Metadata is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if df.empty:
            raise ValueError("Metadata file is empty.")

        return df.copy()

    def split(self) -> SplitArtifacts:
        train_frames: List[pd.DataFrame] = []
        val_frames: List[pd.DataFrame] = []
        test_frames: List[pd.DataFrame] = []

        terrains = sorted(self.df["terrain"].astype(str).unique())
        for terrain in terrains:
            terrain_df = self.df[self.df["terrain"].astype(str) == terrain].copy()
            terrain_df["_tile_group"] = terrain_df["tile"].astype(str)

            train_df, val_df, test_df = self._split_single_terrain(terrain_df, terrain)
            train_frames.append(train_df.drop(columns="_tile_group"))
            val_frames.append(val_df.drop(columns="_tile_group"))
            test_frames.append(test_df.drop(columns="_tile_group"))

        train_df = pd.concat(train_frames, ignore_index=True)
        val_df = pd.concat(val_frames, ignore_index=True)
        test_df = pd.concat(test_frames, ignore_index=True)

        self._validate_no_tile_leakage(train_df, val_df, test_df)

        summary = self._build_summary(train_df, val_df, test_df)
        return SplitArtifacts(
            train=train_df.to_dict("records"),
            val=val_df.to_dict("records"),
            test=test_df.to_dict("records"),
            summary=summary,
        )

    def _split_single_terrain(
        self,
        terrain_df: pd.DataFrame,
        terrain: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        tile_sizes = (
            terrain_df.groupby("_tile_group")
            .size()
            .astype(int)
            .to_dict()
        )
        tiles = list(tile_sizes.keys())

        if len(tiles) < 3:
            raise ValueError(
                f"Terrain '{terrain}' has only {len(tiles)} unique tiles. "
                "At least 3 are required for train/val/test splitting."
            )

        train_tiles, val_tiles, test_tiles = self._assign_tiles_greedy(
            tile_sizes=tile_sizes,
            terrain=terrain,
        )

        train_df = terrain_df[terrain_df["_tile_group"].isin(train_tiles)].copy()
        val_df = terrain_df[terrain_df["_tile_group"].isin(val_tiles)].copy()
        test_df = terrain_df[terrain_df["_tile_group"].isin(test_tiles)].copy()

        if val_df.empty or test_df.empty:
            raise ValueError(
                f"Terrain '{terrain}' produced an empty validation or test split. "
                "Try different ratios or inspect tile distribution."
            )

        return train_df, val_df, test_df

    def _assign_tiles_greedy(
        self,
        tile_sizes: Dict[str, int],
        terrain: str,
    ) -> Tuple[List[str], List[str], List[str]]:
        tiles = sorted(tile_sizes.keys())
        total = sum(tile_sizes.values())
        targets = {
            "train": total * self.train_ratio,
            "val": total * self.val_ratio,
            "test": total * self.test_ratio,
        }

        best_assignments: Tuple[List[str], List[str], List[str]] | None = None
        best_score = float("inf")
        num_tiles = len(tiles)
        split_labels = ("train", "val", "test")

        for mask in range(3**num_tiles):
            buckets: Dict[str, List[str]] = {name: [] for name in split_labels}
            temp = mask

            for tile in tiles:
                bucket_idx = temp % 3
                temp //= 3
                buckets[split_labels[bucket_idx]].append(tile)

            if any(not buckets[name] for name in split_labels):
                continue

            counts = {
                name: sum(tile_sizes[tile] for tile in bucket)
                for name, bucket in buckets.items()
            }
            score = sum(abs(counts[name] - targets[name]) for name in split_labels)

            if score < best_score:
                best_score = score
                best_assignments = (
                    buckets["train"],
                    buckets["val"],
                    buckets["test"],
                )

        if best_assignments is None:
            raise ValueError(f"Could not find a valid tile split for terrain '{terrain}'.")

        return best_assignments

    def _validate_no_tile_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        split_tiles = {
            "train": set(train_df["terrain"].astype(str) + "::" + train_df["tile"].astype(str)),
            "val": set(val_df["terrain"].astype(str) + "::" + val_df["tile"].astype(str)),
            "test": set(test_df["terrain"].astype(str) + "::" + test_df["tile"].astype(str)),
        }

        overlap_checks = (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        )
        for left, right in overlap_checks:
            overlap = split_tiles[left].intersection(split_tiles[right])
            if overlap:
                raise ValueError(
                    f"Tile leakage detected between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )

    def _build_summary(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict:
        split_frames = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }
        total = sum(len(frame) for frame in split_frames.values())

        summary = {
            "metadata_path": str(self.metadata_path),
            "ratios": {
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            "counts": {},
            "percentages": {},
            "unique_tiles": {},
            "terrain_counts": {},
        }

        for split_name, frame in split_frames.items():
            summary["counts"][split_name] = int(len(frame))
            summary["percentages"][split_name] = round(len(frame) / total, 4)
            tile_keys = frame["terrain"].astype(str) + "::" + frame["tile"].astype(str)
            summary["unique_tiles"][split_name] = int(tile_keys.nunique())
            summary["terrain_counts"][split_name] = {
                str(k): int(v)
                for k, v in frame.groupby("terrain").size().sort_index().to_dict().items()
            }

        return summary

    def save(self, artifacts: SplitArtifacts, output_dir: str | Path = "splits") -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        payloads = {
            "train.json": artifacts.train,
            "val.json": artifacts.val,
            "test.json": artifacts.test,
            "summary.json": artifacts.summary,
        }
        for filename, payload in payloads.items():
            with (output_path / filename).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

        self._print_summary(artifacts.summary)

    def _print_summary(self, summary: Dict) -> None:
        print("\nDataset Split Summary")
        print("-" * 40)
        for split_name in ("train", "val", "test"):
            count = summary["counts"][split_name]
            pct = 100 * summary["percentages"][split_name]
            tiles = summary["unique_tiles"][split_name]
            print(f"{split_name.title():5} : {count:5d} images | {pct:6.2f}% | {tiles:2d} tiles")

        print("\nPer-terrain image counts")
        print("-" * 40)
        for split_name in ("train", "val", "test"):
            terrain_counts = summary["terrain_counts"][split_name]
            print(f"{split_name.title():5} : {terrain_counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create tile-aware train/val/test splits for SAR2EO."
    )
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splitter = DatasetSplitter(
        metadata_path=args.metadata_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    artifacts = splitter.split()
    splitter.save(artifacts, output_dir=args.output_dir)


if __name__ == "__main__":
    main()