from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_KAGGLE_SRC = (
    "/kaggle/input/datasets/requiemonk/"
    "sentinel12-image-pairs-segregated-by-terrain/v_2"
)
DEFAULT_DST = Path("datasets/sentinel12/v_2")


def resolve_source(src: str | Path) -> Path:
    path = Path(src)
    if not path.exists():
        raise FileNotFoundError(
            f"Kaggle dataset not found at: {path}\n"
            "Attach the dataset in Notebook → Add Data, then rerun with the correct --src path."
        )

    if not (path / "agri" / "s1").exists():
        raise FileNotFoundError(
            f"Dataset found but missing expected agri/s1 folder under: {path}\n"
            "Check that --src points to the v_2 folder (contains agri, urban, etc.)."
        )
    return path


def link_dataset(src: Path, dst: Path = DEFAULT_DST) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.is_symlink() or dst.exists():
        if dst.resolve() == src.resolve():
            print(f"Dataset already linked: {dst} -> {src}")
            return dst
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_dir():
            print(f"WARNING: {dst} already exists as a directory. Skipping link.")
            return dst

    try:
        os.symlink(src, dst, target_is_directory=True)
        print(f"Linked: {dst} -> {src}")
    except OSError:
        import shutil

        print(f"Symlink failed, copying dataset (slow): {src} -> {dst}")
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Copied dataset to: {dst}")

    return dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link Kaggle input dataset to datasets/sentinel12/v_2 for SAR2EO."
    )
    parser.add_argument(
        "--src",
        type=str,
        default=DEFAULT_KAGGLE_SRC,
        help="Path to the v_2 dataset folder on Kaggle.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default=str(DEFAULT_DST),
        help="Local path expected by the repo config and split files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = resolve_source(args.src)
    dst = link_dataset(src, Path(args.dst))
    print("\nNext steps:")
    print(f"  python utils/build_metadata.py --dataset-root {dst}")
    print("  python data/split.py")
    print("  python -m models.generator")
    print("  python train.py --max-batches 5")


if __name__ == "__main__":
    main()
