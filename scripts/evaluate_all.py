from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import evaluate_checkpoint


EXPERIMENT_FILES = {
    "E1": ("E1_results", "E1_baseline.pt"),
    "E2": ("E2_results", "E2.pt"),
    "E3": ("E3_results", "E3.pt"),
    "E4": ("E4_results", "E4.pt"),
}


def search_roots(extra: Optional[Path] = None) -> list[Path]:
    roots = [ROOT, ROOT.parent, Path.cwd(), Path.cwd().parent]
    if extra is not None:
        roots.insert(0, extra)
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_checkpoint(name: str, roots: Iterable[Path]) -> Optional[Path]:
    folder, filename = EXPERIMENT_FILES[name]
    candidates = [
        Path("Results") / folder / filename,
        Path(folder) / filename,
    ]
    for root in roots:
        for relative in candidates:
            path = (root / relative).resolve()
            if path.exists():
                return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all ablation checkpoints.")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--experiments", nargs="+", default=["E1", "E2", "E3", "E4"])
    parser.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Optional directory containing E*_results folders (e.g. /kaggle/working).",
    )
    args = parser.parse_args()

    extra_root = Path(args.results_root).resolve() if args.results_root else None
    roots = search_roots(extra_root)

    summary = {}
    for name in args.experiments:
        weights = resolve_checkpoint(name, roots)
        if weights is None:
            print(f"Skipping {name}: checkpoint not found under {', '.join(str(r) for r in roots)}")
            continue

        print(f"\n{'=' * 60}\nRunning evaluation for {name}\nWeights: {weights}\n{'=' * 60}")
        metrics = evaluate_checkpoint(
            weights_path=weights,
            splits=args.splits,
            output_dir=args.output_dir,
        )
        summary[name] = metrics

    summary_path = Path(args.output_dir) / "ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nSaved ablation summary: {summary_path}")


if __name__ == "__main__":
    main()
