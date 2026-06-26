from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import evaluate_checkpoint


EXPERIMENT_WEIGHTS = {
    "E1": "Results/E1_results/E1_baseline.pt",
    "E2": "Results/E2_results/E2.pt",
    "E3": "Results/E3_results/E3.pt",
    "E4": "Results/E4_results/E4.pt",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all ablation checkpoints.")
    parser.add_argument("--output_dir", type=str, default="outputs/evaluation")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--experiments", nargs="+", default=["E1", "E2", "E3", "E4"])
    args = parser.parse_args()

    summary = {}
    for name in args.experiments:
        weights = Path(EXPERIMENT_WEIGHTS[name])
        if not weights.exists():
            print(f"Skipping {name}: missing {weights}")
            continue

        print(f"\n{'=' * 60}\nRunning evaluation for {name}\n{'=' * 60}")
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
