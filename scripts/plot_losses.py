from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


DEFAULT_HISTORY_FILES = {
    "E1": Path("Results/E1_results/E1_history.json"),
    "E2": Path("Results/E2_results/E2_history.json"),
    "E3": Path("Results/E3_results/E3_history.json"),
    "E4": Path("Results/E4_results/E4_history.json"),
}


def load_history(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_single_experiment(name: str, history: List[Dict], output_dir: Path) -> None:
    epochs = [row["epoch"] for row in history]
    output_dir.mkdir(parents=True, exist_ok=True)

    if history[0].get("loss_d", 0.0) > 0 or any(row.get("loss_gan", 0.0) > 0 for row in history):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        axes[0].plot(epochs, [row["loss_g"] for row in history], label="Generator")
        axes[0].plot(epochs, [row["loss_l1"] for row in history], label="L1")
        axes[0].set_title(f"{name} Generator + L1")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()

        axes[1].plot(epochs, [row["loss_d"] for row in history], color="tab:orange")
        axes[1].set_title(f"{name} Discriminator")
        axes[1].set_xlabel("Epoch")

        axes[2].plot(epochs, [row.get("loss_gan", 0.0) for row in history], color="tab:green")
        axes[2].set_title(f"{name} GAN")
        axes[2].set_xlabel("Epoch")
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [row["loss_g"] for row in history], label="Loss_G")
        ax.plot(epochs, [row["loss_l1"] for row in history], label="L1")
        ax.set_title(f"{name} Training Loss")
        ax.set_xlabel("Epoch")
        ax.legend()

    if any(row.get("loss_perceptual", 0.0) > 0 for row in history):
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(epochs, [row.get("loss_perceptual", 0.0) for row in history], label="Perceptual")
        ax2.plot(epochs, [row.get("loss_ssim", 0.0) for row in history], label="SSIM")
        ax2.set_title(f"{name} Auxiliary Losses")
        ax2.set_xlabel("Epoch")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(output_dir / f"{name}_aux_loss.png", dpi=150)
        plt.close(fig2)

    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_loss_curve.png", dpi=150)
    plt.close(fig)


def plot_ablation_l1(histories: Dict[str, List[Dict]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["loss_l1"] for row in history], label=name)

    ax.set_title("Ablation: L1 Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "ablation_l1_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss curves from history JSON files.")
    parser.add_argument("--output_dir", type=str, default="outputs/loss_curves")
    parser.add_argument(
        "--history",
        nargs="*",
        default=[],
        help="Optional custom history files as NAME=PATH",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    histories: Dict[str, List[Dict]] = {}

    if args.history:
        for item in args.history:
            name, path = item.split("=", maxsplit=1)
            histories[name] = load_history(Path(path))
    else:
        for name, path in DEFAULT_HISTORY_FILES.items():
            if path.exists():
                histories[name] = load_history(path)

    if not histories:
        raise FileNotFoundError("No history JSON files found under Results/.")

    for name, history in histories.items():
        plot_single_experiment(name, history, output_dir)

    plot_ablation_l1(histories, output_dir)
    print(f"Saved loss curves to {output_dir}")


if __name__ == "__main__":
    main()
