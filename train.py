from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torchvision.utils as vutils
from torch import optim
from tqdm import tqdm

from data.dataloaders import build_dataloaders
from losses.losses import GeneratorLosses
from models.discriminator import PatchDiscriminator
from models.generator import SAR2EOGenerator
from utils.config import load_config
from utils.seed import set_seed


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def configure_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def build_models(
    config: Dict,
    device: torch.device,
    use_gan: bool,
) -> Tuple[SAR2EOGenerator, Optional[PatchDiscriminator]]:
    model_cfg = config.get("model", {})

    generator = SAR2EOGenerator(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        pretrained_encoder=bool(model_cfg.get("pretrained_encoder", True)),
        encoder_name=str(model_cfg.get("encoder_name", "efficientnet_b0")),
        use_cbam=bool(model_cfg.get("use_cbam", False)),
        use_residual=bool(model_cfg.get("use_residual", True)),
    ).to(device)

    discriminator = None
    if use_gan:
        discriminator = PatchDiscriminator(
            in_channels=int(model_cfg.get("in_channels", 1)) + int(model_cfg.get("out_channels", 3)),
            features=int(model_cfg.get("ndf", 64)),
        ).to(device)

    return generator, discriminator


def build_optimizers(generator, discriminator, config: Dict):
    opt_cfg = config.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 2e-4))
    beta1 = float(opt_cfg.get("beta1", 0.5))
    beta2 = float(opt_cfg.get("beta2", 0.999))

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = None
    if discriminator is not None:
        optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    return optimizer_g, optimizer_d


def print_loss_formula(losses: GeneratorLosses) -> None:
    terms = []
    if losses.use_gan:
        terms.append("adv")
    terms.append(f"{losses.lambda_l1:.0f} * L1")
    if losses.use_perceptual:
        terms.append(f"{losses.lambda_perceptual:.0f} * perceptual")
    if losses.use_ssim:
        terms.append(f"{losses.lambda_ssim:.0f} * (1 - SSIM)")
    print(f"Loss_G = {' + '.join(terms)}")


def save_sample_grid(
    sar: torch.Tensor,
    real_optical: torch.Tensor,
    fake_optical: torch.Tensor,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sar_vis = sar.repeat(1, 3, 1, 1)
    sar_vis = (sar_vis + 1.0) / 2.0
    real_vis = (real_optical + 1.0) / 2.0
    fake_vis = (fake_optical + 1.0) / 2.0

    nrow = min(4, sar.shape[0])
    grid = torch.cat([sar_vis[:nrow], real_vis[:nrow], fake_vis[:nrow]], dim=0)
    vutils.save_image(grid, output_path, nrow=nrow, normalize=False)


def train_epoch(
    generator,
    discriminator,
    loader,
    losses: GeneratorLosses,
    optimizer_g,
    optimizer_d,
    device: torch.device,
    epoch: int,
    use_amp: bool,
    scaler_g: torch.amp.GradScaler,
    scaler_d: Optional[torch.amp.GradScaler],
    max_batches: Optional[int] = None,
    sample_dir: Optional[Path] = None,
) -> Dict[str, float]:
    generator.train()
    if discriminator is not None:
        discriminator.train()

    totals = {
        "loss_g": 0.0,
        "loss_d": 0.0,
        "loss_l1": 0.0,
        "loss_gan": 0.0,
        "loss_perceptual": 0.0,
        "loss_ssim": 0.0,
    }
    num_batches = 0

    progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
    for batch_idx, batch in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        real_sar = batch["sar"].to(device, non_blocking=True)
        real_optical = batch["optical"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            fake_optical = generator(real_sar)

        loss_d = torch.zeros((), device=device)
        if losses.use_gan and discriminator is not None and optimizer_d is not None:
            optimizer_d.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_real = discriminator(real_sar, real_optical)
                pred_fake = discriminator(real_sar, fake_optical.detach())
                loss_d = losses.discriminator_loss(pred_real, pred_fake)
            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()

        optimizer_g.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            if losses.use_gan and discriminator is not None:
                pred_fake_for_g = discriminator(real_sar, fake_optical)
            else:
                pred_fake_for_g = torch.zeros(1, device=device)
            gen_losses = losses.generator_loss(pred_fake_for_g, fake_optical, real_optical)

        scaler_g.scale(gen_losses["loss_total"]).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()

        totals["loss_g"] += float(gen_losses["loss_total"].item())
        totals["loss_d"] += float(loss_d.item())
        totals["loss_l1"] += float(gen_losses["loss_l1"].item())
        totals["loss_gan"] += float(gen_losses["loss_gan"].item())
        totals["loss_perceptual"] += float(gen_losses["loss_perceptual"].item())
        totals["loss_ssim"] += float(gen_losses["loss_ssim"].item())
        num_batches += 1

        progress.set_postfix(
            loss_g=f"{gen_losses['loss_total'].item():.3f}",
            loss_d=f"{loss_d.item():.3f}",
        )

        if batch_idx == 0 and sample_dir is not None:
            save_sample_grid(
                real_sar.detach().float().cpu(),
                real_optical.detach().float().cpu(),
                fake_optical.detach().float().cpu(),
                sample_dir / f"epoch_{epoch:03d}_batch_000.png",
            )

    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def train(
    config_path: str = "configs/config.yaml",
    max_batches: Optional[int] = None,
    epochs_override: Optional[int] = None,
) -> None:
    config = load_config(config_path)
    set_seed(int(config.get("seed", 42)))

    device = get_device()
    configure_device(device)

    training_cfg = config.get("training", {})
    experiment_cfg = config.get("experiment", {})
    experiment_name = str(experiment_cfg.get("name", "E1"))
    use_gan = bool(experiment_cfg.get("use_gan", False))
    use_amp = bool(training_cfg.get("use_amp", True)) and device.type == "cuda"

    loaders = build_dataloaders(config)
    generator, discriminator = build_models(config, device, use_gan=use_gan)
    optimizer_g, optimizer_d = build_optimizers(generator, discriminator, config)

    losses = GeneratorLosses(
        lambda_l1=float(training_cfg.get("lambda_l1", 100.0)),
        lambda_perceptual=float(training_cfg.get("lambda_perceptual", 10.0)),
        lambda_ssim=float(training_cfg.get("lambda_ssim", 5.0)),
        use_gan=use_gan,
        use_perceptual=bool(experiment_cfg.get("use_perceptual", False)),
        use_ssim=bool(experiment_cfg.get("use_ssim", False)),
        device=device,
    ).to(device)

    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp) if use_gan else None

    checkpoint_dir = Path(training_cfg.get("checkpoint_dir", "checkpoints"))
    sample_dir = Path(training_cfg.get("sample_dir", "outputs/samples"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    epochs = epochs_override if epochs_override is not None else int(training_cfg.get("epochs", 1))
    history = []

    print(f"Device      : {device}")
    print(f"Experiment  : {experiment_name}")
    print(f"AMP         : {use_amp}")
    print(f"Use GAN     : {losses.use_gan}")
    print(f"Perceptual  : {losses.use_perceptual}")
    print(f"SSIM        : {losses.use_ssim}")
    print(f"CBAM        : {generator.use_cbam}")
    print(f"Epochs      : {epochs}")
    print_loss_formula(losses)
    if max_batches is not None:
        print(f"Max batches : {max_batches}")

    for epoch in range(1, epochs + 1):
        metrics = train_epoch(
            generator=generator,
            discriminator=discriminator,
            loader=loaders["train"],
            losses=losses,
            optimizer_g=optimizer_g,
            optimizer_d=optimizer_d,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
            scaler_g=scaler_g,
            scaler_d=scaler_d,
            max_batches=max_batches,
            sample_dir=sample_dir,
        )
        history.append({"epoch": epoch, **metrics})
        print(
            f"Epoch {epoch:03d} | "
            f"G: {metrics['loss_g']:.4f} | "
            f"D: {metrics['loss_d']:.4f} | "
            f"L1: {metrics['loss_l1']:.4f} | "
            f"GAN: {metrics['loss_gan']:.4f} | "
            f"Perc: {metrics['loss_perceptual']:.4f} | "
            f"SSIM: {metrics['loss_ssim']:.4f}"
        )

    ckpt_name = f"{experiment_name}_baseline.pt" if experiment_name == "E1" else f"{experiment_name}.pt"
    ckpt_path = checkpoint_dir / ckpt_name
    latest_path = checkpoint_dir / "latest.pt"

    payload = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict() if discriminator is not None else None,
        "optimizer_g": optimizer_g.state_dict(),
        "optimizer_d": optimizer_d.state_dict() if optimizer_d is not None else None,
        "config": config,
        "history": history,
    }
    torch.save(payload, ckpt_path)
    torch.save(payload, latest_path)

    history_path = checkpoint_dir / f"{experiment_name}_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved latest   : {latest_path}")
    print(f"Saved history  : {history_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAR2EO generator on Kaggle/local GPU.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit batches per epoch for quick smoke tests (E0).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epoch count from config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        config_path=args.config,
        max_batches=args.max_batches,
        epochs_override=args.epochs,
    )
