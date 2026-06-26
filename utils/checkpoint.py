from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from models.generator import SAR2EOGenerator


def load_checkpoint(weights_path: str | Path, device: torch.device) -> Dict[str, Any]:
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=device, weights_only=False)


def build_generator_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> Tuple[SAR2EOGenerator, Dict[str, Any]]:
    config = checkpoint.get("config")
    if config is None:
        raise ValueError("Checkpoint is missing 'config'. Cannot rebuild the generator.")

    model_cfg = config.get("model", {})
    generator = SAR2EOGenerator(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        pretrained_encoder=False,
        encoder_name=str(model_cfg.get("encoder_name", "efficientnet_b0")),
        use_cbam=bool(model_cfg.get("use_cbam", False)),
        use_residual=bool(model_cfg.get("use_residual", True)),
    )

    state_dict = checkpoint.get("generator")
    if state_dict is None:
        raise ValueError("Checkpoint is missing 'generator' state dict.")

    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval()
    return generator, config
