from __future__ import annotations

from typing import List, Tuple

import timm
import torch
import torch.nn as nn


def resolve_out_indices(model_name: str, num_levels: int = 5) -> Tuple[int, ...]:
    """
    Return valid timm feature indices for the last `num_levels` encoder stages.

    EfficientNet-B0 exposes indices 0-4 in current timm builds; older configs used
    (1, 2, 3, 4, 5) which raises IndexError on Kaggle.
    """
    probe = timm.create_model(model_name, pretrained=False, features_only=True)
    total = len(probe.feature_info)
    del probe

    if total < 1:
        raise ValueError(f"No feature levels found for model: {model_name}")

    count = min(num_levels, total)
    start = total - count
    return tuple(range(start, total))


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder with multi-scale skip features for SAR input."""

    def __init__(
        self,
        in_channels: int = 1,
        pretrained: bool = True,
        model_name: str = "efficientnet_b0",
        num_feature_levels: int = 5,
    ) -> None:
        super().__init__()

        self.out_indices = resolve_out_indices(model_name, num_levels=num_feature_levels)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=self.out_indices,
        )

        self.out_channels = list(self.backbone.feature_info.channels())
        self.reductions = list(self.backbone.feature_info.reduction())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.backbone(x)


def _smoke_test() -> None:
    model = EfficientNetEncoder(in_channels=1, pretrained=False)
    x = torch.randn(1, 1, 256, 256)
    features = model(x)

    print("Encoder smoke test")
    print("-" * 50)
    print(f"out_indices  : {model.out_indices}")
    print(f"Input        : {tuple(x.shape)}")
    for idx, (feat, channels, reduction) in enumerate(
        zip(features, model.out_channels, model.reductions)
    ):
        name = "Bottleneck" if idx == len(features) - 1 else f"Stage {idx + 1}"
        spatial = feat.shape[-1]
        print(
            f"{name:12} : {tuple(feat.shape)}  "
            f"(ch={channels}, stride={reduction}, spatial={spatial}x{spatial})"
        )


if __name__ == "__main__":
    _smoke_test()
