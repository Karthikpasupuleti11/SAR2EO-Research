from __future__ import annotations

from typing import List

import timm
import torch
import torch.nn as nn


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 encoder with multi-scale skip features for SAR input."""

    def __init__(
        self,
        in_channels: int = 1,
        pretrained: bool = True,
        model_name: str = "efficientnet_b0",
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(1, 2, 3, 4, 5),
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
    print(f"Input        : {tuple(x.shape)}")
    stage_names = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Bottleneck"]
    for name, feat, channels, reduction in zip(
        stage_names, features, model.out_channels, model.reductions
    ):
        spatial = feat.shape[-1]
        print(
            f"{name:12} : {tuple(feat.shape)}  "
            f"(ch={channels}, stride={reduction}, spatial={spatial}x{spatial})"
        )


if __name__ == "__main__":
    _smoke_test()
