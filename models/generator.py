from __future__ import annotations

import torch
import torch.nn as nn

from .cbam import CBAM
from .decoder import ResidualDecoder
from .encoder import EfficientNetEncoder


class SAR2EOGenerator(nn.Module):
    """
    SAR (1x256x256) -> EfficientNet-B0 encoder -> CBAM -> residual decoder -> RGB (3x256x256)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 3,
        pretrained_encoder: bool = True,
        encoder_name: str = "efficientnet_b0",
        use_cbam: bool = True,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = EfficientNetEncoder(
            in_channels=in_channels,
            pretrained=pretrained_encoder,
            model_name=encoder_name,
        )
        bottleneck_channels = self.encoder.out_channels[-1]

        self.use_cbam = use_cbam
        self.cbam = CBAM(bottleneck_channels) if use_cbam else nn.Identity()

        self.decoder = ResidualDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=out_channels,
            use_residual=use_residual,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        features[-1] = self.cbam(features[-1])
        return self.decoder(features)

    @torch.no_grad()
    def forward_with_trace(self, x: torch.Tensor) -> torch.Tensor:
        print("Generator shape trace")
        print("-" * 50)
        print(f"Input        : {tuple(x.shape)}")

        features = self.encoder(x)
        stage_names = ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Bottleneck"]
        for name, feat, channels in zip(stage_names, features, self.encoder.out_channels):
            print(f"{name:12} : {tuple(feat.shape)}  (ch={channels})")

        features[-1] = self.cbam(features[-1])
        if self.use_cbam:
            print(f"After CBAM   : {tuple(features[-1].shape)}")

        output = self.decoder.forward_with_trace(features)
        print(f"Output range : [{output.min().item():.3f}, {output.max().item():.3f}]")
        return output


def _smoke_test() -> None:
    model = SAR2EOGenerator(pretrained_encoder=False, use_cbam=True)
    x = torch.randn(1, 1, 256, 256)
    model.eval()
    model.forward_with_trace(x)
    print(f"Params       : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


if __name__ == "__main__":
    _smoke_test()
