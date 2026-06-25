from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class DecoderUpBlock(nn.Module):
    """Bilinear resize-conv upsampling with skip fusion and optional residual block."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        fuse_in = out_channels + skip_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual = ResidualBlock(out_channels) if use_residual else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.reduce(x)

        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.fuse(x)
        return self.residual(x)


class ResidualDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        out_channels: int = 3,
        decoder_channels: List[int] | None = None,
        use_residual: bool = True,
    ) -> None:
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64]

        skip_channels = encoder_channels[:-1][::-1]
        in_channels = encoder_channels[-1]

        self.up_blocks = nn.ModuleList()
        for skip_ch, out_ch in zip(skip_channels, decoder_channels):
            self.up_blocks.append(
                DecoderUpBlock(
                    in_channels=in_channels,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    use_residual=use_residual,
                )
            )
            in_channels = out_ch

        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        skips = features[:-1]
        x = features[-1]

        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)

        return self.final_up(x)

    def forward_with_trace(self, features: List[torch.Tensor]) -> torch.Tensor:
        skips = features[:-1]
        x = features[-1]
        print(f"Bottleneck in : {tuple(x.shape)}")

        for idx, (block, skip) in enumerate(zip(self.up_blocks, reversed(skips)), start=1):
            print(f"  Skip {idx}     : {tuple(skip.shape)}")
            x = block(x, skip)
            print(f"  After up {idx} : {tuple(x.shape)}")

        out = self.final_up(x)
        print(f"Final output  : {tuple(out.shape)}")
        return out


def _smoke_test() -> None:
    from .encoder import EfficientNetEncoder

    encoder = EfficientNetEncoder(in_channels=1, pretrained=False)
    x = torch.randn(1, 1, 256, 256)
    features = encoder(x)

    decoder = ResidualDecoder(encoder_channels=encoder.out_channels, out_channels=3)
    decoder.forward_with_trace(features)


if __name__ == "__main__":
    _smoke_test()
