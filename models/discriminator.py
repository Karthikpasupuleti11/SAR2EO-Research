from __future__ import annotations

import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (Pix2Pix-style)."""

    def __init__(
        self,
        in_channels: int = 4,
        features: int = 64,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        in_filters = features
        for layer_idx in range(1, n_layers):
            out_filters = min(in_filters * 2, 512)
            layers.extend(
                [
                    nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_filters),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            in_filters = out_filters

        out_filters = min(in_filters * 2, 512)
        layers.extend(
            [
                nn.Conv2d(in_filters, out_filters, 4, 1, 1, bias=False),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_filters, 1, 4, 1, 1),
            ]
        )

        self.model = nn.Sequential(*layers)

    def forward(self, sar: torch.Tensor, optical: torch.Tensor) -> torch.Tensor:
        return self.model(torch.cat([sar, optical], dim=1))


def _smoke_test() -> None:
    model = PatchDiscriminator(in_channels=4)
    sar = torch.randn(2, 1, 256, 256)
    optical = torch.randn(2, 3, 256, 256)
    out = model(sar, optical)
    print("Discriminator smoke test")
    print("-" * 40)
    print(f"SAR    : {tuple(sar.shape)}")
    print(f"Optical: {tuple(optical.shape)}")
    print(f"Output : {tuple(out.shape)}")


if __name__ == "__main__":
    _smoke_test()
