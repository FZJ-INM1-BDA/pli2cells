import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(
            self,
            feature_maps: int,
            image_channels: int,
            depth: int = 5,
            activation: str = "Sigmoid",
            dropout: float = 0.0,
    ):
        super().__init__()

        assert depth >= 2

        self.activation = getattr(nn, activation)()

        modules = [self._make_disc_block(image_channels, feature_maps, batch_norm=False, dropout=dropout)]
        for i in range(depth - 2):
            modules.append(
                self._make_disc_block(feature_maps * 2 ** i, feature_maps * 2 ** (i + 1), dropout=dropout)
            )
        modules.append(
            self._make_disc_block(feature_maps * 2 ** (depth - 2), 1, kernel_size=3,
                                  stride=1, padding=0, last_block=True)
        )

        self.disc = nn.Sequential(*modules)

    def _make_disc_block(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            bias: bool = False,
            batch_norm: bool = True,
            dropout: float = 0.0,
            last_block: bool = False,
    ) -> nn.Sequential:
        if not last_block:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            )
        else:
            disc_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
                self.activation,
            )

        return disc_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.disc(x)

        return out