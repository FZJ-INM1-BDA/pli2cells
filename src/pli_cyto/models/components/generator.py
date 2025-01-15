# Implementation following https://www.pyimagesearch.com/2021/11/08/u-net-training-image-segmentation-models-in-pytorch/

import torch
import torch.nn as nn
from torchvision.transforms import functional as T
from torch.nn import functional as F


class Block(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, dropout=0.0, norm=nn.BatchNorm2d) -> None:
        super().__init__(
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, out_channels, 3),
            norm(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3),
            norm(out_channels),
            nn.LeakyReLU(),
        )


class BlockSmall(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, dropout=0.0, norm=nn.BatchNorm2d) -> None:
        super().__init__(
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, out_channels, 3),
            norm(out_channels),
            nn.LeakyReLU(),
        )


class Encoder(nn.Module):

    def __init__(self, channels=(3, 64, 128, 256, 512, 1024), block=Block, dropout=0.0, norm=nn.BatchNorm2d):
        super().__init__()
        self.channels = channels
        self.enc_blocks = nn.ModuleList([
            block(channels[i], channels[i + 1], dropout=dropout, norm=norm) for i in range(len(channels) - 1)
        ])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feature_list = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            feature_list.append(x)
            if (i + 1) < len(self.enc_blocks):
                x = self.pool(x)
        return feature_list


class Decoder(nn.Module):

    def __init__(
            self,
            channels=(1024, 512, 256, 128, 64),
            enc_channels=(64, 128, 256, 512, 1024),
            block=Block,
            dropout=0.0,
            norm=nn.BatchNorm2d
    ):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2) for i in range(len(channels) - 1)
        ])
        self.dec_blocks = nn.ModuleList([
            block(channels[i + 1] + enc_channels[-i - 2], channels[i + 1], dropout=dropout, norm=norm) for i in range(len(channels) - 1)
        ])

    def forward(self, x, feature_list):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            features = Decoder.crop(feature_list[i], x)
            x = torch.cat([x, features], dim=1)
            x = self.dec_blocks[i](x)

        return x

    @staticmethod
    def crop(features, x):
        features = T.center_crop(features, x.shape[-2:])
        return features


class PLIEncoder(nn.Module):

    def __init__(
        self,
        enc_channels=(64, 128, 256, 512),
        block=Block,
        dropout=0.0,
        norm=nn.BatchNorm2d
    ):
        super(PLIEncoder, self).__init__()
        self.input_layer = pli.nn.layers.pli.PLIMods()
        self.norm_layer = pli.nn.layers.RunningNorm2D(3)
        self.encoder = Encoder((3, *enc_channels), block=block, dropout=dropout, norm=norm)

    def forward(self, trans, dir, ret):
        x = self.input_layer(trans, dir, ret)
        x = self.norm_layer(x)
        x = self.encoder(x)
        return x


class PLIUnet(nn.Module):

    def __init__(
        self,
        enc_channels=(64, 128, 256, 512),
        dec_channels=(512, 256, 128, 64),
        n_classes=3,
        block=Block,
        dropout=0.0,
        retain_dim=True,
        activation=None,
    ):
        super(PLIUnet, self).__init__()
        self.encoder = PLIEncoder(enc_channels, block=block, dropout=dropout)
        self.decoder = Decoder(dec_channels, enc_channels, block=block, dropout=dropout)
        self.head = nn.Conv2d(dec_channels[-1], n_classes, 1)
        self.retain_dim = retain_dim
        if activation:
            if activation.lower() == 'sigmoid':
                self.activation = F.sigmoid
        else:
            self.activation = lambda x: x

    def forward(self, trans, dir, ret):
        enc_features = self.encoder(trans, dir, ret)
        dec_features = self.decoder(enc_features[::-1][0],
                                    enc_features[::-1][1:])
        map = self.activation(self.head(dec_features))

        return map
