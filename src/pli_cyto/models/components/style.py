import torch.nn as nn
import torchvision
from torchvision.models import vgg19

import segmentation_models_pytorch as smp


class VGGStyle(nn.Module):
    def __init__(
            self,
            name: str = 'vgg19',
            weights: str = 'imagenet',
            depth: int = 4,
            in_channels: int = 1,
    ):
        super().__init__()

        # Define the Generator Network
        # self.encoder = vgg19(weights=torchvision.models.VGG19_Weights)
        self.encoder = smp.encoders.get_encoder(name, in_channels=in_channels, depth=depth, weights=weights)

    def forward(self, x):
        features = self.encoder(x)
        
        return features
