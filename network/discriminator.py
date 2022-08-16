from typing import Tuple

import torch

from network.segmentation.unet import DownSamplingBlock


class Discriminator(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            input_image_size: Tuple[int, int] = (128, 128)
    ):
        super(Discriminator, self).__init__()

        self.conv_layers = torch.nn.Sequential(
            DownSamplingBlock(in_channels, 16, return_just_downsampled_result=True),
            DownSamplingBlock(16, 32, return_just_downsampled_result=True),
            DownSamplingBlock(32, 64, return_just_downsampled_result=True),
            DownSamplingBlock(64, 128, return_just_downsampled_result=True),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(128 * input_image_size[0] // 2**4 * input_image_size[1] // 2**4, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

    def clip_weights(self):
        def clip(m):
            if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
                for param in m.parameters():
                    param.clamp(-0.01, 0.01)

        self.apply(clip)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layers.forward(self.conv_layers.forward(x).flatten(start_dim=1))
