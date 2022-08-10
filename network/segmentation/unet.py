from typing import List

import torch


class DownSamplingBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, use_pooling: bool = False):
        super(DownSamplingBlock, self).__init__()

        self.forward_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_features, out_features, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU()
        )

        if use_pooling:
            self.down_sampling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.down_sampling_layer = torch.nn.Sequential(
                torch.nn.Conv2d(out_features, out_features, kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU()
            )

    def forward(self, x):
        x_hat = self.forward_layers.forward(x)
        return x_hat, self.down_sampling_layer.forward(x_hat)


class UpSamplingBlock(torch.nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_transpose_conv: bool = True):
        super(UpSamplingBlock, self).__init__()

        if not use_transpose_conv:
            self.up_conv = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2),
                torch.nn.ZeroPad2d((0, 1, 0, 1)),
                torch.nn.Conv2d(in_channels, in_channels // 2, kernel_size=2, stride=1, padding=0),
                torch.nn.ReLU()
            )
        else:
            self.up_conv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, padding=0),
                torch.nn.ReLU()
            )

        self.forward_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU()
        )

    def forward(self, x, x_skip):
        x_hat = self.up_conv.forward(x)
        return self.forward_layers.forward(torch.cat((x_skip, x_hat), dim=1))


class UNet(torch.nn.Module):
    def __init__(self, in_channels: int, channels_per_depth: List[int],  final_out_channels: int):
        super(UNet, self).__init__()

        self.downsampling_layers = list()
        self.downsampling_layers.append(DownSamplingBlock(in_channels, channels_per_depth[0]))
        for current_channels_in, current_channels_out in zip(channels_per_depth[:-2], channels_per_depth[1:-1]):
            self.downsampling_layers.append(DownSamplingBlock(current_channels_in, current_channels_out))
        self.downsampling_layers = torch.nn.Sequential(*self.downsampling_layers)

        self.intermediate_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels_per_depth[-2], out_channels=channels_per_depth[-1], kernel_size=3,
                            padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=channels_per_depth[-1], out_channels=channels_per_depth[-1], kernel_size=3,
                            padding=1, stride=1),
            torch.nn.ReLU()
        )

        channels_per_depth.reverse()

        self.upsampling_layers = list()
        for current_channels_in, current_channels_out in zip(channels_per_depth[0:-1], channels_per_depth[1:]):
            self.upsampling_layers.append(UpSamplingBlock(current_channels_in, current_channels_out, current_channels_out))
        self.upsampling_layers = torch.nn.Sequential(*self.upsampling_layers)

        self.final_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels_per_depth[-1], out_channels=final_out_channels, kernel_size=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        skip_outputs = list()

        for layer in self.downsampling_layers:
            x_skip, x = layer.forward(x)
            skip_outputs.append(x_skip)

        x = self.intermediate_layers.forward(x)

        skip_outputs.reverse()

        for layer, skip_input in zip(self.upsampling_layers, skip_outputs):
            x = layer.forward(x, skip_input)

        return self.final_layer.forward(x)


if __name__ == '__main__':
    from pathlib import Path
    from PIL import Image
    from torchvision.transforms import ToTensor

    img = Image.open(
        Path("~\\Documents\\data\\aml\\masked128png\\00000_Mask.png").expanduser()
    )

    net = UNet(3, [16, 32, 64], 1)
    print(net)
    net.forward(ToTensor()(img)[None, :])

