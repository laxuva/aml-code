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
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # TODO
        self.down1 = DownSamplingBlock(in_channels, 8, use_pooling=False)
        self.down2 = DownSamplingBlock(8, 16, use_pooling=True)

        self.up1 = UpSamplingBlock(16, 16, 8, use_transpose_conv=True)
        self.up2 = UpSamplingBlock(8, 8, 1, use_transpose_conv=False)

    def forward(self, x):
        # TODO
        x_1, x_1_down = self.down1(x)
        x_2, x_2_down = self.down2(x_1_down)
        x_3 = self.up1(x_2_down, x_2)
        x_4 = self.up2(x_3, x_1)
        return x_4


if __name__ == '__main__':
    from pathlib import Path
    from PIL import Image
    from torchvision.transforms import ToTensor

    img = Image.open(
        Path("~\\Documents\\data\\aml\\img_facemask_downsampled\\00000_Mask.png").expanduser()
    )

    net = UNet(3, 1)
    net.forward(ToTensor()(img)[None, :])

