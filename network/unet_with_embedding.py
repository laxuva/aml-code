from typing import List

import torch
import math


class TimeEmbeddingLayer(torch.nn.Module):
    def __init__(self, input_dim, out_dim):
        super(TimeEmbeddingLayer, self).__init__()

        self.layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, out_dim),
            torch.nn.ReLU()
        )

    def forward(self, t):
        return self.layer.forward(t)[(...,) + (None,) * 2]


class DownSamplingBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            time_emb_dim=32,
            use_pooling: bool = False,
            return_just_downsampled_result: bool = False
    ):
        super(DownSamplingBlock, self).__init__()

        self.time_embedding_layer = TimeEmbeddingLayer(time_emb_dim, out_channels)

        self.return_just_downsampled_result = return_just_downsampled_result

        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )

        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )

        if use_pooling:
            self.down_sampling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.down_sampling_layer = torch.nn.Sequential(
                torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU()
            )

    def forward(self, x, t):
        time_embedding = self.time_embedding_layer.forward(t)

        x_hat = self.conv_1.forward(x)
        x_hat = self.conv_2.forward(x_hat * time_embedding)

        if self.return_just_downsampled_result:
            return self.down_sampling_layer.forward(x_hat)

        return x_hat, self.down_sampling_layer.forward(x_hat)


class UpSamplingBlock(torch.nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, time_emb_dim=32, use_transpose_conv: bool = True):
        super(UpSamplingBlock, self).__init__()

        self.time_embedding_layer = TimeEmbeddingLayer(time_emb_dim, out_channels)

        if not use_transpose_conv:
            self.up_conv = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2),
                torch.nn.ZeroPad2d((0, 1, 0, 1)),
                torch.nn.Conv2d(in_channels, in_channels // 2, kernel_size=2, stride=1, padding=0),
                torch.nn.BatchNorm2d(in_channels // 2),
                torch.nn.ReLU()
            )
        else:
            self.up_conv = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2, padding=0),
                torch.nn.BatchNorm2d(in_channels // 2),
                torch.nn.ReLU()
            )

        self.forward_conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )

        self.forward_conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )

    def forward(self, x, x_skip, t):
        time_embedding = self.time_embedding_layer.forward(t)

        x_hat = self.up_conv.forward(x)

        x_hat = self.forward_conv_1.forward(torch.cat((x_skip, x_hat), dim=1))

        return self.forward_conv_2.forward(x_hat * time_embedding)


class SinusoidalPositionEmbeddings(torch.nn.Module):
    """"
    Adapted from: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=uuckjpW_k1LN
    Video: https://www.youtube.com/watch?v=a4Yfz2FxXiY
    """
    def __init__(self, dim):
        super().__init__()

        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            channels_per_depth: List[int],
            final_out_channels: int,
            time_emb_dim: int = 32,
            output_activation_function: str = "Sigmoid"
    ):
        super(UNet, self).__init__()

        # Time embedding
        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            torch.nn.Linear(time_emb_dim, time_emb_dim),
            torch.nn.ReLU()
        )

        self.downsampling_layers = list()
        self.downsampling_layers.append(DownSamplingBlock(in_channels, channels_per_depth[0], time_emb_dim))
        for current_channels_in, current_channels_out in zip(channels_per_depth[:-2], channels_per_depth[1:-1]):
            self.downsampling_layers.append(DownSamplingBlock(current_channels_in, current_channels_out, time_emb_dim))
        self.downsampling_layers = torch.nn.Sequential(*self.downsampling_layers)

        self.intermediate_conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels_per_depth[-2], out_channels=channels_per_depth[-1], kernel_size=3,
                            padding=1, stride=1),
            torch.nn.BatchNorm2d(channels_per_depth[-1]),
            torch.nn.ReLU()
        )

        self.intermediate_time_embedding_layer = TimeEmbeddingLayer(time_emb_dim, channels_per_depth[-1])

        self.intermediate_conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels_per_depth[-1], out_channels=channels_per_depth[-1], kernel_size=3,
                            padding=1, stride=1),
            torch.nn.BatchNorm2d(channels_per_depth[-1]),
            torch.nn.ReLU()
        )

        channels_per_depth.reverse()

        self.upsampling_layers = list()
        for current_channels_in, current_channels_out in zip(channels_per_depth[0:-1], channels_per_depth[1:]):
            self.upsampling_layers.append(
                UpSamplingBlock(current_channels_in, current_channels_out, current_channels_out, time_emb_dim)
            )
        self.upsampling_layers = torch.nn.Sequential(*self.upsampling_layers)

        self.final_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=channels_per_depth[-1], out_channels=final_out_channels, kernel_size=1))
            # torch.nn.Sigmoid() if output_activation_function == "Sigmoid" else torch.nn.Tanh())

    def forward(self, x, t):
        skip_outputs = list()

        time_embedding = self.time_mlp(t)

        for layer in self.downsampling_layers:
            x_skip, x = layer.forward(x, time_embedding)
            skip_outputs.append(x_skip)

        x = self.intermediate_conv_1.forward(x)
        x = self.intermediate_conv_2.forward(x * self.intermediate_time_embedding_layer.forward(time_embedding))

        skip_outputs.reverse()

        for layer, skip_input in zip(self.upsampling_layers, skip_outputs):
            x = layer.forward(x, skip_input, time_embedding)

        return self.final_layer.forward(x)
