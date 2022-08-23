""""
Adapted from: https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=uuckjpW_k1LN
Video: https://www.youtube.com/watch?v=a4Yfz2FxXiY
"""

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

from data.diffusion_model_dataset import DiffusionModelDataset
from utils.config_parser import ConfigParser
from network.segmentation.unet_with_embedding import UNet


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

def get_betas():
    return betas

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

IMG_SIZE = 128
BATCH_SIZE = 32

BASE_DATA_PATH = Path("~/Documents/data/aml/")
PRELOAD_PERCENTAGE = 0.25


def load_transformed_dataset(device="cpu"):
    train = DiffusionModelDataset.load_from_label_file(
        str(BASE_DATA_PATH.joinpath("train_dataset_small.json")),
        image_path=str(BASE_DATA_PATH.joinpath("original128png")),
        preload_percentage=PRELOAD_PERCENTAGE,
        device=torch.device(device),
        transforms=[
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ]
    )
    test = DiffusionModelDataset.load_from_label_file(
        str(BASE_DATA_PATH.joinpath("val_dataset_small.json")),
        image_path=str(BASE_DATA_PATH.joinpath("original128png")),
        preload_percentage=PRELOAD_PERCENTAGE,
        device=torch.device(device),
        transforms=[
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ]
    )

    return torch.utils.data.ConcatDataset([train, test])


def show_tensor_image(image, show=True):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    if show:
        plt.imshow(reverse_transforms(image))
    else:
        return reverse_transforms(image)


def get_loss(model, x_0, t, device="cpu"):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None,) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
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


@torch.no_grad()
def sample_timestep(x, t, model):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, device="cpu", epoch=None):
    # Sample noise
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, model)
        if i % stepsize == 0:
            plt.subplot(1, num_images, i // stepsize + 1)
            plt.title(f"Step: {i}")
            show_tensor_image(img.detach().cpu())
    if epoch is not None:
        plt.suptitle(f"Epoch: {epoch}")
    plt.show()


def main(train=True, modelpath="./model.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = ConfigParser.read("../../configs/debugging_diffusion_model.yaml")
    model = UNet(**config["model"]["params"])
    model.to(device)

    if train:
        data = load_transformed_dataset(device)
        dataloader = DataLoader(data,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                drop_last=True)
        optimizer = Adam(model.parameters(), lr=0.001)
        epochs = 250  # Try more!
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

        for epoch in range(epochs):
            for step, batch in tqdm.tqdm(enumerate(dataloader)):
                optimizer.zero_grad()

                t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
                loss = get_loss(model, batch[0], t, device)
                loss.backward()
                optimizer.step()

                if step == 0:
                    print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
                    sample_plot_image(model, device, epoch)

            lr_scheduler.step()
            torch.save(model.state_dict(), Path(".").joinpath("model.pt"))
    else:
        model.load_state_dict(torch.load(Path(modelpath), map_location=device))
        # model.eval()

        sample_plot_image(model, device)


if __name__ == '__main__':
    main()
