from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from network.unet import UNet
from utils.config_parser import ConfigParser


@torch.no_grad()
def test_prediction(model_path, image_path, seg_map_path, config_file="../configs/autoencoder.yaml"):
    config = ConfigParser.read(config_file)
    image_path = Path(image_path).expanduser()
    seg_map_path = Path(seg_map_path).expanduser()

    to_tensor: Callable = ToTensor()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UNet(**config["model"]["params"])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y = to_tensor(Image.open(image_path)).to(device)
    seg_map = to_tensor(Image.open(seg_map_path)).to(device)

    x = y.clone()
    x[:, seg_map[0] != 0] = 0

    # ToPILImage()(x).save("./masked.png")
    # ToPILImage()(seg_map).save("./mask.png")

    x = torch.cat((x, seg_map))

    y_pred = model.forward(x[None, :])[0].cpu().detach()

    ToPILImage()(y_pred).save("./test.png")

    x = np.transpose(x.cpu().detach().numpy(), (1, 2, 0))
    y_pred = np.transpose(y_pred.numpy(), (1, 2, 0))
    y = np.transpose(y.cpu().detach().numpy(), (1, 2, 0))

    plt.imshow(y)
    plt.show()

    plt.imshow(y_pred)
    plt.show()

    y_pred[x[:, :, 3] == 0] = 0
    plt.imshow(y_pred)
    plt.show()

    plt.imshow(y_pred + x[:, :, 0:3])
    plt.show()

    # Image.fromarray(((y_pred + x[:, :, 0:3]) * 255).astype(np.uint8)).save("final_image.png")

    print(f"MAE: {np.mean(np.abs(y_pred - y))}")


if __name__ == '__main__':
    test_prediction(
        model_path="../evaluation/autoencoder/best_model.pt",
        image_path="~/Documents/data/aml/original128png/00018.png",  # 00186 00048 00018 45844 00375 00019
        seg_map_path="~/Documents/data/aml/seg_mask128png/00018.png"  # 00071 00102 00112 00116 00043
    )
