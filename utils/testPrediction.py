from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

from network.segmentation.unet import UNet
from train.unet_trainer import UNetTrainer
from utils.config_parser import ConfigParser


def test_prediction(
        model_path="../train/final_model.pt",
        image_path="D:/aml/localData/masked128png/00000_Mask.png"
        ):
    config = ConfigParser.read("../configs/debugging.yaml")
    image_path = Path(image_path).expanduser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        model = UNet(**config["model"])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
    except RuntimeError:
        model = UNetTrainer(config["model"], config["training"], device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    x = ToTensor()(Image.open(image_path))
    y_pred = model.forward(x[None, :]).cpu().detach().numpy()[0]

    print(np.min(y_pred), np.max(y_pred))

    y_pred = y_pred / np.max(y_pred) * 255

    plt.imshow(np.transpose(x, (1, 2, 0)))
    plt.show()
    plt.imshow(np.transpose(y_pred, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    test_prediction(image_path="~\\Documents\\data\\aml\\maskedSubset\\00001_Mask.png")
