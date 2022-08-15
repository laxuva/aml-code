from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from network.segmentation.unet import UNet
from utils.config_parser import ConfigParser


def test_prediction(
        model_path,
        image_path="D:/aml/localData/masked128png/00000_Mask.png",
        label_path="D:/aml/localData/seg_mask128png/00000_Mask.png"
):
    config = ConfigParser.read("../configs/debugging_autoencoder.yaml")
    image_path = Path(image_path).expanduser()
    label_path = Path(label_path).expanduser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UNet(**config["model"]["params"])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = ToTensor()(Image.open(image_path)).to(device)
    y = ToTensor()(Image.open(label_path))
    y_pred = model.forward(x[None, :])[0].cpu().detach()

    ToPILImage()(y_pred).save("./test.png")

    if config["training"]["predict_difference"]:
        print(f"MAE: {torch.mean(torch.abs(y_pred + x - y))}")
    else:
        print(f"MAE: {torch.mean(torch.abs(y_pred - y))}")

    x = np.transpose(x.cpu().detach().numpy(), (1, 2, 0))
    y_pred = np.transpose(y_pred.numpy(), (1, 2, 0))
    y = np.transpose(y.numpy(), (1, 2, 0))

    plt.imshow(y)
    plt.show()

    plt.imshow(y_pred)
    plt.show()

    y_pred[x != 0] = 0
    plt.imshow(y_pred)
    plt.show()

    if config["training"]["predict_difference"]:
        plt.imshow(y_pred + x)
        plt.show()


if __name__ == '__main__':
    test_prediction(
        model_path="../evaluation/best_model.pt",
        image_path="~\\Documents\\data\\aml\\autoencoder128png\\45844_Mask.png",
        label_path="~\\Documents\\data\\aml\\original128png\\45844.png"
    )
