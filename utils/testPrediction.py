from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor

from network.segmentation.unet import UNet
from train.unet_trainer import UNetTrainer
from utils.config_parser import ConfigParser
from metrics.segmentation.iou import iou


def test_prediction(
        model_path="../train/best_model_12_epochs.pt",
        image_path="D:/aml/localData/masked128png/00000_Mask.png",
        label_path="D:/aml/localData/seg_mask128png/00000_Mask.png",
        th: float = 0.25
):
    config = ConfigParser.read("../configs/debugging.yaml")
    image_path = Path(image_path).expanduser()
    label_path = Path(label_path).expanduser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        model = UNet(**config["model"])
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError:
        model = UNetTrainer(config["model"], config["training"], device=device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    x = ToTensor()(Image.open(image_path)).to(device)
    y = ToTensor()(Image.open(label_path))
    y_pred = model.forward(x[None, :])[0].cpu().detach()

    print(y_pred.min(), y_pred.max())
    print(iou(y_pred, y, th=th))

    y_pred = y_pred.numpy()

    x = np.transpose(x.cpu().detach().numpy(), (1, 2, 0))
    y_pred = np.transpose(y_pred, (1, 2, 0))

    plt.imshow(x)
    plt.show()
    plt.imshow(y_pred / np.max(y_pred) * 255)
    plt.show()
    plt.imshow((y_pred > th).astype(np.uint8) * 255)
    plt.show()

    overlay = np.zeros_like(x)
    overlay[:, :, 1] = y_pred[:, :, 0] > th
    plt.imshow(cv2.addWeighted(x, 0.5, overlay, 0.3, 0))
    plt.show()


if __name__ == '__main__':
    test_prediction(
        model_path="../train/best_model.pt",
        image_path="~\\Documents\\data\\aml\\masked128png\\45844_Mask.png",
        label_path="~\\Documents\\data\\aml\\seg_mask128png\\45844_seg.png",
        th=0.002
    )
