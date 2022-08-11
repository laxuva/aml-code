from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from metrics.segmentation.iou import iou
from network.segmentation.unet import UNet
from plots.segmentation_overlay import SegmentationOverlay
from train.unet_trainer import UNetTrainer
from utils.config_parser import ConfigParser


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

    print(f"y_pred min: {y_pred.min().item()}; max: {y_pred.max().item()}")
    print(f"IOU: {iou(y_pred, y, th=th).item()}")

    x = np.transpose(x.cpu().detach().numpy(), (1, 2, 0))
    y_pred = np.transpose(y_pred.numpy(), (1, 2, 0))

    overlay = SegmentationOverlay(x, y_pred, y.numpy()[0, :, :], th)
    overlay.plot_image()
    overlay.plot_y_pred(binary=False)
    overlay.plot_y_pred(binary=True)
    # overlay.plot_y_true()
    # overlay.plot_y_true_overlay()
    # overlay.plot_y_pred_overlay()
    overlay.plot_y_pred_and_y_true_overlay()


if __name__ == '__main__':
    test_prediction(
        model_path="../train/final_model.pt",
        image_path="~\\Documents\\data\\aml\\masked128png\\45844_Mask.png",
        label_path="~\\Documents\\data\\aml\\seg_mask128png\\45844_seg.png",
        th=0.002
    )
