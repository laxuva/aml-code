from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from metrics.segmentation.iou import iou
from network.unet import UNet
from plots.segmentation_overlay import SegmentationOverlay
from utils.config_parser import ConfigParser


@torch.no_grad()
def test_prediction(model_path, image_path, label_path, out_path, th=0.25, config_file="../configs/segmentation.yaml"):
    config = ConfigParser.read(config_file)
    image_path = Path(image_path).expanduser()
    label_path = Path(label_path).expanduser()
    out_path = Path(out_path).expanduser()

    to_tensor = ToTensor()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UNet(**config["model"]["params"])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = to_tensor(Image.open(image_path)).to(device)
    y = to_tensor(Image.open(label_path))
    y_pred = model.forward(x[None, :])[0].cpu().detach()

    ToPILImage()(torch.tensor(y_pred > th, dtype=float)).save(out_path.joinpath(f"predicted_segmentation.png"))
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
        model_path="../evaluation/segmentation/final_model.pt",
        image_path="~/Documents/data/aml/masked128png/65959.png",
        label_path="~/Documents/data/aml/seg_mask128png/65959.png",
        out_path="~/Documents/data/aml/out/",
        th=0.002
    )
