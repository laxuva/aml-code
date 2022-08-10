import torch

from PIL import Image
from network.segmentation.unet import UNet
from utils.config_parser import ConfigParser
from matplotlib import pyplot as plt


def test_prediction(
        model_path="../train/best_model.pt",
        image_path="D:/aml/localData/masked128png/00000_Mask.png"
        ):
    config = ConfigParser.read("../configs/debugging.yaml")

    model = UNet(**config["model"])
    model.load_state_dict(torch.load(model_path))

    model.eval()
    x = Image.open(image_path)
    y_pred = model.forward(x)

    plt.imshow(x)
    plt.show()
    plt.imshow(y_pred)
    plt.show()


if __name__ == '__main__':
    test_prediction()
