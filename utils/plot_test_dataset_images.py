import matplotlib.pyplot as plt
import numpy as np

from datasets import DiffusionModelDataset
from utils.config_parser import ConfigParser


if __name__ == '__main__':
    config = ConfigParser.read("../configs/diffusion_model.yaml")

    dataset_test = DiffusionModelDataset.load_from_label_file(
        config["dataset"]["test_label"],
        **config["dataset"]["params"],
    )

    for idx, (img, _) in enumerate(dataset_test):
        img = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))

        plt.imshow(img)
        plt.title(dataset_test.images[idx])
        plt.show()
