from typing import List, Dict
from pathlib import Path

import json

import matplotlib.pyplot as plt


class TrainMetricsPlots:
    def __init__(self, train_info: Dict[str, List[float]], out_dir: str = None):
        self.train_info = train_info
        self.out_dir = Path(out_dir).expanduser() if out_dir is not None else None

    def plot_loss(self, save: bool = False):
        plt.plot(self.train_info["train_loss"], label="train loss")
        plt.plot(self.train_info["val_loss"], label="val loss")
        plt.legend()
        if save and self.out_dir is not None:
            plt.savefig(self.out_dir.joinpath("loss.pdf"), bbox_inches="tight")
        plt.show()

    def plot_iou(self, save: bool = False):
        plt.plot(self.train_info["train_iou"], label="train iou")
        plt.plot(self.train_info["val_iou"], label="val iou")
        plt.legend()
        if save and self.out_dir is not None:
            plt.savefig(self.out_dir.joinpath("iou.pdf"), bbox_inches="tight")
        plt.show()


if __name__ == '__main__':
    # with open("../evaluation/diffusion_model/train_info.json") as f:
    #     TrainMetricsPlots(json.loads(f.read()), ".").plot_loss(save=True)

    with open("../evaluation/autoencoder/train_info.json") as f:
        TrainMetricsPlots(json.loads(f.read()), ".").plot_loss(save=True)
