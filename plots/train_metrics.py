from typing import List, Dict

import json

import matplotlib.pyplot as plt


class TrainMetricsPlots:
    def __init__(self, train_info: Dict[str, List[float]]):
        self.train_info = train_info

    def plot_loss(self):
        plt.plot(self.train_info["train_loss"], label="train loss")
        plt.plot(self.train_info["val_loss"], label="val loss")
        plt.legend()
        plt.show()

    def plot_iou(self):
        plt.plot(self.train_info["train_iou"], label="train iou")
        plt.plot(self.train_info["val_iou"], label="val iou")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    with open("../train/train_info.json") as f:
        train_plots = TrainMetricsPlots(json.loads(f.read()))

    train_plots.plot_loss()
    train_plots.plot_iou()
