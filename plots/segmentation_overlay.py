from sys import stderr

import cv2
import matplotlib.pyplot as plt
import numpy as np


class SegmentationOverlay:
    def __init__(self, image: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, y_pred_th: float):
        """
        Create various segmentation overlays

        :param image: rgb image (w x h x 3)
        :param y_pred: float segmentation mask (w x h) in range [0, 1]
        :param y_true: boolean segmentation mask (w x h)
        """
        if len(y_pred.shape) == 3 and y_pred.shape[2] == 1:
            y_pred = y_pred[:, :, 0]
        elif len(y_pred.shape) != 2:
            raise ValueError("Expected grayscale/boolean segmentation mask")

        if len(y_true.shape) == 3 and y_true.shape[2] == 1:
            y_true = y_true[:, :, 0]
        elif len(y_true.shape) != 2:
            raise ValueError("Expected grayscale/boolean segmentation mask")

        if y_true.dtype != np.bool:
            print("The given y_true is not of type bool!", file=stderr)
            y_true = y_true != 0

        self.image = image
        self.y_pred = y_pred
        self.y_pred_bin = y_pred > y_pred_th
        self.y_true = y_true

    @staticmethod
    def plot_overlay(image: np.ndarray, segmentation_mask: np.ndarray, title=""):
        overlay = np.zeros_like(image)
        overlay[segmentation_mask, 1] = 1
        plt.imshow(cv2.addWeighted(image, 0.7, overlay, 0.3, 0))

        if title != "":
            plt.title(title)

        plt.show()

    @staticmethod
    def _plot(img: np.ndarray, title: str = ""):
        plt.imshow(img)

        if title != "":
            plt.title(title)

        plt.show()

    def plot_y_pred(self, binary: bool = False):
        if binary:
            self._plot(self.y_pred_bin.astype(np.uint8), title="y_pred (binary)")
        else:
            self._plot(self.y_pred, title="y_pred (float)")

    def plot_y_true(self):
        self._plot(self.y_true.astype(np.uint8), title="y_true")

    def plot_image(self):
        self._plot(self.image, title="image")

    def plot_y_pred_overlay(self):
        return self.plot_overlay(self.image, self.y_pred_bin, title="y_pred")

    def plot_y_true_overlay(self):
        return self.plot_overlay(self.image, self.y_true, title="y_true")

    def plot_y_pred_and_y_true_overlay(self):
        overlay = np.zeros_like(self.image)
        overlay[np.logical_xor(np.logical_and(self.y_pred_bin, self.y_true), self.y_true), 0] = 1
        overlay[np.logical_and(self.y_pred_bin, self.y_true), 1] = 1
        overlay[np.logical_xor(np.logical_and(self.y_pred_bin, self.y_true), self.y_pred_bin), 2] = 1
        plt.imshow(cv2.addWeighted(self.image, 0.7, overlay, 0.3, 0))
        plt.show()
