import torch

from metrics.segmentation import recall, precision


def f1_score(y_pred: torch.Tensor, y_true: torch.Tensor, th: float = 0, eps: float = 10 ** -14):
    r = recall(y_pred, y_true, th, eps)
    p = precision(y_pred, y_true, th, eps)
    return 2 * r * p / (p + r)
