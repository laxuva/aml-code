import torch


def iou(y_pred: torch.Tensor, y: torch.Tensor, th: float = 0):
    axes = list(range(1, len(y_pred.shape)))

    th = th * 2

    intersection = y_pred * y * 2
    union = (y_pred + y)

    return torch.mean(torch.sum(intersection > th, dim=axes) / torch.sum(union > th, dim=axes))
