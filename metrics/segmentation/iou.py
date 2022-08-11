import torch


def iou(y_pred: torch.Tensor, y: torch.Tensor, th: float = 0, eps: float = 10**-14):
    axes = list(range(1, len(y_pred.shape)))

    y_pred = y_pred > th
    y = y > th

    intersection = y_pred * y * 2
    union = y_pred + y

    return torch.mean((torch.sum(intersection, dim=axes) + eps) / (torch.sum(union, dim=axes) + eps))
