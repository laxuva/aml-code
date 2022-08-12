import torch


def iou(y_pred: torch.Tensor, y_true: torch.Tensor, th: float = 0, eps: float = 10**-14):
    axes = list(range(1, len(y_pred.shape)))

    y_pred = y_pred > th
    y_true = y_true != 0

    intersection = torch.logical_and(y_pred, y_true)
    union = torch.logical_or(y_pred, y_true)

    numerator = torch.sum(intersection, dim=axes) + eps
    denominator = torch.sum(union, dim=axes) + eps

    return torch.mean(numerator / denominator)
