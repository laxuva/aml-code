import torch


def precision(y_pred: torch.Tensor, y_true: torch.Tensor, th: float = 0, eps: float = 10 ** -14):
    axes = list(range(1, len(y_pred.shape)))

    y_pred = y_pred > th
    y_true = y_true != 0

    true_positives = torch.sum(torch.logical_and(y_pred, y_true), dim=axes)
    false_positives = torch.sum(torch.logical_and(torch.logical_xor(y_pred, y_true), y_pred), dim=axes)

    return torch.mean((true_positives + eps) / (true_positives + false_positives + eps))
