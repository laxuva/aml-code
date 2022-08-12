import torch


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, th: float = 0):
    axes = list(range(1, len(y_pred.shape)))

    y_pred = y_pred > th
    y_true = y_true != 0
    
    return torch.mean(torch.mean((y_true == y_pred).float(), dim=axes))
