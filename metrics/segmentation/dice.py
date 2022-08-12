import torch


def dice(y_pred: torch.Tensor, y_true: torch.Tensor, th: float = 0, epsilon=1e-14):
    axes = list(range(1, len(y_pred.shape)))

    y_pred = y_pred > th
    y_true = y_true != 0

    numerator = 2. * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum(torch.square(y_pred) + torch.square(y_true), axes)

    return torch.mean((numerator + epsilon) / (denominator + epsilon))
