import torch


class DiceLoss:
    def __init__(self):
        pass

    def __call__(self, y_pred, y_true) -> torch.Tensor:
        return 2 * torch.sum(y_pred * y_true) / (torch.sum(torch.square(y_pred)) + torch.sum(torch.square(y_true)))
