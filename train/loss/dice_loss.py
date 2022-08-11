import torch


class DiceLoss:
    def __init__(
            self
            ):
        pass

    def __call__(
            self,
            y_pred,
            y_true,
            epsilon=1e-14
            ) -> torch.Tensor:
        # return 2 * torch.sum(y_pred * y_true) / (torch.sum(torch.square(y_pred)) + torch.sum(torch.square(y_true)))

        axes = list(range(1, len(y_pred.shape)))

        numerator = 2. * torch.sum(y_pred * y_true,
                                   axes)
        denominator = torch.sum(torch.square(y_pred) + torch.square(y_true),
                                axes)

        return 1 - torch.mean((numerator + epsilon) / (denominator + epsilon))