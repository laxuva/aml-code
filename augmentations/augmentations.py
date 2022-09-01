from abc import ABC, abstractmethod
from typing import Tuple
from torchvision.transforms import RandomHorizontalFlip
import numpy as np

import torch


class BaseAugmentation(ABC):
    def __init__(self, p: float = 1):
        self.p = p

    def __call__(self, img: torch.Tensor, mask: torch.Tensor = None):
        if np.random.random() < self.p:
            return self.apply(img)

        if mask is None:
            return img
        return img, mask

    @abstractmethod
    def apply(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class Identity:
    def __call__(self, img: torch.Tensor, mask: torch.Tensor = None):
        if mask is None:
            return img
        return img, mask


class FlipLeftRight(BaseAugmentation):
    def __init__(self, p: float = 1):
        super(FlipLeftRight, self).__init__(p)
        self.flip_method = RandomHorizontalFlip(p=1)

    def apply(self, img: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            return self.flip_method(img)
        return self.flip_method(img), self.flip_method(mask)
