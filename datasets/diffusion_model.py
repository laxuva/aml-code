import json
from pathlib import Path
from typing import List, Callable, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class DiffusionModelDataset(Dataset):
    to_tensor: Callable = ToTensor()

    def __init__(
            self,
            image_path: Path,
            images: List[str],
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu"),
            transforms: List[Callable] = []
    ):
        self.image_path = image_path
        self.images = images

        self.device = device
        self.transforms = transforms

        self.loaded_images = None

        self.load_data(preload_percentage)

    def load_data(self, preload_percentage: float = 1):
        self.loaded_images: List[Optional[torch.Tensor]] = [None] * len(self)

        for idx in range(int(np.floor(preload_percentage * len(self)))):
            self.loaded_images[idx] = self._load_image(self.image_path.joinpath(self.images[idx]))

    def _load_image(self, image_path):
        img = DiffusionModelDataset.to_tensor(Image.open(image_path)).to(self.device)

        for transform in self.transforms:
            img = transform(img)

        return img

    @staticmethod
    def load_from_label_file(
            label_file: str,
            image_path: str,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu"),
            transforms: List[Callable] = []
    ) -> "DiffusionModelDataset":
        file = Path(label_file).expanduser()

        with open(file, "r") as f:
            label_file = json.loads(f.read())

        return DiffusionModelDataset(
            Path(image_path).expanduser(),
            label_file["images"],
            preload_percentage,
            device,
            transforms
        )

    @staticmethod
    def load_dataset(
            image_path: str,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu"),
            transforms: List[Callable] = []
    ) -> "DiffusionModelDataset":
        image_path = Path(image_path).expanduser()
        images = sorted([f.name for f in image_path.glob("*.png")])

        return DiffusionModelDataset(
            image_path,
            images,
            preload_percentage,
            device,
            transforms
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.loaded_images[idx] is not None:
            return self.loaded_images[idx], 0
        return self._load_image(self.image_path.joinpath(self.images[idx])), 0

    def save(self, file: str):
        file = Path(file).expanduser()

        with open(file, "w") as f:
            f.write(json.dumps({
                "images": self.images
            }))

    def split(
            self,
            split_percentage: float = 0.8,
            seed: int = None,
            preload_percentage: float = 1,
    ) -> Tuple["DiffusionModelDataset", "DiffusionModelDataset"]:
        idxs = np.random.RandomState(seed=seed).permutation(len(self))

        images = np.array(self.images)[idxs].tolist()
        split_idx = int(split_percentage * len(self))

        return (
            DiffusionModelDataset(
                self.image_path,
                images[:split_idx],
                preload_percentage,
                self.device,
                self.transforms
            ),
            DiffusionModelDataset(
                self.image_path,
                images[split_idx:],
                preload_percentage,
                self.device,
                self.transforms
            )
        )
