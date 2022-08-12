import json
from pathlib import Path
from typing import List, Callable, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class AutoencoderDataset(Dataset):
    to_tensor: Callable = ToTensor()

    def __init__(
            self,
            original_image_path: Path,
            facemask_image_path: Path,
            original_images: List[str],
            facemask_images: List[str],
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu")
    ):
        self.original_image_path = original_image_path
        self.facemask_image_path = facemask_image_path

        self.original_images = original_images
        self.facemask_images = facemask_images

        self.device = device

        self.loaded_original_images = None
        self.loaded_facemask_images = None

        self.load_data(preload_percentage)

    def load_data(self, preload_percentage: float = 1):
        self.loaded_original_images = [None] * len(self)
        self.loaded_facemask_images = [None] * len(self)

        num_of_instances = int(np.floor(preload_percentage * len(self)))
        for idx in range(num_of_instances):
            self.loaded_original_images[idx] = self._load_image(
                self.original_image_path.joinpath(self.original_images[idx])
            )
            self.loaded_facemask_images[idx] = self._load_image(
                self.facemask_image_path.joinpath(self.facemask_images[idx])
            )

    def _load_image(self, image_path):
        return AutoencoderDataset.to_tensor(Image.open(image_path)).to(self.device)

    @staticmethod
    def load_from_label_file(
            label_file: str,
            original_image_path: Path,
            facemask_image_path: Path,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu")
    ) -> "AutoencoderDataset":
        file = Path(label_file).expanduser()

        with open(file, "r") as f:
            label_file = json.loads(f.read())

        return AutoencoderDataset(
            Path(original_image_path).expanduser(),
            Path(facemask_image_path).expanduser(),
            label_file["original_images"],
            label_file["facemask_images"],
            preload_percentage,
            device
        )

    @staticmethod
    def load_dataset(
            original_image_path: str,
            facemask_image_path: str,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu")
    ) -> "AutoencoderDataset":
        original_image_path = Path(original_image_path).expanduser()
        facemask_image_path = Path(facemask_image_path).expanduser()
        facemask_images = sorted([f.name for f in facemask_image_path.glob("*.png")])
        original_images = [f for f in facemask_images]

        return AutoencoderDataset(
            original_image_path,
            facemask_image_path,
            original_images,
            facemask_images,
            preload_percentage,
            device
        )

    @staticmethod
    def load_train_val_and_test_data(
            original_image_path: str,
            facemask_image_path: str,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu")
    ) -> Tuple["AutoencoderDataset", "AutoencoderDataset", "AutoencoderDataset"]:
        full_dataset = AutoencoderDataset.load_dataset(
            original_image_path,
            facemask_image_path,
            preload_percentage=0,
            device=device
        )

        train_set_full, test_set = full_dataset.split(0.9, preload_percentage=0)

        test_set.load_data(preload_percentage)
        train_set, val_set = train_set_full.split(0.8, preload_percentage=preload_percentage)

        return train_set, val_set, test_set

    def __len__(self):
        return len(self.facemask_images)

    def __getitem__(self, idx):
        if self.loaded_original_images[idx] is not None:
            return self.loaded_original_images[idx], self.loaded_facemask_images[idx]
        return (
            self._load_image(self.original_image_path.joinpath(self.original_images[idx])),
            self._load_image(self.facemask_image_path.joinpath(self.facemask_images[idx]))
        )

    def save(self, file: str):
        file = Path(file).expanduser()

        with open(file, "w") as f:
            f.write(json.dumps({
                "original_images": self.original_images,
                "facemask_images": self.facemask_images
            }))

    def split(
            self,
            split_percentage: float = 0.8,
            seed: int = None,
            preload_percentage: float = 1
    ) -> Tuple["AutoencoderDataset", "AutoencoderDataset"]:
        idxs = np.random.RandomState(seed=seed).permutation(len(self))

        original_images = np.array(self.original_images)[idxs].tolist()
        facemask_images = np.array(self.facemask_images)[idxs].tolist()

        split_idx = int(split_percentage * len(self))

        return (
            AutoencoderDataset(
                self.original_image_path,
                self.facemask_image_path,
                original_images[:split_idx],
                facemask_images[:split_idx],
                preload_percentage,
                self.device
            ),
            AutoencoderDataset(
                self.original_image_path,
                self.facemask_image_path,
                original_images[split_idx:],
                facemask_images[split_idx:],
                preload_percentage,
                self.device
            )
        )


if __name__ == '__main__':
    dataset_train, dataset_val, dataset_test = AutoencoderDataset.load_train_val_and_test_data(
        "~\\Documents\\data\\aml\\masked128png",
        "~\\Documents\\data\\aml\\seg_mask128png",
        preload_percentage=0
    )

    img, label = dataset_train[0]
    print(img.size(), label.size())
