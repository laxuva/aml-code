import json
from pathlib import Path
from typing import List, Callable, Tuple

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class SegmentationDataset(Dataset):
    to_tensor: Callable = ToTensor()

    def __init__(
            self,
            image_path: Path,
            label_path: Path,
            images: List[str],
            labels: List[str],
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu")
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.images = images
        self.labels = labels

        self.device = device

        self.loaded_images = None
        self.loaded_labels = None

        self.load_data(preload_percentage)

    def load_data(self, preload_percentage: float = 1):
        self.loaded_images = [None] * len(self)
        self.loaded_labels = [None] * len(self)

        num_of_instances = int(np.floor(preload_percentage * len(self)))
        for idx in range(num_of_instances):
            self.loaded_images[idx] = self._load_image(self.image_path.joinpath(self.images[idx]))
            self.loaded_labels[idx] = self._load_image(self.label_path.joinpath(self.labels[idx]))

    def _load_image(self, image_path):
        return SegmentationDataset.to_tensor(Image.open(image_path)).to(self.device)

    @staticmethod
    def load_from_label_file(
            label_file: str,
            image_path: str,
            label_path: str,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu")
    ) -> "SegmentationDataset":
        file = Path(label_file).expanduser()

        with open(file, "r") as f:
            label_file = json.loads(f.read())

        return SegmentationDataset(
            Path(image_path).expanduser(),
            Path(label_path).expanduser(),
            label_file["images"],
            label_file["labels"],
            preload_percentage,
            device
        )

    @staticmethod
    def load_dataset(
            image_path: str, 
            label_path: str,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu")
    ) -> "SegmentationDataset":
        image_path = Path(image_path).expanduser()
        label_path = Path(label_path).expanduser()

        images = sorted([f.name for f in image_path.glob("*.png")])
        labels = images.copy()

        return SegmentationDataset(image_path, label_path, images, labels, preload_percentage, device)

    @staticmethod
    def load_train_val_and_test_data(
            image_path: str, 
            label_path: str,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu")
    ) -> Tuple["SegmentationDataset", "SegmentationDataset", "SegmentationDataset"]:
        full_dataset = SegmentationDataset.load_dataset(image_path, label_path, preload_percentage=0, device=device)

        train_set_full, test_set = full_dataset.split(0.9, preload_percentage=0)

        test_set.load_data(preload_percentage)
        train_set, val_set = train_set_full.split(0.8, preload_percentage=preload_percentage)

        return train_set, val_set, test_set

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.loaded_images[idx] is not None:
            return self.loaded_images[idx], self.loaded_labels[idx]
        return (
            self._load_image(self.image_path.joinpath(self.images[idx])),
            self._load_image(self.label_path.joinpath(self.labels[idx]))
        )

    def save(self, file: str):
        file = Path(file).expanduser()

        with open(file, "w") as f:
            f.write(json.dumps({
                "images": self.images,
                "labels": self.labels
            }))

    def split(
            self, 
            split_percentage: float = 0.8, 
            seed: int = None,
            preload_percentage: float = 1
    ) -> Tuple["SegmentationDataset", "SegmentationDataset"]:
        idxs = np.random.RandomState(seed=seed).permutation(len(self))

        images = np.array(self.images)[idxs].tolist()
        labels = np.array(self.labels)[idxs].tolist()

        split_idx = int(split_percentage * len(self))

        return (
            SegmentationDataset(
                self.image_path,
                self.label_path,
                images[:split_idx],
                labels[:split_idx],
                preload_percentage,
                self.device
            ),
            SegmentationDataset(
                self.image_path,
                self.label_path,
                images[split_idx:],
                labels[split_idx:],
                preload_percentage,
                self.device
            )
        )


if __name__ == '__main__':
    dataset_train, dataset_val, dataset_test = SegmentationDataset.load_train_val_and_test_data(
        "~\\Documents\\data\\aml\\masked128png",
        "~\\Documents\\data\\aml\\seg_mask128png",
        preload_percentage=0
    )

    img, label = dataset_train[0]
    print(img.size(), label.size())
