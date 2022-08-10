from pathlib import Path
from typing import List, Callable, Tuple

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
            preload_percentage: float = 1
    ):
        self.image_path = image_path
        self.label_path = label_path
        self.images = images
        self.labels = labels

        self.loaded_images = None
        self.loaded_labels = None

        self.load_data(preload_percentage)

    def load_data(self, preload_percentage: float = 1):
        self.loaded_images = [None] * len(self)
        self.loaded_labels = [None] * len(self)

        for idx in range(int(preload_percentage * len(self))):
            self.loaded_images[idx] = self._load_image(self.image_path.joinpath(self.images[idx]))
            self.loaded_labels[idx] = self._load_image(self.label_path.joinpath(self.labels[idx]))

    @staticmethod
    def _load_image(image_path):
        return SegmentationDataset.to_tensor(Image.open(image_path))

    @staticmethod
    def load_dataset(
            image_path: str, 
            label_path: str,
            preload_percentage: float = 1
    ) -> "SegmentationDataset":
        image_path = Path(image_path).expanduser()
        label_path = Path(label_path).expanduser()

        images = sorted([f.name for f in image_path.glob("*.png")])
        labels = [f.replace("Mask", "seg") for f in images]

        return SegmentationDataset(image_path, label_path, images, labels, preload_percentage)

    @staticmethod
    def load_train_val_and_test_data(
            image_path: str, 
            label_path: str,
            preload_percentage: float = 1
    ) -> Tuple["SegmentationDataset", "SegmentationDataset", "SegmentationDataset"]:
        full_dataset = SegmentationDataset.load_dataset(image_path, label_path, preload_percentage=0)

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
                preload_percentage
            ),
            SegmentationDataset(
                self.image_path,
                self.label_path,
                images[split_idx:],
                labels[split_idx:],
                preload_percentage
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
