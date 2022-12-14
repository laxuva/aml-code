import json
from pathlib import Path
from typing import List, Callable, Tuple, Optional

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from augmentations import Identity


class AutoencoderDataset(Dataset):
    to_tensor: Callable = ToTensor()

    def __init__(
            self,
            original_image_path: Path,
            seg_map_image_path: Path,
            original_images: List[str],
            seg_map_images: List[str],
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu"),
            add_seg_map_channel_to_x: bool = True,
            augmentations: Callable = Identity()
    ):
        self.original_image_path = original_image_path
        self.seg_map_image_path = seg_map_image_path

        self.original_images = original_images
        self.seg_map_images = seg_map_images

        self.device = device
        self.add_seg_map_channel_to_x = add_seg_map_channel_to_x

        self.loaded_original_images = None
        self.loaded_facemask_images = None

        self.augmentations = augmentations

        self.load_data(preload_percentage)

    def load_data(self, preload_percentage: float = 1):
        self.loaded_facemask_images: List[Optional[torch.Tensor]] = [None] * len(self)
        self.loaded_original_images: List[Optional[torch.Tensor]] = [None] * len(self)

        num_of_instances = int(np.floor(preload_percentage * len(self)))
        for idx in range(num_of_instances):
            self.loaded_original_images[idx] = self._load_image(
                self.original_image_path.joinpath(self.original_images[idx])
            )

            seg_map = self._load_image(
                self.seg_map_image_path.joinpath(self.seg_map_images[idx])
            )

            self.loaded_facemask_images[idx] = self.loaded_original_images[idx].clone()
            self.loaded_facemask_images[idx][:, seg_map[0] != 0] = 0

            if self.add_seg_map_channel_to_x:
                self.loaded_facemask_images[idx] = torch.cat((self.loaded_facemask_images[idx], seg_map))

    def _load_image(self, image_path):
        return AutoencoderDataset.to_tensor(Image.open(image_path)).to(self.device)

    @staticmethod
    def load_from_label_file(
            label_file: str,
            original_image_path: Path,
            seg_map_image_path: Path,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu"),
            augmentations: Callable = Identity()
    ) -> "AutoencoderDataset":
        file = Path(label_file).expanduser()

        with open(file, "r") as f:
            label_file = json.loads(f.read())

        return AutoencoderDataset(
            Path(original_image_path).expanduser(),
            Path(seg_map_image_path).expanduser(),
            label_file["images"],
            label_file["images"],
            preload_percentage,
            device,
            augmentations=augmentations
        )

    @staticmethod
    def load_dataset(
            original_image_path: str,
            seg_map_image_path: str,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu"),
            augmentations: Callable = Identity()
    ) -> "AutoencoderDataset":
        original_image_path = Path(original_image_path).expanduser()
        seg_map_image_path = Path(seg_map_image_path).expanduser()
        seg_map_images = sorted([f.name for f in seg_map_image_path.glob("*.png")])
        original_images = seg_map_images

        return AutoencoderDataset(
            original_image_path,
            seg_map_image_path,
            original_images,
            seg_map_images,
            preload_percentage,
            device,
            augmentations=augmentations
        )

    @staticmethod
    def load_train_val_and_test_data(
            original_image_path: str,
            seg_map_image_path: str,
            preload_percentage: float = 1,
            device: torch.device = torch.device("cpu")
    ) -> Tuple["AutoencoderDataset", "AutoencoderDataset", "AutoencoderDataset"]:
        full_dataset = AutoencoderDataset.load_dataset(
            original_image_path,
            seg_map_image_path,
            preload_percentage=0,
            device=device
        )

        train_set_full, test_set = full_dataset.split(0.9, preload_percentage=0)

        test_set.load_data(preload_percentage)
        train_set, val_set = train_set_full.split(0.8, preload_percentage=preload_percentage)

        return train_set, val_set, test_set

    def __len__(self):
        return len(self.seg_map_images)

    def __getitem__(self, idx):
        if self.loaded_original_images[idx] is not None:
            return self.augmentations(self.loaded_facemask_images[idx], self.loaded_original_images[idx])

        img = self._load_image(self.original_image_path.joinpath(self.original_images[idx]))
        seg_map = self._load_image(self.seg_map_image_path.joinpath(self.seg_map_images[idx]))

        face_mask_img = img.clone()
        face_mask_img[:, seg_map[0] != 0] = 0

        if self.add_seg_map_channel_to_x:
            face_mask_img = torch.cat((face_mask_img, seg_map))

        return self.augmentations(face_mask_img, img)

    def save(self, file: str):
        file = Path(file).expanduser()

        with open(file, "w") as f:
            f.write(json.dumps({
                "images": self.original_images,  # as the images are named exactly the same
                # "seg_map_images": self.seg_map_images
            }))

    def split(
            self,
            split_percentage: float = 0.8,
            seed: int = None,
            preload_percentage: float = 1
    ) -> Tuple["AutoencoderDataset", "AutoencoderDataset"]:
        idxs = np.random.RandomState(seed=seed).permutation(len(self))

        original_images = np.array(self.original_images)[idxs].tolist()
        seg_map_images = np.array(self.seg_map_images)[idxs].tolist()

        split_idx = int(split_percentage * len(self))

        return (
            AutoencoderDataset(
                self.original_image_path,
                self.seg_map_image_path,
                original_images[:split_idx],
                seg_map_images[:split_idx],
                preload_percentage,
                self.device
            ),
            AutoencoderDataset(
                self.original_image_path,
                self.seg_map_image_path,
                original_images[split_idx:],
                seg_map_images[split_idx:],
                preload_percentage,
                self.device
            )
        )
