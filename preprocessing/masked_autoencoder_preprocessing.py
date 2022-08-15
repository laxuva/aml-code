from pathlib import Path

import tqdm
from PIL import Image
import numpy as np


def mask_the_face_masks(face_mask_folder: str, label_map_folder: str, out_path: str):
    face_mask_folder = Path(face_mask_folder).expanduser()
    label_map_folder = Path(label_map_folder).expanduser()

    out_path = Path(out_path).expanduser()

    if not out_path.exists():
        out_path.mkdir()

    face_mask_images = [f for f in face_mask_folder.glob("*.png")]
    labels_maps = [label_map_folder.joinpath(f.name) for f in face_mask_images]

    for image_file, label_file in tqdm.tqdm(zip(face_mask_images, labels_maps)):
        img = np.array(Image.open(image_file)).astype(float)
        label = np.array(Image.open(label_file))

        img += 1
        img[label != 0] = 0
        img = img / np.max(img) * 255

        Image.fromarray(img.astype(np.uint8)).save(out_path.joinpath(image_file.name))


if __name__ == '__main__':
    mask_the_face_masks(
        "~\\Documents\\data\\aml\\masked128png",
        "~\\Documents\\data\\aml\\seg_mask128png",
        "~\\Documents\\data\\aml\\autoencoder128png"
    )
