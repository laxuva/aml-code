from pathlib import Path

from torchvision.transforms import Resize, ToTensor, ToPILImage, Compose
from PIL import Image


def downsample(in_folder: str, out_folder: str, desired_size: tuple):
    transforms = Compose([
        ToTensor(),
        Resize(desired_size),
        ToPILImage()
    ])

    files = Path(in_folder).expanduser().glob("*.jpg")
    out_folder = Path(out_folder).expanduser()

    for img_file in files:
        transforms(Image.open(img_file)).save(out_folder.joinpath(img_file.name.replace(".jpg", ".png")))


if __name__ == '__main__':
    downsample("~/Documents/data/aml/img_facemask", "~/Documents/data/aml/img_facemask_downsampled", (128, 128))
