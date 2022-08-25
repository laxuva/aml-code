import json
from pathlib import Path


def create_smaller_dataset(label_file: str, new_label_file: str, percentage: float = 0.5):
    with open(Path(label_file).expanduser(), "r") as f:
        label_file_content = json.load(f)

    images = label_file_content["images"]
    images = images[:int(percentage * len(images)) + 1]

    print(len(label_file_content["images"]), len(images))

    with open(Path(new_label_file).expanduser(), "w") as f:
        json.dump({"images": images}, f)


if __name__ == '__main__':
    create_smaller_dataset("~/Documents/data/aml/train_dataset.json", "~/Documents/data/aml/train_dataset_large.json")
    create_smaller_dataset("~/Documents/data/aml/val_dataset.json", "~/Documents/data/aml/val_dataset_large.json")
