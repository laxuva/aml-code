from pathlib import Path

import torch
import tqdm

from datasets import SegmentationDataset
from metrics.segmentation import *
from network.segmentation import UNet
from utils.config_parser import ConfigParser


def evaluate(y_pred, y_true, th):
    return torch.tensor([
        accuracy(y_pred, y_true, th=th),
        dice(y_pred, y_true, th=th),
        iou(y_pred, y_true, th=th),
        precision(y_pred, y_true, th=th),
        recall(y_pred, y_true, th=th),
        f1_score(y_pred, y_true, th=th)
    ])


def evaluate_network(model: UNet, dataset: SegmentationDataset, th=0.02):
    model.eval()

    metrics = ["accuracy", "dice", "iou", "precision", "recall", "f1_score"]
    scores = torch.zeros((6,))

    with torch.no_grad():
        for instance, y_true in tqdm.tqdm(dataset):
            y_pred = model.forward(instance[None, :])
            y_true = y_true[None, :]

            scores += evaluate(y_pred, y_true, th)

    scores /= len(dataset)

    print("\n".join(f"{metrics[i]}: {scores[i]}" for i in range(len(metrics))))


def test_main(config_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ConfigParser.read(str(Path(config_path).expanduser()))

    model = UNet(**config["model"])
    model.to(device)
    model_path = Path(config["training"]["out_path"]).expanduser().joinpath("best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))

    dataset = SegmentationDataset.load_from_label_file(
        config["dataset"]["test_label"],
        **config["dataset"]["params"],
        device=device
    )

    evaluate_network(model, dataset, th=config["training"]["iou_th"])


if __name__ == '__main__':
    test_main("../../configs/segmentation.yaml")
