from typing import Dict, Any
from pathlib import Path
import json

import numpy as np
from torch.utils.data import DataLoader
import torch

from data.segmentation_dataset import SegmentationDataset
from train.unet_trainer import UNetTrainer
from utils.config_parser import ConfigParser


def train(config: Dict[str, Any]):
    train_config = config["training"]
    out_path = Path(train_config["out_path"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset_train, dataset_val, _ = SegmentationDataset.load_train_val_and_test_data(**config["dataset"], device=device)
    train_loader = DataLoader(dataset_train, **config["train_loader"])
    val_loader = DataLoader(dataset_val, **config["val_loader"])

    model = UNetTrainer(config["model"], train_config, device=device)
    lr_scheduler = model.lr_scheduler

    epoch_train_loss = list()
    epoch_val_loss = list()
    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(train_config["max_epochs"]):
        train_loss = list()
        val_loss = list()

        for x, y in train_loader:
            train_loss.append(model.train_on_batch(x, y))

        for x, y in val_loader:
            val_loss.append(model.validation_step(x, y))

        epoch_train_loss.append(np.mean(train_loss).item())
        epoch_val_loss.append(np.mean(val_loss).item())

        lr_scheduler.step()

        print(f"[epoch {epoch + 1}/{train_config['max_epochs']}] "
              f"train loss: {epoch_train_loss[-1]}; val loss: {epoch_val_loss[-1]}")

        if epoch_val_loss[-1] < best_val_loss:
            best_val_loss = epoch_val_loss[-1]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), out_path.joinpath("best_model.pt"))
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement > train_config["break_criterion"]:
                break

    with open(out_path.joinpath("train_info.json"), "w") as f:
        f.write(json.dumps({
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss
        }))

    torch.save(model.state_dict(), out_path.joinpath("final_model.pt"))


if __name__ == '__main__':
    train(ConfigParser.read("../configs/debugging.yaml"))
