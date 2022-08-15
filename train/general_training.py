from typing import Dict, Any
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
import torch

from data.segmentation_dataset import SegmentationDataset
from data.autoencoder_dataset import AutoencoderDataset
from train.segmentation.unet_trainer import UNetTrainer
from train.autoencoder.autoencoder_trainer import AutoencoderTrainer
from train.autoencoder.adversarial_trainer import AdversarialAutoencoderTrainer
from utils.config_parser import ConfigParser
from tqdm import tqdm


def train(config: Dict[str, Any]):
    train_config = config["training"]
    out_path = Path(train_config["out_path"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if config["dataset"]["type"] == "SegmentationDataset":
        dataset_class = SegmentationDataset
    elif config["dataset"]["type"] == "AutoencoderDataset":
        dataset_class = AutoencoderDataset
    else:
        raise NotImplementedError(f"The dataset class {config['dataset']['type']} is not available")

    dataset_train = dataset_class.load_from_label_file(
        config["dataset"]["train_label"],
        **config["dataset"]["params"],
        device=device
    )
    dataset_val = dataset_class.load_from_label_file(
        config["dataset"]["val_label"],
        **config["dataset"]["params"],
        device=device
    )
    print(len(dataset_train), len(dataset_val))

    train_loader = DataLoader(dataset_train, **config["train_loader"])
    val_loader = DataLoader(dataset_val, **config["val_loader"])

    if config["model"]["type"] == "UNetTrainer":
        model_class = UNetTrainer
    elif config["model"]["type"] == "AutoencoderTrainer":
        model_class = AutoencoderTrainer
    elif config["model"]["type"] == "AdversarialAutoencoderTrainer":
        model_class = AdversarialAutoencoderTrainer
    else:
        raise NotImplementedError(f"The model class {config['model']['type']} is not available")

    if model_class == AdversarialAutoencoderTrainer:
        model = model_class(
            config["model"]["params"],
            train_config,
            device=device,
            train_dataset=dataset_train,
            val_dataset=dataset_val
        )
    else:
        model = model_class(config["model"]["params"], train_config, device=device)

    best_val_loss = np.inf
    epochs_without_improvement = 0

    for epoch in range(train_config["max_epochs"]):
        train_loss = list()
        val_loss = list()

        model.train()
        for x, y in tqdm(train_loader):
            train_loss.append(model.train_on_batch(x, y))

        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                val_loss.append(model.validation_step(x, y))

        model.end_epoch()  # does a lr scheduler step and a metrics logger step

        if model.metrics_logger.get_last("val_loss") < best_val_loss:
            best_val_loss = model.metrics_logger.get_last("val_loss")
            epochs_without_improvement = 0
            torch.save(model.get_model().state_dict(), out_path.joinpath("best_model.pt"))
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement > train_config["break_criterion"]:
                break

    torch.save(model.get_model().state_dict(), out_path.joinpath("final_model.pt"))


if __name__ == '__main__':
    train(ConfigParser.read("../configs/debugging_autoencoder.yaml"))
