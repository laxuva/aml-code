import shutil
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SegmentationDataset, AutoencoderDataset, DiffusionModelDataset
from train.trainer import AdversarialAutoencoderTrainer, AutoencoderTrainer, DiffusionModelTrainer, SegmentationTrainer
from utils.config_parser import ConfigParser
from augmentations import FlipLeftRight


def train(config: Dict[str, Any], save_output: bool = True):
    train_config = config["training"]
    out_path = Path(train_config["out_path"]).expanduser()

    if save_output:
        if out_path.exists():
            shutil.rmtree(out_path, ignore_errors=True)
        out_path.mkdir()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    try:
        dataset_class = {
            "SegmentationDataset": SegmentationDataset,
            "AutoencoderDataset": AutoencoderDataset,
            "DiffusionModelDataset": DiffusionModelDataset
        }[config["dataset"]["type"]]
    except KeyError:
        raise NotImplementedError(f"The dataset class {config['dataset']['type']} is not available")

    if "do_augmentations" in config["training"] and config["training"]["do_augmentations"]:
        config["dataset"]["params"]["augmentations"] = FlipLeftRight(p=0.5)

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

    try:
        model_class = {
            "SegmentationTrainer": SegmentationTrainer,
            "AutoencoderTrainer": AutoencoderTrainer,
            "AdversarialAutoencoderTrainer": AdversarialAutoencoderTrainer,
            "DiffusionModelTrainer": DiffusionModelTrainer
        }[config["model"]["type"]]
    except KeyError:
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
        model = model_class(config["model"]["params"], train_config, device=device, save_output=save_output)

    best_val_loss = np.inf
    epochs_without_improvement = 0
    best_state_dict = dict()

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

        if model.metrics_logger.get_last("val_loss") <= best_val_loss:
            best_val_loss = model.metrics_logger.get_last("val_loss")
            epochs_without_improvement = 0

            if save_output:
                torch.save(model.get_model().state_dict(), out_path.joinpath("best_model.pt"))
            best_state_dict = model.get_model().state_dict().copy()
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement > train_config["break_criterion"]:
                break

    if save_output:
        torch.save(model.get_model().state_dict(), out_path.joinpath("final_model.pt"))
        model.metrics_logger.save()

    return best_state_dict, best_val_loss


if __name__ == '__main__':
    train(ConfigParser.read("../configs/autoencoder.yaml"))  # diffusion_model autoencoder
