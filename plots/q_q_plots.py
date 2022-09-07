from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from datasets import AutoencoderDataset
from network.unet import UNet
from network.unet_with_embedding import UNet as UNetWithEmbedding
from utils.config_parser import ConfigParser
from predictions.diffusion_model_for_masked_img import test_prediction


class QQPlot:
    def __init__(self, features_y: List[float], features_y_pred: List[float], title: str = ""):
        assert len(features_y) == len(features_y_pred)

        self.features_y = sorted(features_y)
        self.features_y_pred = sorted(features_y_pred)

        self.min = min(min(features_y), min(features_y_pred))
        self.max = max(max(features_y), max(features_y_pred))

        self.title = title

    def plot(self, save_to: str = None):
        plt.scatter(self.features_y_pred, self.features_y)

        plt.plot([self.min, self.max], [self.min, self.max], "--")

        plt.xlabel("$y_{pred}$")
        plt.ylabel("$y$")

        if save_to is not None:
            plt.savefig(Path(save_to).expanduser(), bbox_inches="tight")
        else:
            plt.title(self.title)

        plt.show()


@torch.no_grad()
def q_q_plots_for_dataset(config_path: str, model_path: str, image_path: str, seg_map_image_path: str):
    config = ConfigParser.read(Path(config_path).expanduser())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model_class = UNet if config["model"]["type"] == "AutoencoderTrainer" else UNetWithEmbedding
    model = model_class(**config["model"]["params"])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = AutoencoderDataset.load_from_label_file(
        config["dataset"]["test_label"],
        original_image_path=Path(image_path).expanduser(),
        seg_map_image_path=Path(seg_map_image_path).expanduser(),
        preload_percentage=config["dataset"]["params"]["preload_percentage"],
        device=device
    )

    feature_calculators = {
        "Mean": torch.mean,
        "Median": torch.median,
        # "Std": torch.std,
        "Variance": torch.var
    }

    features_y = dict()
    features_y_pred = dict()

    for key in feature_calculators:
        features_y[key] = list()
        features_y_pred[key] = list()

    model.eval()

    save_path = "q-q/autoencoder" if config["model"]["type"] == "AutoencoderTrainer" else "q-q/diffusion_model"

    for x, y in tqdm(dataset):
        seg_mask = torch.cat([x[3][None, :]] * 3, dim=0)

        if config["model"]["type"] == "AutoencoderTrainer":
            y_pred = model.forward(x[None, :])[0].cpu().detach()
        elif config["model"]["type"] == "DiffusionModelTrainer":
            diffusion_betas = torch.linspace(
                config["training"]["diffusion_beta_1"],
                config["training"]["diffusion_beta_capital_t"],
                config["training"]["diffusion_steps"]
            ).to(device)

            y_pred = test_prediction(
                model,
                x[:3][None, :],
                seg_mask[None, :],
                config["training"]["diffusion_steps"],
                config["evaluation"]["harmonization_steps"],
                diffusion_betas,
                device,
                show_tqdm=False
            ).cpu().detach()
        else:
            raise NotImplementedError("The given model is not supported")

        for feature_name, calculator in feature_calculators.items():
            features_y[feature_name].append(calculator(y.cpu()[seg_mask != 0]).item())
            features_y_pred[feature_name].append(calculator(y_pred[seg_mask != 0]).item())

    for feature_name in feature_calculators:
        QQPlot(features_y[feature_name], features_y_pred[feature_name], feature_name).plot(save_to=f"{save_path}/{feature_name}.pdf")


if __name__ == '__main__':
    q_q_plots_for_dataset(
        "../configs/autoencoder.yaml",
        "../evaluation/autoencoder/best_model.pt",
        "~/Documents/data/aml/original128png",
        "~/Documents/data/aml/seg_mask128png"
    )
    # q_q_plots_for_dataset(
    #     "../configs/diffusion_model.yaml",
    #     "../evaluation/diffusion_model/best_model.pt",
    #     "~/Documents/data/aml/original128png",
    #     "~/Documents/data/aml/seg_mask128png"
    # )
