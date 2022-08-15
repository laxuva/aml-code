from typing import Dict, Any, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

from network.segmentation.unet import UNet
from network.discriminator import Discriminator
from train.utils.metrics_logger import MetricsLogger


class AdversarialAutoencoderTrainer(pl.LightningModule):
    def __init__(
            self,
            unet_config: Dict[str, Any],
            train_config: Dict[str, Any],
            train_dataset,
            val_dataset,
            device: torch.device = torch.device("cpu"),
    ):
        super(AdversarialAutoencoderTrainer, self).__init__()
        self.model = UNet(**unet_config).to(device)
        self.discriminator = Discriminator(3).to(device)

        self.reconstruction_loss_function = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(lr=train_config["learning_rate"], params=self.model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **train_config["lr_scheduler"])

        self.optimizer_discriminator = torch.optim.Adam(
            lr=train_config["learning_rate"],
            params=self.discriminator.parameters()
        )
        self.lr_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(
            self.optimizer_discriminator,
            **train_config["lr_scheduler"]
        )

        self.metrics_logger = MetricsLogger("train_loss", "val_loss", "train_loss_d", "val_loss_d")

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def sample_real_images(self, batch_size, dataset):
        len_dataset = len(dataset)
        return torch.cat([dataset[idx][1][None, :] for idx in np.random.permutation(range(len_dataset))[:batch_size]])

    def compute_reconstruction_loss(self, y_pred, y, x):
        batch_size, _, w, h = x.shape
        seg_map = x[:, 3].reshape(batch_size, 1, w, h)
        seg_map = torch.cat((seg_map, seg_map, seg_map), dim=1) != 0
        return self.reconstruction_loss_function(y_pred[seg_map], y[seg_map])

    def configure_optimizers(self):
        return [self.optimizer, self.lr_scheduler]

    def train_on_batch(self, x, y) -> float:
        self.optimizer.zero_grad()

        y_pred = self.model.forward(x)

        self.discriminator.eval()
        diss_pred = self.discriminator.forward(x[:, 0:3] + y_pred)  # TODO just add it where it is needed!
        loss_autoencoder = self.compute_reconstruction_loss(y_pred, y, x) + torch.mean(torch.log(1 - diss_pred))

        loss_autoencoder.backward()
        self.optimizer.step()

        self.optimizer_discriminator.zero_grad()

        self.model.eval()
        self.discriminator.train()
        diss_pred = self.discriminator.forward(x[:, 0:3] + y_pred.detach())
        loss_discriminator = torch.mean(torch.log(self.discriminator.forward(y))) + torch.mean(torch.log(1 - diss_pred))

        loss_discriminator.backward()
        self.optimizer_discriminator.step()

        self.model.train()

        self.metrics_logger.log("train_loss", loss_autoencoder.cpu().detach().item())
        self.metrics_logger.log("train_loss_d", loss_discriminator.cpu().detach().item())

        return loss_autoencoder.cpu().detach().item()

    def validation_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        diss_pred = self.discriminator.forward(x[:, 0:3] + y_pred)
        diss_loss = torch.mean(torch.log(1 - diss_pred))

        loss_autoencoder = self.compute_reconstruction_loss(y_pred, y, x) + diss_loss
        loss_discriminator = torch.mean(torch.log(self.discriminator.forward(y))) + diss_loss

        self.metrics_logger.log("val_loss", loss_autoencoder.cpu().detach().item())
        self.metrics_logger.log("val_loss_d", loss_discriminator.cpu().detach().item())

        return loss_autoencoder

    def end_epoch(self):
        self.metrics_logger.end_epoch()
        self.lr_scheduler.step()
        self.lr_scheduler_discriminator.step()

    def get_model(self):
        return self.model
