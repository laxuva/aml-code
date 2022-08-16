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

    def autoencoder_loss(self, x, y, y_pred, dis_pred):
        return 0.75 * self.compute_reconstruction_loss(y_pred, y, x) + 0.25 * (1 - torch.mean(dis_pred))

    def discriminator_loss(self, dis_pred_true, dis_pred_fake):
        return 1 - (torch.mean(dis_pred_true) - torch.mean(dis_pred_fake))

    def sample_real_images(self, batch_size, dataset):
        len_dataset = len(dataset)
        return torch.cat([dataset[idx][1][None, :] for idx in np.random.permutation(range(len_dataset))[:batch_size]])

    def sample_fake_images(self, batch_size, dataset):
        len_dataset = len(dataset)
        x = torch.cat([dataset[idx][0][None, :] for idx in np.random.permutation(range(len_dataset))[:batch_size]])
        seg_map = self.get_segmentation_map(x)
        return self.model.forward(x).detach() * seg_map + x[:, :3]

    def get_segmentation_map(self, x):
        batch_size, _, w, h = x.shape
        seg_map = x[:, 3].reshape(batch_size, 1, w, h)
        return torch.cat((seg_map, seg_map, seg_map), dim=1) != 0

    def compute_reconstruction_loss(self, y_pred, y, x):
        seg_map = self.get_segmentation_map(x)
        return self.reconstruction_loss_function(y_pred[seg_map], y[seg_map])

    def configure_optimizers(self):
        return [self.optimizer, self.lr_scheduler]

    def do_discriminator_steps(self, x):
        self.optimizer_discriminator.zero_grad()

        self.model.eval()

        discriminator_losses = list()

        for k in range(10):
            dis_pred_fake = self.discriminator.forward(self.sample_fake_images(x.shape[0], self.train_dataset))
            dis_pred_true = self.discriminator.forward(self.sample_real_images(x.shape[0], self.train_dataset))
            loss_discriminator = self.discriminator_loss(dis_pred_true, dis_pred_fake)
            discriminator_losses.append(loss_discriminator.cpu().detach())

            loss_discriminator.backward()
            self.optimizer_discriminator.step()

            self.discriminator.clip_weights()

        self.model.train()

        return torch.mean(torch.tensor(discriminator_losses)).item()

    def train_on_batch(self, x, y) -> float:
        discriminator_loss = self.do_discriminator_steps(x)

        self.optimizer.zero_grad()

        y_pred = self.model.forward(x)
        seg_map = self.get_segmentation_map(x)

        dis_pred = self.discriminator.forward(x[:, 0:3] + y_pred * seg_map)  # TODO just add it where it is needed!
        loss_autoencoder = self.autoencoder_loss(x, y, y_pred, dis_pred)

        loss_autoencoder.backward()
        self.optimizer.step()

        self.metrics_logger.log("train_loss", loss_autoencoder.cpu().detach().item())
        self.metrics_logger.log("train_loss_d", discriminator_loss)

        return loss_autoencoder.cpu().detach().item()

    def validation_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        dis_pred_fake = self.discriminator.forward(x[:, 0:3] + y_pred)
        dis_pred_true = self.discriminator.forward(self.sample_real_images(x.shape[0], self.val_dataset))

        loss_autoencoder = self.autoencoder_loss(x, y, y_pred, dis_pred_fake)
        loss_discriminator = self.discriminator_loss(dis_pred_true, dis_pred_fake)

        self.metrics_logger.log("val_loss", loss_autoencoder.cpu().detach().item())
        self.metrics_logger.log("val_loss_d", loss_discriminator.cpu().detach().item())

        return loss_autoencoder

    def end_epoch(self):
        self.metrics_logger.end_epoch()
        self.lr_scheduler.step()
        self.lr_scheduler_discriminator.step()

    def get_model(self):
        return self.model
