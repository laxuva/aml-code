from typing import Dict, Any

import pytorch_lightning as pl
import torch

from network.segmentation.unet import UNet
from train.utils.metrics_logger import MetricsLogger


class AutoencoderTrainer(pl.LightningModule):
    def __init__(
            self,
            unet_config: Dict[str, Any],
            train_config: Dict[str, Any],
            device: torch.device = torch.device("cpu")
    ):
        super(AutoencoderTrainer, self).__init__()
        self.model = UNet(**unet_config).to(device)

        self.loss_function = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(lr=train_config["learning_rate"], params=self.model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **train_config["lr_scheduler"])

        self.metrics_logger = MetricsLogger("train_loss", "val_loss")

        self.predict_difference = train_config["predict_difference"]

    def configure_optimizers(self):
        return [self.optimizer, self.lr_scheduler]

    def training_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        if self.predict_difference:
            y_pred = y_pred + x

        loss = self.loss_function(y_pred[x == 0], y[x == 0])

        self.metrics_logger.log("train_loss", loss.cpu().detach().item())

        return loss

    def train_on_batch(self, x, y) -> float:
        self.optimizer.zero_grad()

        loss = self.training_step(x, y)
        loss.backward()

        self.optimizer.step()

        return loss.cpu().detach().item()

    def validation_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        if self.predict_difference:
            y_pred = y_pred + x

        loss = self.loss_function(y_pred[x == 0], y[x == 0])

        self.metrics_logger.log("val_loss", loss.cpu().detach().item())

        return loss

    def end_epoch(self):
        self.metrics_logger.end_epoch()
        self.lr_scheduler.step()

    def get_model(self):
        return self.model
