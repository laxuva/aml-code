from typing import Dict, Any

import pytorch_lightning as pl
import torch

from network.segmentation.unet import UNet
from train.loss.dice_loss import DiceLoss
from train.utils.metrics_logger import MetricsLogger
from metrics.segmentation.iou import iou


class UNetTrainer(pl.LightningModule):
    def __init__(
            self,
            unet_config: Dict[str, Any],
            train_config: Dict[str, Any],
            device: torch.device = torch.device("cpu")
    ):
        super(UNetTrainer, self).__init__()
        self.model = UNet(**unet_config).to(device)

        self.loss_function = DiceLoss()

        self.optimizer = torch.optim.Adam(lr=train_config["learning_rate"], params=self.model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **train_config["lr_scheduler"])

        self.metrics_logger = MetricsLogger("train_loss", "val_loss", "train_iou", "val_iou")

    def configure_optimizers(self):
        return [self.optimizer, self.lr_scheduler]

    def training_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        loss = self.loss_function(y_pred, y)

        self.metrics_logger.log("train_loss", loss.cpu().detach().item())
        self.metrics_logger.log(
            "train_iou",
            iou(y_pred, y, th=0.25).cpu().detach().item()
        )

        return loss

    def train_on_batch(self, x, y) -> float:
        self.optimizer.zero_grad()

        loss = self.training_step(x, y)
        loss.backward()

        self.optimizer.step()

        return loss.cpu().detach().item()

    def validation_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        loss = self.loss_function(y_pred, y)

        self.metrics_logger.log("val_loss", loss.cpu().detach().item())
        self.metrics_logger.log(
            "val_iou",
            iou(y_pred, y, th=0.25).cpu().detach().item()
        )

        return loss

    def end_epoch(self):
        self.metrics_logger.end_epoch()
        self.lr_scheduler.step()

    def get_model(self):
        return self.model
