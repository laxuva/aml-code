from typing import Dict, Any

import pytorch_lightning as pl
import torch

from network.segmentation.unet import UNet
from train.loss.dice_loss import DiceLoss


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

    def configure_optimizers(self):
        return [self.optimizer, self.lr_scheduler]

    def training_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        loss = self.loss_function(y_pred, y)

        return loss

    def train_on_batch(self, x, y) -> float:
        self.optimizer.zero_grad()

        loss = self.training_step(x, y)
        loss.backward()

        self.optimizer.step()

        return loss.cpu().detach().item()

    def validation_step(self, x, y) -> float:
        y_pred = self.model.forward(x)

        loss = self.loss_function(y_pred, y)

        return loss.cpu().detach().item()

    def get_model(self):
        return self.model
