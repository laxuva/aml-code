from typing import Dict, Any

import pytorch_lightning as pl
import torch

from network.segmentation.unet_with_embedding import UNet
from train.utils.metrics_logger import MetricsLogger


class GeneralTrainer(pl.LightningModule):
    def __init__(
            self,
            unet_config: Dict[str, Any],
            train_config: Dict[str, Any],
            device: torch.device = torch.device("cpu")
    ):
        super(GeneralTrainer, self).__init__()
        self.model = UNet(**unet_config).to(device)

        self.optimizer = torch.optim.Adam(lr=train_config["learning_rate"], params=self.model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **train_config["lr_scheduler"])

        self.metrics_logger: MetricsLogger = None

    def configure_optimizers(self):
        return [self.optimizer, self.lr_scheduler]

    def train_on_batch(self, x, y) -> float:
        self.optimizer.zero_grad()

        loss = self.training_step(x, y)
        loss.backward()

        self.optimizer.step()

        return loss.cpu().detach().item()

    def end_epoch(self):
        self.metrics_logger.end_epoch()
        self.lr_scheduler.step()

    def get_model(self):
        return self.model
