from typing import Optional

import pytorch_lightning as pl
import torch

from train.utils.metrics_logger import MetricsLogger


class TrainerBase(pl.LightningModule):
    def __init__(
            self,
            device: torch.device = torch.device("cpu")
    ):
        super(TrainerBase, self).__init__()

        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        self.used_device = device

        self.metrics_logger: Optional[MetricsLogger] = None

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
