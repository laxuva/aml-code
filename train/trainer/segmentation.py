from typing import Dict, Any

import torch

from metrics.segmentation.iou import iou
from network.unet import UNet
from train.loss.dice_loss import DiceLoss
from train.trainer.trainer_base import TrainerBase
from train.utils.metrics_logger import MetricsLogger


class SegmentationTrainer(TrainerBase):
    def __init__(
            self,
            unet_config: Dict[str, Any],
            train_config: Dict[str, Any],
            device: torch.device = torch.device("cpu"),
            save_output: bool = True
    ):
        super(SegmentationTrainer, self).__init__(device, save_output=save_output)
        self.model = UNet(**unet_config).to(device)

        self.loss_function = DiceLoss()

        self.optimizer = torch.optim.Adam(lr=train_config["learning_rate"], params=self.model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **train_config["lr_scheduler"])

        self.metrics_logger = MetricsLogger(
            "train_loss", "val_loss", "train_iou", "val_iou", out_path=train_config["out_path"]
        )
        self.iou_th = train_config["iou_th"]

    def training_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        loss = self.loss_function(y_pred, y)

        self.metrics_logger.log("train_loss", loss.cpu().detach().item())
        self.metrics_logger.log(
            "train_iou",
            iou(y_pred, y, th=self.iou_th).cpu().detach().item()
        )

        return loss

    def validation_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        loss = self.loss_function(y_pred, y)

        self.metrics_logger.log("val_loss", loss.cpu().detach().item())
        self.metrics_logger.log(
            "val_iou",
            iou(y_pred, y, th=self.iou_th).cpu().detach().item()
        )

        return loss
