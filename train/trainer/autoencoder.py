from typing import Dict, Any

import torch

from network.unet import UNet
from train.trainer.trainer_base import TrainerBase
from train.utils.metrics_logger import MetricsLogger


class AutoencoderTrainer(TrainerBase):
    def __init__(
            self,
            unet_config: Dict[str, Any],
            train_config: Dict[str, Any],
            device: torch.device = torch.device("cpu"),
            save_output: bool = True
    ):
        super(AutoencoderTrainer, self).__init__(device, save_output=save_output)
        self.model = UNet(**unet_config).to(device)

        self.loss_function = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(lr=train_config["learning_rate"], params=self.model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **train_config["lr_scheduler"])

        self.metrics_logger = MetricsLogger("train_loss", "val_loss", out_path=train_config["out_path"])

        if "pretrained_model_path" in train_config:
            self.model.load_state_dict(torch.load(train_config["pretrained_model_path"], map_location=device))
            print("Model was loaded.")

    def compute_loss(self, y_pred, y, x):
        batch_size, _, w, h = x.shape
        seg_map = x[:, 3].reshape(batch_size, 1, w, h)
        seg_map = torch.cat((seg_map, seg_map, seg_map), dim=1) != 0
        return self.loss_function(y_pred[seg_map], y[seg_map])

    def training_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        loss = self.compute_loss(y_pred, y, x)

        self.metrics_logger.log("train_loss", loss.cpu().detach().item())

        return loss

    def validation_step(self, x, y) -> torch.Tensor:
        y_pred = self.model.forward(x)

        loss = self.compute_loss(y_pred, y, x)

        self.metrics_logger.log("val_loss", loss.cpu().detach().item())

        return loss
