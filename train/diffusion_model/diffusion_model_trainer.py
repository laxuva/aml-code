from typing import Dict, Any

import matplotlib.pyplot as plt
import torch

from train.general_trainer import GeneralTrainer
from train.utils.metrics_logger import MetricsLogger


class DiffusionModelTrainer(GeneralTrainer):
    def __init__(
            self,
            unet_config: Dict[str, Any],
            train_config: Dict[str, Any],
            device: torch.device = torch.device("cpu")
    ):
        super(DiffusionModelTrainer, self).__init__(unet_config, train_config, device)

        self.diffusion_betas = torch.linspace(
            train_config["diffusion_beta_1"],
            train_config["diffusion_beta_capital_t"],
            train_config["diffusion_steps"]
        ).to(device)

        # self.loss_function = torch.nn.MSELoss()
        self.loss_function = torch.nn.L1Loss()
        # self.loss_function = torch.nn.SmoothL1Loss()

        self.metrics_logger = MetricsLogger("train_loss", "val_loss")

    def backward_diffusion_process(self, img, train: bool = True):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        T = len(self.diffusion_betas)

        img = img * 2 - 1  # centralization with normal distribution
        img_shape = list(img.shape)
        t = torch.randint(0, T, (img_shape[0],), device=device).long()

        losses = list()

        if train:
            self.optimizer.zero_grad()

        e = torch.normal(0,
                         1,
                         img.shape).to(device)
        # alpha = torch.prod(1 - self.diffusion_betas[:t])
        input_for_model = torch.zeros_like(img)
        for idx, t_b in enumerate(t):
            alpha = torch.prod(1 - self.diffusion_betas[:t_b])
            input_for_model[idx] = torch.sqrt(alpha) * img[idx] + torch.sqrt(1 - alpha) * e[idx]

        e_pred = self.model.forward(input_for_model, t)

        loss = self.loss_function(e_pred, e)
        losses.append(loss)

        if train:
            loss.backward()
            self.optimizer.step()

        return torch.mean(torch.Tensor(losses))

    def train_on_batch(self, x, y) -> float:
        return self.training_step(x, y).cpu().detach().item()

    def training_step(self, img, seg_mask) -> torch.Tensor:
        loss = self.backward_diffusion_process(img)

        self.metrics_logger.log("train_loss", loss.cpu().detach().item())

        return loss

    def validation_step(self, img, seg_mask) -> torch.Tensor:
        loss = self.backward_diffusion_process(img, train=False)

        self.metrics_logger.log("val_loss", loss.cpu().detach().item())

        return loss
