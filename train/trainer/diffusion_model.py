from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToPILImage

from network.unet_with_embedding import UNet
from train.trainer.trainer_base import TrainerBase
from train.utils.metrics_logger import MetricsLogger


class DiffusionModelTrainer(TrainerBase):
    def __init__(
            self,
            unet_config: Dict[str, Any],
            train_config: Dict[str, Any],
            device: torch.device = torch.device("cpu")
    ):
        super(DiffusionModelTrainer, self).__init__(device)

        self.model = UNet(**unet_config).to(device)

        self.optimizer = torch.optim.Adam(lr=train_config["learning_rate"], params=self.model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, **train_config["lr_scheduler"])

        try:
            self.loss_function = {
                "L1Loss": torch.nn.L1Loss,
                "MSELoss": torch.nn.MSELoss,
                "SmoothL1Loss": torch.nn.SmoothL1Loss
            }[train_config["loss_function"]]()
        except KeyError:
            raise KeyError(f"The given loss function {train_config['loss_function']} is not supported.")

        self.diffusion_betas = torch.linspace(
            train_config["diffusion_beta_1"],
            train_config["diffusion_beta_capital_t"],
            train_config["diffusion_steps"]
        ).to(self.used_device)

        self.show_sampled_images = train_config["show_sampled_images"]
        self.sampled_images_location = (
            Path(train_config["sampled_images_location"]).expanduser()
            if "sampled_images_location" in train_config
            else None
        )

        if self.sampled_images_location is not None and not self.sampled_images_location.exists():
            self.sampled_images_location.mkdir()

        self.metrics_logger = MetricsLogger("train_loss", "val_loss", out_path=train_config["out_path"])
        self.epoch = 0

    def backward_diffusion_process(self, img, train: bool = True):
        T = len(self.diffusion_betas)

        img = img * 2 - 1  # normalize to range [-1, 1]

        t = torch.randint(0, T, (img.shape[0],), device=self.used_device).long()

        losses = list()

        if train:
            self.optimizer.zero_grad()

        e = torch.normal(0, 1, img.shape).to(self.used_device)

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

    def training_step(self, img, _) -> torch.Tensor:
        loss = self.backward_diffusion_process(img)

        self.metrics_logger.log("train_loss", loss.cpu().detach().item())

        return loss

    def validation_step(self, img, _) -> torch.Tensor:
        loss = self.backward_diffusion_process(img, train=False)

        self.metrics_logger.log("val_loss", loss.cpu().detach().item())

        return loss

    def end_epoch(self):
        super(DiffusionModelTrainer, self).end_epoch()

        self.epoch += 1

        if self.show_sampled_images or self.sampled_images_location is not None:
            self.sample_plot_image()

    @torch.no_grad()
    def sample_plot_image(self):
        num_images = 10
        diffusion_steps = len(self.diffusion_betas)
        step_size = int(diffusion_steps / num_images)
        img_size = 128
        img = torch.randn((1, 3, img_size, img_size), device=self.used_device)
        fig, axes = plt.subplots(1, num_images)
        plt.axis('off')

        alpha_head_t_minus_one = 0

        for t in range(0, len(self.diffusion_betas))[::-1]:
            alpha_head = torch.prod(1 - self.diffusion_betas[:t + 1]).to(self.used_device)
            alpha = 1 - self.diffusion_betas[t].to(self.used_device)
            noise_to_reduce = self.model.forward(img, torch.tensor([t]).to(self.used_device))
            img = 1 / torch.sqrt(alpha) * (
                        img - self.diffusion_betas[t] * noise_to_reduce / torch.sqrt(1 - alpha_head))

            if t > 0:
                z = torch.randn_like(img)
                img += z * torch.sqrt((self.diffusion_betas[t] * (1 - alpha_head_t_minus_one) / (1 - alpha_head)))
            alpha_head_t_minus_one = alpha_head

            if t % step_size == 0:
                idx = len(axes) - 1 - t // step_size
                img_to_show = ToPILImage()(torch.clip(img[0].detach().cpu() / 2 + 0.5, 0, 1))
                axes[idx].imshow(img_to_show)
                axes[idx].set_title(t)
                axes[idx].set_axis_off()

        if self.sampled_images_location is not None:
            plt.savefig(self.sampled_images_location.joinpath(f"epoch_{self.epoch}.pdf"), bbox_inches="tight")
        if self.show_sampled_images:
            plt.suptitle(f"Epoch: {self.epoch}")
            plt.show()
