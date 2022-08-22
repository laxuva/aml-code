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

        self.metrics_logger = MetricsLogger("train_loss", "val_loss")

    def _do_multiple_diffusion_steps(self, img_0, t): # formula 7
        alpha_t = torch.prod(1 - self.diffusion_betas[:t])
        return torch.normal(alpha_t * img_0, (1 - alpha_t))

    def training_new(self, img):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        T = len(self.diffusion_betas)

        img = img * 2 - 1 # zentralization with normal distribution
        img_shape = list(img.shape)
        img_shape[1] = img_shape[1] + 1
        img_with_t = torch.ones(img_shape).to(device)
        losses = list()
        for t in range(1, T+1):
            self.optimizer.zero_grad()

            e = torch.normal(0, 1, img.shape).to(device)
            alpha = torch.prod(1 - self.diffusion_betas[:t])
            input_for_model = torch.clip(torch.sqrt(alpha) * img + torch.sqrt(1-alpha)*e, -1, 1)
            img_with_t[:,:3,:,:] = input_for_model
            img_with_t[:,3,:,:] = alpha
            # input_for_model = torch.clip(alpha * img + torch.sqrt(1 - alpha) * e,
            #                              -1,
            #                              1)
            e_pred = self.model.forward(img_with_t)

            # if t in range(T-5,T):
            #     print(self.diffusion_betas)
            #     print(alpha)
            #     plt.imshow((torch.permute(input_for_model[0], (1, 2, 0)).cpu().detach() + 1) * 0.5)
            #     plt.show()
            #     plt.imshow((torch.permute(self._do_multiple_diffusion_steps(img,t)[0], (1, 2, 0)).cpu().detach() + 1) * 0.5)
            #     plt.show()
            #
            #     plt.imshow((torch.permute(e[0],
            #                               (1, 2, 0)).cpu().detach() + 1) * 0.5)
            #     plt.show()
            #
            #     plt.imshow((torch.permute(e_pred[0], (1, 2, 0)).cpu().detach() + 1) * 0.5)
            #     plt.show()

            loss = self.loss_function(e, e_pred)
            losses.append(loss)

            loss.backward()
            self.optimizer.step()

        return torch.mean(torch.tensor(losses))

    def backward_diffusion_process(self, img, train: bool = True):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        T = len(self.diffusion_betas)

        img = img * 2 - 1  # zentralization with normal distribution
        img_shape = list(img.shape)
        img_shape[1] = img_shape[1] + 1
        img_with_t = torch.ones(img_shape).to(device)

        t = torch.randint(0,
                          T,
                          (img_shape[0],),
                          device=device).long()

        losses = list()

        if train:
            self.optimizer.zero_grad()

        e = torch.normal(0,
                         1,
                         img.shape).to(device)
        # alpha = torch.prod(1 - self.diffusion_betas[:t])
        for idx, t_b in enumerate(t):
            alpha = torch.prod(1 - self.diffusion_betas[:t_b])
            input_for_model = torch.clip(torch.sqrt(alpha) * img[idx] + torch.sqrt(1 - alpha) * e[idx], -1, 1)
            img_with_t[idx, :3, :, :] = input_for_model
            img_with_t[idx, 3, :, :] = alpha

        e_pred = self.model.forward(img_with_t)

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
