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

        self.loss_function = torch.nn.MSELoss()

        self.metrics_logger = MetricsLogger("train_loss", "val_loss")

    def _do_multiple_diffusion_steps(self, img_0, t): # formula 7
        alpha_t = torch.prod(1 - self.diffusion_betas[:t])
        return torch.normal(alpha_t * img_0, (1 - alpha_t))

    def training_new(self, img):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        T = len(self.diffusion_betas)

        img = img * 2 - 1 # zentralization with normal distribution
        img_shape = img.shape
        img_shape = img_shape[1] + 1
        img_with_t = torch.ones(img_shape)
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
            e_pred = self.model.forward(input_for_model)

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

    def backward_diffusion_process(self, img, seg_mask, train: bool = True):
        T = len(self.diffusion_betas)

        seg_mask = torch.cat((seg_mask, seg_mask, seg_mask), dim=1)

        img_t = self._do_multiple_diffusion_steps(img, T)
        losses = list()

        for t in range(T, 0, -1):
            if train:
                self.optimizer.zero_grad()

            img_t_minus_1 = self._do_multiple_diffusion_steps(img, t - 1)

            img_t_minus_1_pred = self.model.forward(img_t)

            # loss = self.loss_function(img_t_minus_1_pred[seg_mask == 0], img_t_minus_1[seg_mask == 0])
            loss = self.loss_function(img_t_minus_1_pred, img_t_minus_1)
            losses.append(loss)

            if train:
                loss.backward()
                self.optimizer.step()

            # img_t in next step
            img_t = img_t_minus_1
            # img_t[seg_mask != 0] = img_t_minus_1_pred[seg_mask != 0].detach()

        if train:
            self.optimizer.zero_grad()

        img_0_pred = self.model.forward(img_t)

        loss = self.loss_function(img_0_pred, img)
        losses.append(loss)

        if train:
            loss.backward()
            self.optimizer.step()

        return torch.mean(torch.Tensor(losses))

    def train_on_batch(self, x, y) -> float:
        return self.training_step(x, y).cpu().detach().item()

    def training_step(self, img, seg_mask) -> torch.Tensor:
        loss = self.training_new(img)

        self.metrics_logger.log("train_loss", loss.cpu().detach().item())

        return loss

    def validation_step(self, img, seg_mask) -> torch.Tensor:
        loss = self.backward_diffusion_process(img, seg_mask, train=False)

        self.metrics_logger.log("val_loss", loss.cpu().detach().item())

        return loss
