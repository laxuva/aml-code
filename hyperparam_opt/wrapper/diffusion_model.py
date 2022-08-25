import json
from typing import List
from pathlib import Path

from train.general_training import train
from utils.config_parser import ConfigParser


class DiffusionModelWrapper:
    def __init__(
            self,
            learning_rate: int = 0.001,
            lr_scheduler_step_size: int = 25,
            lr_scheduler_gamma: float = 0.5,
            channels_per_depth: List[int] = json.dumps([64, 128, 256, 512, 1024]),
            no_early_stopping: bool = False
    ):
        self.base_config_file = str(Path(__file__).parent.parent.parent.joinpath("configs").joinpath("diffusion_model.yaml"))
        self.config = dict()
        self.model = None
        self.best_val_loss = float("inf")
        self.no_early_stopping = no_early_stopping

        self.set_params(learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, channels_per_depth)

    def set_params(self, learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, channels_per_depth):
        self.config = ConfigParser.read(self.base_config_file)
        self.config["training"]["learning_rate"] = learning_rate
        self.config["training"]["lr_scheduler"]["step_size"] = lr_scheduler_step_size
        self.config["training"]["lr_scheduler"]["gamma"] = lr_scheduler_gamma
        self.config["model"]["params"]["channels_per_depth"] = json.loads(channels_per_depth)

        if self.no_early_stopping:
            self.config["model"]["break_criterion"] = self.config["model"]["max_epochs"]

    def get_params(self):
        return (
            self.config["training"]["learning_rate"],
            self.config["training"]["lr_scheduler"]["step_size"],
            self.config["training"]["lr_scheduler"]["step_size"],
            self.config["training"]["lr_scheduler"]["gamma"],
            self.config["training"]["loss_function"],
            self.config["model"]["params"]["channels_per_depth"]
        )

    def fit_with_params(self, args):
        self.set_params(*args)
        self.fit()
        return self.best_val_loss

    def fit(self):
        self.model, self.best_val_loss = train(self.config)
