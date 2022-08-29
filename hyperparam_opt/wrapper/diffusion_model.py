import json
from pathlib import Path
from typing import List

from hyperparam_opt.wrapper.base_wrapper import BaseWrapper
from utils.config_parser import ConfigParser


class DiffusionModelWrapper(BaseWrapper):
    def __init__(
            self,
            base_config_file,
            batch_size: int = 32,
            learning_rate: int = 0.001,
            lr_scheduler_step_size: int = 25,
            lr_scheduler_gamma: float = 0.5,
            channels_per_depth: List[int] = json.dumps([64, 128, 256, 512, 1024]),
            no_early_stopping: bool = False,
            save_intermediate_results: bool = False,
            out_path: Path = Path("."),
            n_calls: int = 50
    ):
        super(DiffusionModelWrapper, self).__init__(
            base_config_file, no_early_stopping, save_intermediate_results, out_path, n_calls
        )

        self.set_params(batch_size, learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, channels_per_depth)

    def set_params(self, batch_size, learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, channels_per_depth):
        self.config = ConfigParser.read(self.base_config_file)
        self.config["train_loader"]["batch_size"] = int(batch_size)
        self.config["val_loader"]["batch_size"] = int(batch_size)
        self.config["training"]["learning_rate"] = learning_rate
        self.config["training"]["lr_scheduler"]["step_size"] = lr_scheduler_step_size
        self.config["training"]["lr_scheduler"]["gamma"] = lr_scheduler_gamma
        self.config["model"]["params"]["channels_per_depth"] = json.loads(channels_per_depth)

        if self.no_early_stopping:
            self.config["training"]["break_criterion"] = self.config["training"]["max_epochs"]

    def get_params(self):
        return {
            "batch_size":  self.config["train_loader"]["batch_size"],
            "learning_rate": self.config["training"]["learning_rate"],
            "lr_scheduler_step_size": self.config["training"]["lr_scheduler"]["step_size"],
            "lr_scheduler_gamma": self.config["training"]["lr_scheduler"]["gamma"],
            "channels_per_depth": self.config["model"]["params"]["channels_per_depth"]
        }
