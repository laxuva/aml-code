import json
from typing import List
from pathlib import Path

from train.general_training import train
from utils.config_parser import ConfigParser
from hyperparam_opt.utils.convert_to_std_types import convert_to_std_type


class DiffusionModelWrapper:
    def __init__(
            self,
            learning_rate: int = 0.001,
            lr_scheduler_step_size: int = 25,
            lr_scheduler_gamma: float = 0.5,
            channels_per_depth: List[int] = json.dumps([64, 128, 256, 512, 1024]),
            no_early_stopping: bool = False,
            save_intermediate_results: bool = False,
            out_path: Path = Path(".")
    ):
        self.base_config_file = str(Path(__file__).parent.parent.parent.joinpath("configs").joinpath("diffusion_model.yaml"))
        self.config = dict()

        self.best_state_dict = None

        self.last_val_loss = float("inf")
        self.best_val_loss = float("inf")

        self.no_early_stopping = no_early_stopping
        self.save_intermediate_results = save_intermediate_results
        self.out_path = out_path

        if self.save_intermediate_results:
            with open(self.out_path.joinpath("intermediate_results.json"), "w") as f:
                json.dump([], f)

        self.set_params(learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, channels_per_depth)

    def set_params(self, learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, channels_per_depth):
        self.config = ConfigParser.read(self.base_config_file)
        self.config["training"]["learning_rate"] = learning_rate
        self.config["training"]["lr_scheduler"]["step_size"] = lr_scheduler_step_size
        self.config["training"]["lr_scheduler"]["gamma"] = lr_scheduler_gamma
        self.config["model"]["params"]["channels_per_depth"] = json.loads(channels_per_depth)

        if self.no_early_stopping:
            self.config["training"]["break_criterion"] = self.config["training"]["max_epochs"]

    def get_params(self):
        return {
            "learning_rate": self.config["training"]["learning_rate"],
            "lr_scheduler_step_size": self.config["training"]["lr_scheduler"]["step_size"],
            "lr_scheduler_gamma": self.config["training"]["lr_scheduler"]["gamma"],
            "channels_per_depth": self.config["model"]["params"]["channels_per_depth"]
        }

    def fit_with_params(self, args):
        self.set_params(*args)
        self.fit()
        return self.last_val_loss

    def fit(self):
        state_dict, self.last_val_loss = train(self.config, save_output=False)

        if self.last_val_loss <= self.best_val_loss:
            self.best_val_loss = self.last_val_loss
            self.best_state_dict = state_dict

        if self.save_intermediate_results:
            with open(self.out_path.joinpath("intermediate_results.json"), "r+") as f:
                data = json.load(f)
                data.append({
                    "hyperparameters": {
                        key: convert_to_std_type(param) for key, param in self.get_params().items()
                    },
                    "loss": self.last_val_loss
                })
                f.seek(0)
                json.dump(data, f)
