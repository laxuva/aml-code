import json
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from hyperparam_opt.utils.convert_to_std_types import convert_to_std_type
from train.general_training import train


class BaseWrapper(ABC):
    def __init__(
            self,
            base_config_file,
            no_early_stopping: bool = False,
            save_intermediate_results: bool = False,
            out_path: Path = Path("."),
            n_calls: int = 50
    ):
        self.base_config_file = base_config_file
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

        self.i = 0
        self.n_calls = n_calls

    @abstractmethod
    def set_params(self, learning_rate, lr_scheduler_step_size, lr_scheduler_gamma, channels_per_depth):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def fit_with_params(self, args):
        print("-" * 50)
        self.i += 1
        print(f"[{self.i}/{self.n_calls}] Fit with parameters:", *args)

        self.set_params(*args)
        self.fit()
        return self.last_val_loss

    def fit(self):
        state_dict, self.last_val_loss = train(self.config, save_output=False)

        if self.last_val_loss <= self.best_val_loss:
            self.best_val_loss = self.last_val_loss
            self.best_state_dict = state_dict

            torch.save(state_dict, self.out_path.joinpath("best_model_weights.pt"))

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
