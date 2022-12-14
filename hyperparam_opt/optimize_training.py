import json
from pathlib import Path

import torch
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

from hyperparam_opt.utils.convert_to_std_types import convert_to_std_type
from hyperparam_opt.wrapper import AutoencoderWrapper, DiffusionModelWrapper
from utils.config_parser import ConfigParser


def optimize(base_config_file, out_path: str = "."):
    out_path = Path(out_path).expanduser()

    if not out_path.exists():
        out_path.mkdir()

    wrapper_class = {
        "AutoencoderTrainer": AutoencoderWrapper,
        "DiffusionModelTrainer": DiffusionModelWrapper
    }[ConfigParser.read(base_config_file)["model"]["type"]]

    model = wrapper_class(
        base_config_file,
        no_early_stopping=True,
        save_intermediate_results=True,
        out_path=out_path
    )

    search_space = {
        "batch_size": Categorical([16, 32, 64]),
        "learning_rate": Real(0.00001, 0.1, prior="log-uniform"),
        "lr_scheduler_step_size": Integer(10, 50),
        "lr_scheduler_gamma": Real(0.1, 0.75, prior="uniform"),
        "channels_per_depth": Categorical([
            json.dumps([16, 32, 64, 128]),
            json.dumps([16, 32, 64, 128, 256]),
            json.dumps([32, 64, 128, 256, 512]),
            json.dumps([32, 64, 128, 256]),
        ])
    }

    results = gp_minimize(model.fit_with_params, search_space.values(), n_calls=50)
    print("Best hyperparameters:", results.x)
    print("Best loss:", results.fun)

    with open(out_path.joinpath("results.json"), "w") as f:
        f.write(json.dumps({
            "hyperparams": {
                key: convert_to_std_type(results.x[idx]) for idx, key in enumerate(search_space)
            },
            "loss": float(results.fun)
        }))

    torch.save(model.best_state_dict, out_path.joinpath("best_model_weights.pt"))


if __name__ == '__main__':
    optimize("../configs/autoencoder.yaml", "results")
