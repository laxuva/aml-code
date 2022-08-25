import json

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer

from hyperparam_opt.wrapper.diffusion_model import DiffusionModelWrapper


def optimize():
    model = DiffusionModelWrapper()

    search_space = {
        "learning_rate": Real(0.00001, 0.1, prior="log-uniform"),
        "lr_scheduler_step_size": Integer(10, 50),
        "lr_scheduler_gamma": Real(0.1, 0.75, prior="uniform"),
        "channels_per_depth": Categorical([
            json.dumps([64, 128, 256, 512, 1024]),
            json.dumps([32, 64, 128, 256, 512]),
            json.dumps([64, 128, 256, 512]),
            json.dumps([32, 64, 128, 256]),
        ])
    }

    results = gp_minimize(model.fit_with_params, search_space.values(), n_calls=10)
    print("Best hyperparameters:", results.x)
    print("Best loss:", results.fun)

    with open("./results.json", "w") as f:
        f.write(json.dumps({
            "hyperparams": {
                key: float(results.x[idx]) if is_non_std_type(results.x[idx]) else results.x[idx]
                for idx, key in enumerate(search_space)
            },
            "loss": float(results.fun)
        }))


def is_non_std_type(value):
    return isinstance(value, np.float64) or isinstance(value, np.int64)


if __name__ == '__main__':
    optimize()
