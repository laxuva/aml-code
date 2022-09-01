import json

import numpy as np


def get_best_hyperparams(intermediate_results: str):
    with open(intermediate_results, "r") as f:
        results = json.load(f)

        idx = np.argmin([r["loss"] for r in results])

        return results[idx]


if __name__ == '__main__':
    print(get_best_hyperparams("../../evaluation/autoencoder/intermediate_results.json"))
