import json
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np


def plot_box_plots(hyperparams_intermediate_results: List[Dict[str, Any]], model_name: str):
    print(hyperparams_intermediate_results)

    hyperparams = hyperparams_intermediate_results[0]["hyperparameters"].keys()

    results = dict()

    for hyperparam in hyperparams:
        results[hyperparam] = dict()

        for training_result in hyperparams_intermediate_results:
            used_value = training_result["hyperparameters"][hyperparam]

            if isinstance(used_value, list):
                used_value = str(used_value)
            elif hyperparam == "learning_rate":
                used_value = 10**round(np.log10(used_value))
            elif hyperparam == "lr_scheduler_step_size":
                used_value = int(np.round(used_value / 10) * 10)
            elif hyperparam == "lr_scheduler_gamma":
                used_value = np.round(used_value, 1)

            if used_value not in results[hyperparam]:
                results[hyperparam][used_value] = [training_result["loss"]]
            else:
                results[hyperparam][used_value].append(training_result["loss"])

    for hyperparam in hyperparams:
        values = [results[hyperparam][used_value] for used_value in results[hyperparam]]
        labels = [used_value for used_value in results[hyperparam]]

        if isinstance(labels[0], float) or isinstance(labels[0], int):
            args = np.argsort(labels)
            values = np.array(values, dtype=object)[args].tolist()
            labels = np.array(labels)[args].tolist()

        plt.boxplot(
            values,
            labels=labels
        )
        if hyperparam == "channels_per_depth":
            plt.xticks(rotation=-7)
        plt.ylabel("Best Loss")
        plt.savefig(f"box_plots/{model_name}_{hyperparam}", bbox_inches="tight")
        plt.title(hyperparam)
        plt.show()


if __name__ == '__main__':
    with open("../evaluation/autoencoder/intermediate_results.json") as f:
        plot_box_plots(json.load(f), "autoencoder")
    with open("../evaluation/diffusion_model/intermediate_results.json") as f:
        plot_box_plots(json.load(f), "diffusion_model")
