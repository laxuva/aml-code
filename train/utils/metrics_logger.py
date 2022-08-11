from pathlib import Path
import json

import numpy as np


class MetricsLogger:
    def __init__(self, *metrics_to_track: str, out_path: str = ".", save_each_epoch: bool = True):
        self.metrics_to_track = metrics_to_track

        self.current_epoch_metrics = dict()
        self.epoch_mean_per_metric = dict()

        for metric_name in metrics_to_track:
            self.current_epoch_metrics[metric_name] = list()
            self.epoch_mean_per_metric[metric_name] = list()

        self.epoch = 0
        self.out_path = Path(out_path)
        self.save_each_epoch = save_each_epoch

    def log(self, key: str, value):
        self.current_epoch_metrics[key].append(value)

    def end_epoch(self):
        self.epoch += 1

        for metric_name in self.metrics_to_track:
            self.epoch_mean_per_metric[metric_name].append(np.mean(self.current_epoch_metrics[metric_name]))
            self.current_epoch_metrics[metric_name] = list()

        print(f"[epoch {self.epoch}] " +
              "; ".join(f"{metric_name}: {self.get_last(metric_name)}"
                        for metric_name in self.metrics_to_track)
              )

        if self.save_each_epoch:
            self.save()

    def get_last(self, metric_name: str) -> float:
        return self.epoch_mean_per_metric[metric_name][-1]

    def save(self):
        with open(self.out_path.joinpath("train_info.json"), "w") as f:
            f.write(json.dumps({
                metric_name: self.epoch_mean_per_metric[metric_name] for metric_name in self.metrics_to_track
            }))
