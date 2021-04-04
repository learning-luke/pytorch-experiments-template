import os
import torch
import numpy as np

from rich.console import Console
from rich.table import Table


def compute_accuracy(logits, targets):
    acc = (targets == logits.argmax(-1)).float().detach().cpu().numpy()
    return float(np.mean(acc))


class MetricTracker:
    def __init__(
        self,
        tracker_name,
        metrics_to_track=None,
        load=True,
        path="",
    ):
        if metrics_to_track is None:
            self.metrics_to_track = {
                "cross_entropy": lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(),
                "accuracy": compute_accuracy,
            }
        else:
            self.metrics_to_track = metrics_to_track

        self.metrics_to_receive = [
            "epochs",
            "iterations",
            "data/s",
            "batch/s",
            "epoch time",
        ]
        self.metrics = {}

        self.tracker_name = tracker_name

        if not path.endswith(".pt"):
            self.path = f"{path}.pt"
        else:
            self.path = path

        #  set up the metrics dictionary as combo of track and receive
        for k in self.metrics_to_receive:
            self.metrics[k] = []

        for k, _ in self.metrics_to_track.items():
            self.metrics[k] = []

        if load and os.path.isfile(path):
            metrics_from_file = torch.load(self.path)
            self.metrics = metrics_from_file

        # set up a table for printing
        self.table = Table(title=self.tracker_name)
        for key in self.metrics.keys():
            self.table.add_column(key)

    def push(self, epoch, iteration, data, batch_time, epoch_time, logits, targets):
        self.metrics["epochs"].append(epoch)
        self.metrics["iterations"].append(iteration)
        self.metrics["data/s"].append(data)
        self.metrics["batch/s"].append(batch_time)
        self.metrics["epoch time"].append(epoch_time)

        for k, fnc in self.metrics_to_track.items():
            self.metrics[k].append(fnc(logits, targets))

        self.update_metric_table()

    def update_metric_table(
        self,
    ):
        row = [
            f"{value[-1]:0.4f}" if isinstance(value[-1], float) else f"{value[-1]}"
            for key, value in self.metrics.items()
        ]
        self.table.add_row(*row)

    def save(self, overwrite=True):
        if not self.path.endswith(".pt"):
            self.path = f"{self.path}.pt"

        if overwrite:
            if os.path.exists(self.path):
                os.remove(self.path)

        torch.save(self.metrics, self.path)
