import os
import torch
import numpy as np


def compute_accuracy(logits, targets):
    acc = (targets == logits.argmax(-1)).float().detach().cpu().numpy()
    return np.mean(acc)


class MetricTracker:
    def __init__(
        self,
        tracker_name,
        metrics_to_track=None,
        load=True,
        path="",
    ):
        if metrics_to_track is None:
            metrics_to_track = {
                "cross_entropy": lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(),
                "accuracy": compute_accuracy,
            }

        self.metrics_to_track = metrics_to_track
        self.tracker_name = tracker_name
        self.metrics = {"epochs": [], "iterations": []}

        if not path.endswith(".pt"):
            self.path = f"{path}.pt"
        else:
            self.path = path

        for k, _ in self.metrics_to_track.items():
            self.metrics[k] = []

        if load and os.path.isfile(path):
            metrics_from_file = torch.load(self.path)
            self.metrics = metrics_from_file

    def push(self, epoch, iteration, logits, targets):
        self.metrics["epochs"].append(epoch)
        self.metrics["iterations"].append(iteration)

        for k, fnc in self.metrics_to_track.items():
            self.metrics[k].append(fnc(logits, targets))

    def get_current_iteration_metric_trace_string(self):
        return "".join(
            [
                (
                    f"{key}: {value[-1]:0.4f}; "
                    if (key != "epochs" and key != "iterations")
                    else ""
                )
                for key, value in self.metrics.items()
            ]
        ).replace("(", "")

    def save(self, overwrite=True):
        if not self.path.endswith(".pt"):
            self.path = f"{self.path}.pt"

        if overwrite:
            if os.path.exists(self.path):
                os.remove(self.path)

        torch.save(self.metrics, self.path)
