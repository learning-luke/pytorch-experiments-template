import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from rich.console import Console
from rich.progress import TextColumn
from rich.table import Table
import matplotlib.pyplot as plt

def compute_accuracy(logits, targets):
    acc = (targets == logits.argmax(-1)).float().detach().cpu().numpy()
    return float(np.mean(acc)) * 100

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
        else:
            self.metrics_to_track = metrics_to_track

        self.metrics_to_receive = [
            "epochs",
            "iter",
            "data/s",
            "iter/s",
            "epoch ETA",
        ]
        self.metrics = {}

        self.tracker_name = tracker_name
        self.metrics = {"epochs": [], "iterations": []}
        self.path = path
        for k, _ in metrics_to_track.items():
            self.metrics[k] = []
        if load and os.path.isfile(path):
            metrics_from_file = load_metrics_dict_from_pt(path=path)
            self.metrics = metrics_from_file

        # set up a table for printing
        self.per_epoch_table = Table(title=f'{self.tracker_name.capitalize()} Epoch Summary')
        for key in self.collect_per_epoch().keys():
            self.per_epoch_table.add_column(key)

    def push(self, epoch, iteration, data, batch_time, epoch_time, logits, targets):
        self.metrics["epochs"].append(epoch)
        self.metrics["iter"].append(iteration)
        self.metrics["data/s"].append(1./data)
        self.metrics["iter/s"].append(1./batch_time)
        self.metrics["epoch ETA"].append(epoch_time)

        for key, metric_function in self.metrics_to_track.items():
            self.metrics[key].append(metric_function(logits, targets))


    def get_metric_text_column(self):

        return TextColumn(
            "\t".join(
                [
                    f"{key}: {{task.fields[{key}]}}"
                    for key, value in self.metrics.items()
                ]
            ).expandtabs(2)
        )

    def get_current_iteration_metric_text_column_fields(self):
        return {
            key: 'None' if len(value) == 0
            else (f"{value[-1]:0.2f}" if isinstance(value[-1], float)
            else value[-1]) for key, value in self.metrics.items()
        }

    def get_current_iteration_metric_trace_string(self):
        return "".join(
            [
                (
                    "{}: {:0.4f}; ".format(key, value[-1])
                    if isinstance(value[-1], float)
                    else ""
                )
                for key, value in self.metrics.items()
            ]
        ).replace("(", "")

    def update_per_epoch_table(
        self,
    ):
        row = [
            f"{value[-1]:0.3f}" if isinstance(value[-1], float) else f"{value[-1]}"
            for key, value in self.collect_per_epoch().items()
        ]
        self.per_epoch_table.add_row(*row)

    def save(self, overwrite=True):
        if not self.path.endswith(".pt"):
            self.path = f"{self.path}.pt"

        if overwrite:
            if os.path.exists(self.path):
                os.remove(self.path)

        torch.save(self.metrics, self.path)


    def collect_per_epoch(self):
        epoch_metrics = {"epochs": []}
        for k, _ in self.metrics_to_track.items():
            epoch_metrics["{}_mean".format(k)] = []
            epoch_metrics["{}_std".format(k)] = []

        epochs = self.metrics["epochs"]
        unique_epochs = np.unique(epochs)
        epoch_metrics["epochs"] = unique_epochs

        for k, v in self.metrics.items():
            if k in self.metrics_to_track:
                v = np.array(v)
                if k != "iterations" and k != "epochs":
                    for this_epoch in unique_epochs:
                        where_metrics = epochs == this_epoch
                        v_mean = np.mean(v[where_metrics])
                        v_std = np.std(v[where_metrics])
                        epoch_metrics["{}_mean".format(k)].append(v_mean)
                        epoch_metrics["{}_std".format(k)].append(v_std)

        return epoch_metrics

    def get_best_epoch_for_metric(self, metric_name, evaluation_metric=np.argmax):
        return evaluation_metric(self.collect_per_epoch()[metric_name])

    def plot(self, path, plot_std_dev=True):
        epoch_metrics = self.collect_per_epoch()

        x = np.array(epoch_metrics["epochs"])
        keys = [k for k, _ in epoch_metrics.items() if k != "epochs"]
        reduced_keys = []

        for key in keys:
            reduced_key = key.replace("_mean", "").replace("_std", "")
            if reduced_key not in reduced_keys:
                reduced_keys.append(reduced_key)
        num_axes = len(reduced_keys)
        nrow = 2
        ncol = int(np.ceil(num_axes / nrow))
        fig = plt.figure(figsize=(5 * nrow, 5 * ncol))
        for pi, key in enumerate(reduced_keys):
            ax = fig.add_subplot(ncol, nrow, pi + 1)
            y_mean = np.array(epoch_metrics[key + "_mean"])
            y_std = np.array(epoch_metrics[key + "_std"])
            if plot_std_dev:
                ax.fill_between(
                    x,
                    y_mean - y_std,
                    y_mean + y_std,
                    np.ones_like(x) == 1,
                    color="g",
                    alpha=0.1,
                )
            ax.plot(x, y_mean, "g-", alpha=0.9)
            ax.set_ylabel(key)
            ax.set_xlabel("epochs")
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        del fig
