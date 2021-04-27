import os

import numpy as np
import torch

from rich.progress import TextColumn
from rich.table import Table
import matplotlib.pyplot as plt
import time
from utils.storage import load_metrics_dict_from_pt, save_checkpoint
import seaborn as sns

sns.set()


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
            self.metrics_to_track = {
                "cross_entropy": lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(),
                "accuracy": compute_accuracy,
            }
        else:
            self.metrics_to_track = metrics_to_track

        self.metrics_to_receive = ["epochs", "iter", "iter/s"]
        self.metrics = {}

        self.tracker_name = tracker_name
        self.path = path
        for k in self.metrics_to_receive:
            self.metrics[k] = []
        for k, _ in self.metrics_to_track.items():
            self.metrics[k] = []
        self.metrics["epochs_to_rank"] = {}

        if load and os.path.isfile(path):
            metrics_from_file = load_metrics_dict_from_pt(path=path)
            self.metrics = metrics_from_file

        # set up a table for printing
        self.per_epoch_table = Table(
            title=f"{self.tracker_name.capitalize()} Epoch Summary"
        )
        for key in self.collect_per_epoch().keys():
            self.per_epoch_table.add_column(key)

    def push(self, epoch, iteration, epoch_start_time, logits, targets):
        time_taken = time.time() - epoch_start_time
        self.metrics["epochs"].append(epoch)
        self.metrics["iter"].append(iteration)
        self.metrics["iter/s"].append((iteration + 1) / time_taken)

        for key, metric_function in self.metrics_to_track.items():
            self.metrics[key].append(metric_function(logits, targets))

    def get_metric_text_column(self):

        return TextColumn(
            "\t".join(
                f"{key}: {{task.fields[{key}]}}"
                for key, value in self.metrics.items()
                if (key in self.metrics_to_receive) or (key in self.metrics_to_track)
            ).expandtabs(2)
        )

    def get_current_iteration_metric_text_column_fields(self):
        return {
            key: (
                "None"
                if len(value) == 0
                else (
                    f"{value[-1]:0.2f}" if isinstance(value[-1], float) else value[-1]
                )
            )
            for key, value in self.metrics.items()
            if (key in self.metrics_to_receive) or (key in self.metrics_to_track)
        }

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

        if overwrite and os.path.exists(self.path):
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
                if k not in ["iterations", "epochs"]:
                    for this_epoch in unique_epochs:
                        where_metrics = epochs == this_epoch
                        v_mean = np.mean(v[where_metrics])
                        v_std = np.std(v[where_metrics])
                        epoch_metrics["{}_mean".format(k)].append(v_mean)
                        epoch_metrics["{}_std".format(k)].append(v_std)

        return epoch_metrics

    def get_best_n_epochs_for_metric(self, metric_name, n=1, bigger_is_better=True):
        results_for_all_epochs = np.array(
            self.collect_per_epoch()[f"{metric_name}_mean"]
        )
        sorted_epochs = np.argsort(results_for_all_epochs)

        if bigger_is_better:
            return list(sorted_epochs[-n:])
        else:
            return list(sorted_epochs[:n])

    def refresh_best_n_epoch_models(
        self,
        directory,
        filename,
        metric_name,
        n,
        bigger_is_better,
        current_epoch_idx,
        current_epoch_state,
    ):
        if len(self.metrics["epochs_to_rank"].keys()) < n:
            self.metrics["epochs_to_rank"][current_epoch_idx] = len(
                self.metrics["epochs_to_rank"]
            )

            save_checkpoint(
                state=current_epoch_state,
                is_best=True,
                directory=directory,
                filename=filename,
                epoch_idx=current_epoch_idx,
            )
        else:

            previous_top_n = list(self.metrics["epochs_to_rank"].keys())
            current_top_n = self.get_best_n_epochs_for_metric(
                metric_name=metric_name, n=n, bigger_is_better=bigger_is_better
            )

            if current_top_n != previous_top_n:
                self.metrics["epochs_to_rank"] = {}

                for rank_idx, epoch_idx in enumerate(current_top_n):
                    self.metrics["epochs_to_rank"][epoch_idx] = rank_idx

                epoch_idx_models_to_remove = [
                    idx for idx in previous_top_n if idx not in current_top_n
                ]

                for idx_to_remove in epoch_idx_models_to_remove:
                    os.remove(
                        f"{directory}/epoch_{idx_to_remove}_model_{filename}.ckpt"
                    )

                save_checkpoint(
                    state=current_epoch_state,
                    is_best=True,
                    directory=directory,
                    filename=filename,
                    epoch_idx=current_epoch_idx,
                )

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
