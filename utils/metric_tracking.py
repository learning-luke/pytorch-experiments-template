import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
from utils.storage import load_metrics_dict_from_pt, save_metrics_dict_in_pt

def compute_accuracy(logits, targets):
    acc = (targets == logits.argmax(-1)).float().detach().cpu().numpy()
    return np.mean(acc)


class MetricTracker:
    def __init__(self, metrics_to_track={'cross_entropy': lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(), 'accuracy':compute_accuracy},
                 load=True,
                 path=''):
        self.metrics_to_track = metrics_to_track
        self.metrics = {'epochs':[], 'iterations':[]}
        self.path = path
        for k, _ in metrics_to_track.items():
            self.metrics[k] = []
        if load and os.path.isfile(path):
            metrics_from_file = load_metrics_dict_from_pt(path=path)
            self.metrics = metrics_from_file


    def push(self, epoch, iteration, logits, targets):
        self.metrics['epochs'].append(epoch)
        self.metrics['iterations'].append(iteration)
        for k, fnc in self.metrics_to_track.items():
            self.metrics[k].append(fnc(logits, targets))

    def save(self):
        save_metrics_dict_in_pt(path=self.path, metrics_dict=self.metrics, overwrite=True)

    def collect_per_epoch(self):
        epoch_metrics = {'epochs':[]}
        for k, _ in self.metrics_to_track.items():
            epoch_metrics['{}_mean'.format(k)] = []
            epoch_metrics['{}_std'.format(k)] = []

        epochs = self.metrics['epochs']
        unique_epochs = np.unique(epochs)
        epoch_metrics['epochs'] = unique_epochs

        for k, v in self.metrics.items():
            v = np.array(v)
            if k != 'iterations' and k!= 'epochs':
                for this_epoch in unique_epochs:
                    where_metrics = epochs == this_epoch
                    v_mean = np.mean(v[where_metrics])
                    v_std = np.std(v[where_metrics])
                    epoch_metrics['{}_mean'.format(k)].append(v_mean)
                    epoch_metrics['{}_std'.format(k)].append(v_std)
        return epoch_metrics

    def plot(self, path, plot_std_dev=True):
        epoch_metrics = self.collect_per_epoch()

        x = np.array(epoch_metrics['epochs'])
        keys = [k for k, _ in epoch_metrics.items() if k != 'epochs']
        reduced_keys = []

        for key in keys:
            reduced_key = key.replace('_mean', '').replace('_std', '')
            if reduced_key not in reduced_keys:
                reduced_keys.append(reduced_key)
        num_axes = len(reduced_keys)
        nrow = 2
        ncol = int(np.ceil(num_axes/nrow))
        fig = plt.figure(figsize=(5 * nrow, 5 * ncol))
        for pi, key in enumerate(reduced_keys):
            ax = fig.add_subplot(ncol, nrow, pi+1)
            y_mean = np.array(epoch_metrics[key+'_mean'])
            y_std = np.array(epoch_metrics[key+'_std'])
            if plot_std_dev:
                ax.fill_between(x, y_mean-y_std, y_mean+y_std, np.ones_like(x)==1, color='g', alpha=0.1)
            ax.plot(x, y_mean, 'g-', alpha=0.9)
            ax.set_ylabel(key)
            ax.set_xlabel('epochs')
        fig.tight_layout()
        fig.savefig(path, dpi=100)
        plt.close(fig)
        del(fig)




