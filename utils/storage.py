"""
Storage associated utilities
"""
import numpy as np
import scipy.misc
import shutil
import torch
import scipy
import json
import os
from rich import print
import requests
import tqdm  # progress bar
import os.path


def isfloat(x):
    return isinstance(x, float)


def isint(x):
    return isinstance(x, int)


def save_dict_in_json(path, metrics_dict, overwrite):
    """
    Saves a metrics .json file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param metrics_dict: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath,
    if False adds metrics to existing
    """

    if path.endswith(".json"):
        path = path.replace(".json", "")

    metrics_file_path = path

    if overwrite and os.path.exists(metrics_file_path):
        os.remove(metrics_file_path)

    with open("{}.json".format(metrics_file_path), "w+") as json_file:
        json.dump(metrics_dict, json_file, indent=4, sort_keys=True)


def load_dict_from_json(path):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if not path.endswith(".json"):
        path = "{}.json".format(path)
    with open(path) as json_file:
        metrics_dict = json.load(json_file)

    return metrics_dict


def load_metrics_dict_from_pt(path):
    """
    Loads the metrics in a dictionary.
    :param log_dir: The directory in which the log is saved
    :param metrics_file_name: The name of the metrics file
    :return: A dict with the metrics
    """

    if not path.endswith(".pt"):
        path = "{}.pt".format(path)

    metrics_file_path = path

    return torch.load(metrics_file_path)


def save_metrics_dict_in_pt(path, metrics_dict, overwrite):
    """
    Saves a metrics .pt file with the metrics
    :param log_dir: Directory of log
    :param metrics_file_name: Name of .csv file
    :param metrics_dict: A dict of metrics to add in the file
    :param overwrite: If True overwrites any existing files with the same filepath,
    if False adds metrics to existing
    """
    if not path.endswith(".pt"):
        path = "{}.pt".format(path)

    metrics_file_path = path

    if overwrite and os.path.exists(metrics_file_path):
        os.remove(metrics_file_path)

    torch.save(metrics_dict, metrics_file_path)


def save_checkpoint(state, is_best, directory="", filename="", epoch_idx=None):
    """
    Checkpoint saving utility, to ensure that the checkpoints are saved in the
    right place
    :param state: this is what gets saved.
    :param is_best: if this is the current best model, save a copy of it with a `best_`
    :param directory: where to save
    :param filename: using this filename
    :return: nothing, just save things
    """
    if is_best:
        save_path = f"{directory}/epoch_{epoch_idx}_model_{filename}.ckpt"
    else:
        save_path = f"{directory}/latest_{filename}.ckpt"

    torch.save(state, save_path)


def restore_model(restore_fields, directory, filename, epoch_idx=None, device="cpu"):
    """
    Model restoration. This is built into the experiment framework and
    args.latest_loadpath should contain the path
    to the latest restoration point. This is automatically set in the framework
    :param net: Network to restore weights of
    :param optimizer: sometimes the optimizer also needs to be restored.
    :param args:
    :return: Nothing, only restore the network and optimizer.
    """

    checkpoint_filepath = (
        f"{directory}/latest_{filename}.ckpt"
        if epoch_idx is None
        else f"{directory}/epoch_{epoch_idx}_model_{filename}.ckpt"
    )

    if not os.path.isfile(checkpoint_filepath):
        return -1
    else:
        checkpoint = torch.load(
            checkpoint_filepath,
            map_location=lambda storage, loc: storage,
        )

        for name, field in restore_fields.items():
            field.load_state_dict(checkpoint[name])

        return checkpoint["epoch"]


def build_experiment_folder(experiment_name, log_path, save_images=True):
    """
    An experiment logging folder goes along with each experiment. This builds that
    folder
    :param args: dictionary of arguments
    :return: filepaths for saved models, logs, and images
    """
    saved_models_filepath = os.path.join(
        log_path, experiment_name.replace("%.%", "/"), "saved_models"
    )
    logs_filepath = os.path.join(
        log_path, experiment_name.replace("%.%", "/"), "summary_logs"
    )
    images_filepath = os.path.join(
        log_path, experiment_name.replace("%.%", "/"), "images"
    )

    if not os.path.exists(logs_filepath):
        os.makedirs(logs_filepath)

    if not os.path.exists(saved_models_filepath):
        os.makedirs(saved_models_filepath)

    if not os.path.exists(images_filepath):
        os.makedirs(images_filepath)

    if save_images:
        if not os.path.exists(images_filepath + "/train"):
            os.makedirs(images_filepath + "/train")
        if not os.path.exists(images_filepath + "/val"):
            os.makedirs(images_filepath + "/val")
        if not os.path.exists(images_filepath + "/test"):
            os.makedirs(images_filepath + "/test")

    return saved_models_filepath, logs_filepath, images_filepath


def get_best_performing_epoch_on_target_metric(
    metrics_dict, target_metric, ranking_method=np.argmax
):
    """
    utility for finding best epoch thus far
    :param: metrics_dict: A dictionary containing the collected metrics from which to
    extract the best perf. model epoch
    target_metric:
    ranking_method:
    :return: best epoch, and what the best target metric value was
    """
    best_model_epoch = 0
    best_target_metric = None

    if target_metric in metrics_dict and len(metrics_dict[target_metric]) != 0:
        best_epoch_idx = ranking_method(metrics_dict[target_metric])
        best_model_epoch, best_target_metric = (
            metrics_dict["epoch"][best_epoch_idx],
            metrics_dict[target_metric][best_epoch_idx],
        )

    return best_model_epoch, best_target_metric


def download_file(url, filename=False, verbose=False):
    """
    Download file with progressbar
    __author__ = "github.com/ruxi"
    __license__ = "MIT"
    Usage:
        download_file('http://web4host.net/5MB.zip')
    """
    if not filename:
        local_filename = os.path.join(".", url.split("/")[-1])
    else:
        local_filename = filename
    r = requests.get(url, stream=True)
    file_size = int(r.headers["Content-Length"])
    chunk_size = 1024
    num_bars = int(file_size / chunk_size)
    if verbose:
        print(dict(file_size=file_size))
        print(dict(num_bars=num_bars))

    with open(local_filename, "wb") as fp:
        for chunk in tqdm.tqdm(
            r.iter_content(chunk_size=chunk_size),
            total=num_bars,
            unit="KB",
            desc=local_filename,
            leave=True,  # progressbar stays
        ):
            fp.write(chunk)
    return local_filename
