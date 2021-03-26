import argparse
import json
import os
import numpy as np
import tqdm

from models.model_selector import ModelSelector
from utils.arg_parsing import parse_args
from utils.gpu_selection_utils import select_devices
from utils.storage import build_experiment_folder, save_checkpoint, restore_model
import random
import glob
import tarfile


parser = argparse.ArgumentParser()
# data and I/O
parser.add_argument("-w", "--num_workers", type=int, default=8)
parser.add_argument(
    "-ngpus",
    "--num_gpus_to_use",
    type=int,
    default=0,
    help="The number of GPUs to use, use 0 to enable CPU",
)

parser.add_argument(
    "-gpu_ids",
    "--gpu_ids_to_use",
    type=str,
    default=None,
    help="The IDs of the exact GPUs to use, this bypasses num_gpus_to_use if used",
)

parser.add_argument("-d", "--dataset_name", type=str, default="cifar10")
parser.add_argument("-path", "--data_filepath", type=str, default="../data/cifar10")
parser.add_argument("-m.batch", "--model.batch_size", type=int, default=256)
parser.add_argument("-m.evalbatch", "--model.eval_batch_size", type=int, default=100)


parser.add_argument("-x", "--max_epochs", type=int, default=200)
parser.add_argument("-s", "--seed", type=int, default=0)
parser.add_argument(
    "-resume", "--resume", default=False, dest="resume", action="store_true"
)
parser.add_argument("-test", "--test", dest="test", default=True, action="store_true")

# logging
parser.add_argument("-exp", "--experiment_name", type=str, default="dev")
parser.add_argument("-o", "--logs_path", type=str, default="log")

parser.add_argument(
    "-json_config", "--filepath_to_arguments_json_config", type=str, default=None
)

parser.add_argument("-save", "--save", dest="save", default=True, action="store_true")
parser.add_argument(
    "-nosave", "--nosave", dest="save", default=True, action="store_false"
)

# model
parser.add_argument("-m.type", "--model.type", type=str, default="resnet")
parser.add_argument("-m.dropout", "--model.dropout_rate", type=float, default=0.3)
# optimization
parser.add_argument("-l", "--learning_rate", type=float, default=0.1)
parser.add_argument(
    "-sched",
    "--scheduler",
    type=str,
    default="MultiStep",
    help="Scheduler for learning rate annealing: CosineAnnealing | MultiStep",
)
parser.add_argument(
    "-mile",
    "--milestones",
    type=int,
    nargs="+",
    default=[60, 120, 160],
    help="Multi step scheduler annealing milestones",
)
parser.add_argument("-optim", "--optim", type=str, default="SGD", help="Optimizer?")

parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4)
parser.add_argument("-mom", "--momentum", type=float, default=0.9)

args = parse_args(parser)
model_args = args.model

if args.gpu_ids_to_use is None:
    select_devices(args.num_gpus_to_use)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids_to_use.replace(' ', ',')

from utils.dataset_loading_hub import load_dataset

from models.wresnet import WideResNet
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from utils.metric_tracking import MetricTracker, compute_accuracy


######################################################################################################### Admin
saved_models_filepath, logs_filepath, images_filepath = build_experiment_folder(
    experiment_name=args.experiment_name, log_path=args.logs_path
)

######################################################################################################### Data

train_set_loader, val_set_loader, train_set, val_set, data_shape = load_dataset(
    args.dataset_name,
    args.data_filepath,
    batch_size=model_args.batch_size,
    test_batch_size=model_args.eval_batch_size,
    num_workers=args.num_workers,
    download=True,
    test=False,
)

_, test_set_loader, _, test_set, _ = load_dataset(
    args.dataset_name,
    args.data_filepath,
    batch_size=model_args.batch_size,
    test_batch_size=model_args.eval_batch_size,
    num_workers=args.num_workers,
    download=True,
    test=args.test,
)

######################################################################################################### Determinism
# Seeding can be annoying in pytorch at the moment. Based on my experience, the below means of seeding
# allows for deterministic experimentation.
torch.manual_seed(args.seed)
np.random.seed(args.seed)  # set seed
random.seed(args.seed)
device = (
    torch.cuda.current_device()
    if torch.cuda.is_available() and args.num_gpus_to_use > 0
    else "cpu"
)
args.device = device
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

# Always save a snapshot of the current state of the code. I've found this helps immensely if you find that one of your
# many experiments was actually quite good but you forgot what you did

snapshot_filename = "{}/snapshot.tar.gz".format(saved_models_filepath)
filetypes_to_include = [".py"]
all_files = []
for filetype in filetypes_to_include:
    all_files += glob.glob("**/*.py", recursive=True)
with tarfile.open(snapshot_filename, "w:gz") as tar:
    for file in all_files:
        tar.add(file)

######################################################################################################### Model

num_classes = 100 if args.dataset_name.lower() == "cifar100" else 10
model_selector = ModelSelector(input_shape=(2, 32, 32, 3), num_classes=num_classes)
model = model_selector.select(model_type=model_args.type, args=model_args).to(device)

if args.num_gpus_to_use > 1:
    model = nn.parallel.DistributedDataParallel(
        model
    )  # more efficient version of DataParallel

model = model.to(device)


######################################################################################################### Optimisation

params = model.parameters()
criterion = nn.CrossEntropyLoss()

if args.optim.lower() == "sgd":
    optimizer = optim.SGD(
        params,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
else:
    optimizer = optim.Adam(
        params, lr=args.learning_rate, amsgrad=True, weight_decay=args.weight_decay
    )

if args.scheduler == "CosineAnnealing":
    scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=args.max_epochs, eta_min=0)
else:
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

######################################################################################################### Restoring

restore_fields = {
    "model": model,
    "optimizer": optimizer,
    "scheduler": scheduler,
}

start_epoch = 0
if args.resume:
    resume_epoch = restore_model(restore_fields, path=saved_models_filepath)
    if resume_epoch == -1:
        print("Failed to load from {}/ckpt.pth.tar".format(saved_models_filepath))
    else:
        start_epoch = resume_epoch + 1

######################################################################################################### Metric

metrics_to_track = {
    "cross_entropy": lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(),
    "accuracy": compute_accuracy,
}
metric_tracker_train = MetricTracker(
    metrics_to_track=metrics_to_track,
    load=True if start_epoch > 0 else False,
    path="{}/metrics_train.pt".format(logs_filepath),
    tracker_name="training",
)

metric_tracker_val = MetricTracker(
    metrics_to_track=metrics_to_track,
    load=True if start_epoch > 0 else False,
    path="{}/metrics_val.pt".format(logs_filepath),
    tracker_name="validation",
)

metric_tracker_test = MetricTracker(
    metrics_to_track=metrics_to_track,
    load=True if start_epoch > 0 else False,
    path="{}/metrics_test.pt".format(logs_filepath),
    tracker_name="testing",
)

######################################################################################################### Training


def train_iter(metric_tracker, model, x, y, iteration, epoch, set_name):
    inputs, targets = x.to(device), y.to(device)

    model = model.train()

    logits, features = model(inputs)

    loss = criterion(input=logits, target=targets)
    metric_tracker.push(epoch, iteration, logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    log_string = "{}, {}: {}; {}".format(
        args.experiment_name,
        set_name,
        iteration,
        "".join(
            [
                (
                    "{}: {:0.4f}; ".format(key, value[-1])
                    if (key != "epochs" and key != "iterations")
                    else ""
                )
                for key, value in metric_tracker.metrics.items()
            ]
        ),
    )

    return log_string


def eval_iter(metric_tracker, model, x, y, iteration, epoch, set_name):
    x, targets = x.to(device), y.to(device)

    model = model.eval()

    logits, features = model(x)

    metric_tracker.push(epoch, iteration, logits, targets)

    log_string = "{}, {}: {}; {}".format(
        args.experiment_name,
        set_name,
        iteration,
        "".join(
            [
                (
                    "{}: {:0.4f}; ".format(key, value[-1])
                    if (key != "epochs" and key != "iterations")
                    else ""
                )
                for key, value in metric_tracker.metrics.items()
            ]
        ),
    )

    return log_string


train_iterations = 0


def run_epoch(epoch, model, training, metric_tracker, data_loader):
    training_iterations = epoch * len(data_loader)

    with tqdm.tqdm(initial=0, total=len(data_loader), smoothing=0) as pbar:

        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if training:
                log_string = train_iter(
                    model=model,
                    x=inputs,
                    y=targets,
                    iteration=training_iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,
                    metric_tracker=metric_tracker,
                )
                training_iterations += 1
            else:
                log_string = eval_iter(
                    model=model,
                    x=inputs,
                    y=targets,
                    iteration=training_iterations,
                    epoch=epoch,
                    set_name=metric_tracker.tracker_name,
                    metric_tracker=metric_tracker,
                )

            pbar.set_description(log_string)
            pbar.update()


if __name__ == "__main__":
    with tqdm.tqdm(initial=start_epoch, total=args.max_epochs) as epoch_pbar:
        for epoch in range(start_epoch, args.max_epochs):

            run_epoch(
                epoch,
                data_loader=train_set_loader,
                model=model,
                training=True,
                metric_tracker=metric_tracker_train,
            )
            run_epoch(
                epoch,
                data_loader=val_set_loader,
                model=model,
                training=False,
                metric_tracker=metric_tracker_val,
            )
            scheduler.step()

            metric_tracker_train.plot(
                path="{}/train/metrics.png".format(images_filepath)
            )
            metric_tracker_val.plot(path="{}/val/metrics.png".format(images_filepath))
            metric_tracker_train.save()
            metric_tracker_val.save()

            ################################################################################ Saving models
            if args.save:
                state = {
                    "args": args,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }

                current_epoch_filename = "{}_ckpt.pth.tar".format(epoch)
                latest_epoch_filename = "latest_ckpt.pth.tar".format(epoch)

                epoch_pbar.set_description(
                    "Saving at {}/{}".format(
                        saved_models_filepath, current_epoch_filename
                    )
                )

                save_checkpoint(
                    state=state,
                    directory=saved_models_filepath,
                    filename=current_epoch_filename,
                    is_best=False,
                )

                epoch_pbar.set_description(
                    "Saving at {}/{}".format(
                        saved_models_filepath, latest_epoch_filename
                    )
                )

                save_checkpoint(
                    state=state,
                    directory=saved_models_filepath,
                    filename=latest_epoch_filename,
                    is_best=False,
                )
            ############################################################################################################

            epoch_pbar.set_description("")
            epoch_pbar.update(1)

        best_epoch_val_model = metric_tracker_val.get_best_epoch_for_metric(
            evaluation_metric=np.argmax, metric_name="accuracy_mean"
        )
        resume_epoch = restore_model(
            restore_fields, path=saved_models_filepath, epoch=best_epoch_val_model
        )

        run_epoch(
            epoch,
            model=model,
            training=False,
            data_loader=test_set_loader,
            metric_tracker=metric_tracker_test,
        )

        metric_tracker_test.save()
