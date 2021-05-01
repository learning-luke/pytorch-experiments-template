import argparse
import glob
import os
import random
import tarfile
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from pytorch_model_summary import summary
from rich.live import Live
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from datasets.dataset_loading_hub import load_dataset
from models import model_zoo
from utils.arg_parsing import add_extra_option_args, process_args
from utils.gpu_selection_utils import select_devices
from utils.metric_tracking import MetricTracker, compute_accuracy
from utils.pretty_progress_reporting import PrettyProgressReporter
from utils.storage import build_experiment_folder, save_checkpoint, restore_model


def get_base_argument_parser():
    parser = argparse.ArgumentParser()
    # data and I/O
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--num_gpus_to_use",
        type=int,
        default=0,
        help="The number of GPUs to use, use 0 to enable CPU",
    )

    parser.add_argument(
        "--gpu_ids_to_use",
        type=str,
        default=None,
        help="The IDs of the exact GPUs to use, this bypasses num_gpus_to_use if used",
    )

    parser.add_argument("--dataset_name", type=str, default="cifar10")
    parser.add_argument("--data_filepath", type=str, default="../data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)

    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=False, dest="resume", action="store_true")
    parser.add_argument("--test", dest="test", default=True, action="store_true")

    # logging
    parser.add_argument("--experiment_name", type=str, default="dev")
    parser.add_argument("--logs_path", type=str, default="log")
    parser.add_argument("--filepath_to_arguments_json_config", type=str, default=None)

    parser.add_argument("--save_top_n_val_models", type=int, default=1)
    parser.add_argument("--val_set_percentage", type=float, default=0.1)
    # optimization
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="CosineAnnealing",
        help="Scheduler for learning rate annealing: CosineAnnealing | MultiStep",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=[60, 120, 160],
        help="Multi step scheduler annealing milestones",
    )
    parser.add_argument("--optim", type=str, default="Adam", help="Optimizer?")

    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser = add_extra_option_args(parser)

    return parser


def housekeeping():
    argument_parser = get_base_argument_parser()
    args = process_args(argument_parser)

    if args.gpu_ids_to_use is None:
        select_devices(
            args.num_gpus_to_use,
            max_load=args.max_gpu_selection_load,
            max_memory=args.max_gpu_selection_memory,
            exclude_gpu_ids=args.excude_gpu_list,
        )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids_to_use.replace(" ", ",")

    saved_models_filepath, logs_filepath, images_filepath = build_experiment_folder(
        experiment_name=args.experiment_name, log_path=args.logs_path
    )

    args.saved_models_filepath = saved_models_filepath
    args.logs_filepath = logs_filepath
    args.images_filepath = images_filepath

    # Determinism Seeding can be annoying in pytorch at the moment.
    # Based on my experience, the below means of seeding allows for deterministic
    # experimentation.
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

    # Always save a snapshot of the current state of the code. I've found this helps
    # immensely if you find that one of your many experiments was actually quite good
    # but you forgot what you did

    snapshot_filename = f"{saved_models_filepath}/snapshot.tar.gz"
    filetypes_to_include = [".py"]
    all_files = []
    for _ in filetypes_to_include:
        all_files += glob.glob("**/*.py", recursive=True)
    with tarfile.open(snapshot_filename, "w:gz") as tar:
        for file in all_files:
            tar.add(file)

    return args


def train(epoch, data_loader, model, metric_tracker, progress_reporter):
    model = model.train()
    epoch_start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        logits, features = model(inputs)

        loss = criterion(input=logits, target=targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_tracker.push(
            epoch,
            batch_idx,
            epoch_start_time,
            logits,
            targets,
        )

        progress_reporter.update_progress_iter(
            metric_tracker=metric_tracker, reset=batch_idx == 0
        )

    metric_tracker.update_per_epoch_table()


def eval(epoch, data_loader, model, metric_tracker, progress_reporter):
    epoch_start_time = time.time()
    model = model.eval()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        logits, features = model(inputs)

        metric_tracker.push(
            epoch,
            batch_idx,
            epoch_start_time,
            logits,
            targets,
        )

        progress_reporter.update_progress_iter(
            metric_tracker=metric_tracker, reset=batch_idx == 0
        )

    metric_tracker.update_per_epoch_table()


if __name__ == "__main__":
    #############################################HOUSE-KEEPING##########################
    # Set variables, file keeping, logic, etc.
    args = housekeeping()

    #############################################DATA-LOADING###########################

    (
        train_set_loader,
        val_set_loader,
        test_set_loader,
        train_set,
        val_set,
        test_set,
        data_shape,
        num_classes,
    ) = load_dataset(
        args.dataset_name,
        args.data_filepath,
        batch_size=args.batch_size,
        test_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        download=True,
        val_set_percentage=args.val_set_percentage,
    )
    args.model.num_classes = num_classes

    #############################################MODEL-DEFINITION#######################

    model = model_zoo[args.model.type](**args.model)

    # alternatively one can define a model directly as follows
    # ```
    # model = ResNet18(num_classes=num_classes, variant=args.dataset_name)
    # .to(args.device)
    # ```

    print(
        summary(
            model,
            torch.zeros([1] + list(data_shape)),
            show_input=True,
            show_hierarchical=True,
        )
    )

    model = model.to(args.device)

    if args.num_gpus_to_use > 1:
        model = nn.parallel.DataParallel(model)

    #############################################OPTIMISATION###########################

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
            params, lr=args.learning_rate, weight_decay=args.weight_decay
        )

    if args.scheduler == "CosineAnnealing":
        scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=args.max_epochs, eta_min=0
        )
    else:
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

    #############################################RESTART/RESTORE/RESUME#################

    restore_fields = {
        "model": model if not isinstance(model, nn.DataParallel) else model.module,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    start_epoch = 0
    if args.resume:
        resume_epoch = restore_model(
            restore_fields,
            filename=args.experiment_name,
            directory=args.saved_models_filepath,
            epoch_idx=None,
        )

        if resume_epoch == -1:
            raise IOError(
                f"Failed to load from {args.saved_models_filepath}/ckpt.pth.tar, which "
                f"probably means that the "
                f"latest checkpoint is missing, please remove the --resume flag to "
                f"try training from scratch "
            )
        else:
            start_epoch = resume_epoch + 1

    #############################################METRIC-TRACKING########################

    metrics_to_track = {
        "cross_entropy": lambda x, y: torch.nn.CrossEntropyLoss()(x, y).item(),
        "accuracy": compute_accuracy,
    }
    metric_tracker_train, metric_tracker_val, metric_tracker_test = (
        MetricTracker(
            metrics_to_track=metrics_to_track,
            load=True if start_epoch > 0 else False,
            path=f"{args.logs_filepath}/metrics_{tracker_name}.pt",
            tracker_name=tracker_name,
        )
        for tracker_name in ["training", "validation", "testing"]
    )

    #############################################PROGRESS-REPORTING#####################

    progress_reporter = PrettyProgressReporter(
        metric_trackers=(metric_tracker_train, metric_tracker_val, metric_tracker_test),
        set_size_list=(
            len(train_set_loader),
            len(val_set_loader),
            len(test_set_loader),
        ),
        max_epochs=args.max_epochs,
        start_epoch=start_epoch,
        test=args.test,
    )

    #############################################TRAINING###############################

    train_iterations = 0

    with Live(
        progress_reporter.progress_table, refresh_per_second=1
    ) as interface_panel:
        for epoch in range(start_epoch, args.max_epochs):
            train(
                epoch,
                data_loader=train_set_loader,
                model=model,
                metric_tracker=metric_tracker_train,
                progress_reporter=progress_reporter,
            )
            with torch.no_grad():
                eval(
                    epoch,
                    data_loader=val_set_loader,
                    model=model,
                    metric_tracker=metric_tracker_val,
                    progress_reporter=progress_reporter,
                )

            scheduler.step()

            metric_tracker_train.plot(path=f"{args.images_filepath}/train/metrics.png")
            metric_tracker_val.plot(path=f"{args.images_filepath}/val/metrics.png")
            metric_tracker_train.save()
            metric_tracker_val.save()

            # ########################################################### Saving models

            state = {
                "args": args,
                "epoch": epoch,
                "model": model.state_dict()
                if not isinstance(model, nn.DataParallel)
                else model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }

            metric_tracker_val.refresh_best_n_epoch_models(
                directory=args.saved_models_filepath,
                filename=args.experiment_name,
                metric_name="accuracy",
                n=args.save_top_n_val_models,
                bigger_is_better=True,
                current_epoch_idx=epoch,
                current_epoch_state=state,
            )

            save_checkpoint(
                state=state,
                directory=args.saved_models_filepath,
                filename=args.experiment_name,
                is_best=False,
            )

        #############################################TESTING############################

        if args.test:
            if args.val_set_percentage >= 0.0:
                top_n_model_idx = metric_tracker_val.get_best_n_epochs_for_metric(
                    metric_name="accuracy", n=1, bigger_is_better=True
                )[0]
                resume_epoch = restore_model(
                    restore_fields,
                    filename=args.experiment_name,
                    directory=args.saved_models_filepath,
                    epoch_idx=top_n_model_idx,
                )

            eval(
                epoch,
                model=model,
                data_loader=test_set_loader,
                metric_tracker=metric_tracker_test,
                progress_reporter=progress_reporter,
            )

            metric_tracker_test.save()
