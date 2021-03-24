import argparse
import json
from collections import namedtuple

from utils.storage import load_dict_from_json


def merge_json_with_mutable_arguments(json_file_path, arg_dict):
    config_dict = load_dict_from_json(json_file_path)
    for key in config_dict.keys():
        if "num_workers" in key:
            pass
        elif "ngpus" in key:
            pass
        else:
            arg_dict[key] = config_dict[key]

    return config_dict


def parse_args(verbose=True):
    """
    Argument parser
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser()
    # data and I/O

    parser.add_argument("-w", "--num_workers", type=int, default=1)
    parser.add_argument(
        "-ngpus",
        "--num_gpus_to_use",
        type=int,
        default=0,
        help="The number of GPUs to use, use 0 to enable CPU",
    )

    parser.add_argument("-d", "--dataset_name", type=str, default="cifar10")
    parser.add_argument(
        "-path", "--data_filepath", type=str, default="../data/Cifar-10"
    )
    parser.add_argument("-m.batch", "--model.batch_size", type=int, default=256)
    parser.add_argument(
        "-m.evalbatch", "--model.eval_batch_size", type=int, default=100
    )
    parser.add_argument("-x", "--max_epochs", type=int, default=200)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument(
        "-resume", "--resume", default=False, dest="resume", action="store_true"
    )
    parser.add_argument(
        "-test", "--test", dest="test", default=True, action="store_true"
    )

    # logging
    parser.add_argument("-exp", "--experiment_name", type=str, default="dev")
    parser.add_argument("-o", "--logs_path", type=str, default="log")

    parser.add_argument(
        "-json_config", "--filepath_to_arguments_json_config", type=str, default=None
    )

    parser.add_argument(
        "-save", "--save", dest="save", default=True, action="store_true"
    )
    parser.add_argument(
        "-nosave", "--nosave", dest="save", default=True, action="store_false"
    )

    # model
    parser.add_argument("-m.type", "--model.type", type=str, default="resnet")
    parser.add_argument("-m.dep", "--model.depth", type=int, default=18)
    parser.add_argument("-m.wf", "--model.widen_factor", type=int, default=1)
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

    args = parser.parse_args()

    if args.filepath_to_arguments_json_config is not None:
        args_dict = merge_json_with_mutable_arguments(
            json_file_path=args.filepath_to_arguments_json_config, arg_dict=vars(args)
        )
        ArgsObject = namedtuple("ArgsObject", tuple(model_args.keys()))
        args = ArgsObject(**args_dict)

    model_args = {
        key.replace("model.", ""): value
        for key, value in vars(args).items()
        if "model." in key
    }

    ModelArgsObject = namedtuple("ModelArgsObject", tuple(model_args.keys()))
    model_args = ModelArgsObject(**model_args)
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))

    return args, model_args
