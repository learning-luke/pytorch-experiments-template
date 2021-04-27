import argparse
import json
from collections import namedtuple, defaultdict
import pprint
from utils.storage import load_dict_from_json
import sys
from rich import print


# 3. priority should be defaults, json file, then any additional command line arguments
def merge_json_with_mutable_arguments(json_file_path, arg_dict):

    config_dict = load_dict_from_json(json_file_path)
    arguments_passed_to_command_line = get_arguments_passed_on_command_line(
        arg_dict=arg_dict
    )
    print(
        "arguments_passed_to_command_line", arguments_passed_to_command_line, sys.argv
    )
    for key in config_dict.keys():
        if key in arguments_passed_to_command_line:
            config_dict[key] = arg_dict[key]

    return config_dict


class DictWithDotNotation(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __repr__(self):
        return "<DictWithDotNotation " + dict.__repr__(self) + ">"


def get_arguments_passed_on_command_line(arg_dict):

    return [
        option.lower()
        for command_line_argument in sys.argv[1:]
        for option in arg_dict.keys()
        if command_line_argument.lower().replace("--", "") == option.lower()
    ]


def add_extra_option_args(parser):
    """
    Argument parser
    :return: parsed arguments
    """
    # model
    parser.add_argument("--model.type", type=str, default="ViT32LastTimeStep")
    parser.add_argument("--model.dropout_rate", type=float, default=0.3)
    parser.add_argument("--model.depth", type=int, default=18)
    parser.add_argument("--model.widen_factor", type=int, default=1)
    parser.add_argument("--model.pretrained", default=False, action="store_true")
    parser.add_argument("--model.model_name_to_download", type=str, default="ViT-B-32")
    parser.add_argument("--model.grid_patch_size", type=int, default=32)
    parser.add_argument("--model.transformer_num_filters", type=int, default=768)
    parser.add_argument("--model.transformer_num_layers", type=int, default=12)
    parser.add_argument("--model.transformer_num_heads", type=int, default=12)

    parser.add_argument(
        "--max_gpu_selection_load",
        type=float,
        default=0.01,
        help="A float between 0 and 1.0 indicating the max percentage of utilization "
        "a GPU must have in order to "
        "be considered "
        "as available for usage",
    )
    parser.add_argument(
        "--max_gpu_selection_memory",
        type=float,
        default=0.01,
        help="A float between 0 and 1.0 indicating the max memory percentage being "
        "used on a GPU in order to "
        "be considered "
        "as available for usage",
    )
    parser.add_argument(
        "--excude_gpu_list",
        type=list,
        default=[],
        help="A list of GPU IDs to exclude from the auto selection process",
    )

    return parser


def process_args(parser):

    args = parser.parse_args()

    if args.filepath_to_arguments_json_config is not None:
        args_dict = merge_json_with_mutable_arguments(
            json_file_path=args.filepath_to_arguments_json_config, arg_dict=vars(args)
        )
        args = DictWithDotNotation(args_dict)

    if isinstance(args, argparse.Namespace):
        args = vars(args)

    args_tree_like_structure = {}

    for key, value in args.items():
        if "." in key:
            top_level_key = key.split(".")[0]
            lower_level_key = key.replace(key.split(".")[0] + ".", "")

            if top_level_key in args_tree_like_structure:
                args_tree_like_structure[top_level_key][lower_level_key] = value
            else:
                args_tree_like_structure[top_level_key] = DictWithDotNotation(
                    {lower_level_key: value}
                )

        else:
            args_tree_like_structure[key] = value

    for key, value in args_tree_like_structure.items():
        if isinstance(value, dict):
            args_tree_like_structure[key] = DictWithDotNotation(value)

    args = DictWithDotNotation(args_tree_like_structure)
    arg_summary_string = pprint.pformat(args, indent=4)
    print(arg_summary_string)

    return args
