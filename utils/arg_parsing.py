import argparse
import json
from collections import namedtuple, defaultdict
import pprint
from utils.storage import load_dict_from_json
import sys

# 3. priority should be defaults, json file, then any additional command line arguments
def merge_json_with_mutable_arguments(json_file_path, arg_dict):

    config_dict = load_dict_from_json(json_file_path)
    arguments_passed_to_command_line = get_arguments_passed_on_command_line(
        arg_dict=arg_dict
    )
    print(
        "Arguments_passed_to_command_line: ", arguments_passed_to_command_line, sys.argv
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


def process_args(parser):

    args = parser.parse_args()

    if args.filepath_to_arguments_json_config is not None:
        args_dict = merge_json_with_mutable_arguments(
            json_file_path=args.filepath_to_arguments_json_config, arg_dict=vars(args)
        )
        # convert argparse.Namespace to dictionary with vars()
        # add in the defaults as a starting point, then update with the extra parsed stuff
        vars(args).update(args_dict)
        args = DictWithDotNotation(vars(args))

    if isinstance(args, argparse.Namespace):
        args = vars(args)

    args_tree_like_structure = dict()

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

    args = DictWithDotNotation(args_tree_like_structure)
    pprint.pprint(args, indent=4)

    return args
