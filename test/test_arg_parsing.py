"""
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
"""


def test_merge_json_with_mutable_arguments():
    pass
