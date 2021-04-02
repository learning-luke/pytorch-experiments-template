import os
import sys
from unittest.mock import patch


def test_get_arguments_passed_on_command_line():
    from utils.arg_parsing import get_arguments_passed_on_command_line

    # test that the arguments on the command line are extracted properly

    arg_dict = {"thing1": "default1", "thing2": "default2", "thing3": "default3"}

    args_via_cmd_line = ["--thing1", "--thing2", "--thing3"]
    with patch.object(sys, "argv", args_via_cmd_line):
        arguments_passed_to_command_line = get_arguments_passed_on_command_line(
            arg_dict
        )

    assert len(arguments_passed_to_command_line) == 2
    assert [
        s.replace("--", "") for s in args_via_cmd_line[1:]
    ] == arguments_passed_to_command_line


def test_merge_json_with_mutable_arguments():
    import json
    from utils.arg_parsing import merge_json_with_mutable_arguments

    # test that the json args get overwritten

    # dummy arg dict object
    arg_dict = {"three": 5}

    # create a dummy JSON object
    json_dict = {"one": 1, "two": 2, "three": 3, "four": 4}
    json_obj = json.dumps(json_dict)

    with open("dummy.json", "w") as f:
        f.write(json_obj)

    with patch.object(sys, "argv", ["", "--three"]):
        merged = merge_json_with_mutable_arguments("dummy.json", arg_dict)
    os.remove("dummy.json")

    assert merged["one"] == 1
    assert merged["two"] == 2
    assert merged["three"] == 5
