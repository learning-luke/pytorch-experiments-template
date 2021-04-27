import dataclasses
from copy import deepcopy, copy


def generate_hyperparameter_combination_dicts(hyperparameter_config):

    combinations = []

    for key, values in dataclasses.asdict(hyperparameter_config).items():
        temp_copy = deepcopy(combinations)
        combinations = []

        for value in values:
            if not temp_copy:
                combinations.append([value])
            else:
                temp_copy_2 = deepcopy(temp_copy)
                for item in temp_copy_2:
                    item += [value]

                combinations.extend(temp_copy_2)

    return [
        {
            key.replace("dot", "."): value
            for key, value in zip(
                list(dataclasses.asdict(hyperparameter_config).keys()), values
            )
        }
        for values in combinations
    ]


def generate_hyperparameter_search_experiment_configs(
    hyperparameter_combinations_dicts, default_variable_dict
):

    configs_list = []

    for hyperparameter_config_dict in hyperparameter_combinations_dicts:
        cur_config = copy(default_variable_dict)
        for key, value in hyperparameter_config_dict.items():
            cur_config[key] = value
        configs_list.append((cur_config, hyperparameter_config_dict))

    return configs_list


def fill_template(script_text, config):
    for key, value in config.items():
        script_text = script_text.replace("${}$".format(key), str(value))
    return script_text


def load_template(filepath):
    with open(filepath, mode="r") as filereader:
        template = filereader.read()

    return template


def write_text_to_file(text, filepath):
    with open(filepath, mode="w") as filewrite:
        filewrite.write(text)
