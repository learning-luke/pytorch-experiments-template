import os
import shutil
from dataclasses import dataclass

from utils.arg_parsing import add_extra_option_args, process_args
from utils.script_generation_utils.config_utils import (
    generate_hyperparameter_combination_dicts,
    generate_hyperparameter_search_experiment_configs,
)
from utils.storage import save_dict_in_json
from train import get_base_argument_parser

# 4. TODO Antreas: implement:
#  a. grid search (done)
#  b. random search (not done)


@dataclass
class HyperparameterSearchConfig:
    modeldotbatch_size: list
    learning_rate: list


argument_parser = get_base_argument_parser()
default_variable_dict = process_args(argument_parser)


hyperparameter_config = HyperparameterSearchConfig(
    modeldotbatch_size=[64, 128], learning_rate=[0.001]
)

config_list = []
experiment_config_target_json_dir = "../../experiment_files/experiment_config_files/"

if os.path.exists(experiment_config_target_json_dir):
    shutil.rmtree(experiment_config_target_json_dir)

os.makedirs(experiment_config_target_json_dir)


if not os.path.exists(experiment_config_target_json_dir):
    os.makedirs(experiment_config_target_json_dir)


hyperparameter_combinations_dicts = generate_hyperparameter_combination_dicts(
    hyperparameter_config=hyperparameter_config
)

configs_list = generate_hyperparameter_search_experiment_configs(
    hyperparameter_combinations_dicts=hyperparameter_combinations_dicts,
    default_variable_dict=default_variable_dict,
)

for config_dict, hyperparameter_dict in configs_list:
    config_dict["experiment_name"] = (
        "{}_{}".format(
            config_dict["experiment_name"],
            "_".join(
                [
                    f"{key.replace(',', '-').replace(' ', '')}_"
                    f"{str(value).replace(',', '').replace('[', '').replace(']', '')}"
                    for key, value in hyperparameter_dict.items()
                ]
            ),
        )
        .replace('"', "")
        .replace("'", "")
        .replace(" ", "")
    )

    cluster_script_name = "{}/{}.json".format(
        experiment_config_target_json_dir, config_dict["experiment_name"]
    )
    cluster_script_name = os.path.abspath(cluster_script_name)
    save_dict_in_json(
        metrics_dict=config_dict, path=cluster_script_name, overwrite=True
    )
