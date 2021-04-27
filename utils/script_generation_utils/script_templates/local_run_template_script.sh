#!/bin/sh

cd ../../
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:
python $execution_script$ --filepath_to_arguments_json_config experiment_files/experiment_config_files/$experiment_config$ "$@"