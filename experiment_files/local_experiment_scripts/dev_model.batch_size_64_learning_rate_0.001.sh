#!/bin/sh

cd ../../
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:
python train.py --filepath_to_arguments_json_config experiment_files/experiment_config_files/dev_model.batch_size_64_learning_rate_0.001.json "$@"