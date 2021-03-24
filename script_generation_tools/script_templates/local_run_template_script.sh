#!/bin/sh

cd ..
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:

export NUM_GPUS=$1

echo $NUM_GPUS

python $execution_script$ --filepath_to_arguments_json_config experiment_config_files/$experiment_config$ -ngpus $NUM_GPUS -w 4