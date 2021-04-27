#!/bin/sh
export CUDA_HOME=/opt/cuda-10.2.89_440_33/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

cd ../../
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:
python $execution_script$ --filepath_to_arguments_json_config experiment_files/experiment_config_files/$experiment_config$ "$@"