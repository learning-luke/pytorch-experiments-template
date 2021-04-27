#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:4
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=1-12:00:00
#SBATCH --partition=apollo

export CUDA_HOME=/opt/cuda-9.2.148/

export CUDNN_HOME=/opt/cuDNN-7.0/

export STUDENT_ID=$(whoami)

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}


export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/

mkdir -p ${TMP}/datasets/
export DATASET_DIR=${TMP}/datasets/
# Activate the relevant virtual environment:


source /home/${STUDENT_ID}/miniconda3/bin/activate
conda activate mlp

cd ../../
export DATASET_DIR="datasets/"
# Activate the relevant virtual environment:
python $execution_script$ --filepath_to_arguments_json_config experiment_files/experiment_config_files/$experiment_config$ --num_gpus_to_use $NUM_GPUS