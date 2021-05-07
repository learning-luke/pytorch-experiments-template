# Pytorch classification experiments template

![Licence](https://img.shields.io/github/license/BayesWatch/pytorch-experiments-template)
![Code formatting](https://camo.githubusercontent.com/d91ed7ac7abbd5a6102cbe988dd8e9ac21bde0a73d97be7603b891ad08ce3479/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f636f64652532307374796c652d626c61636b2d3030303030302e737667)
![Documentation status](https://readthedocs.org/projects/pytorch-experiments-template-docs/badge/?version=latest)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/bayeswatch/pytorch-experiments-template/CI)

This is a template for deep learning projects using PyTorch (particularly for classification).

This template is designed as a fully working experiment starter. That is, simply running `python train.py` will run a small CNN on cifar-10, while handling logging, checkpointing, neat printing to the terminal, datasets, etc. 

For detailed instructions, see the [docs](https://pytorch-experiments-template-docs.readthedocs.io/).

## Installation and setup

### Installation

We encourage the use of `conda` to get setup. Our current preference is for [miniforge](https://github.com/conda-forge/miniforge).

If you are working on the Edinburgh GPU machines you will want to set your conda home to be in scratch space (i.e. start by doing `cd /disk/scratch/`, then `mkdir YOURNAME`, `cd YOURNAME`, and then install conda).

Assuming you already have conda available, installing the dependencies of this template should be as easy as:

```bash
conda env create --name NAME --file environment.yml
```

Where `NAME` is of your choice.

Then:

```bash
conda activate NAME
```

### Using the template

In order to use this template, go to the GitHub repository and click the "Use this template" button:

![](https://pytorch-experiments-template-docs.readthedocs.io/en/latest/_images/use_this_template.png)

Then you will want to clone and `cd` into your new repository.

**Optional extra: running the tests**

You can also set a `datasets` directory that the repository will read from. In order to do this you will need to add a line to your ``.bashrc`` (or whichever shell you are using) that looks like:

```bash
export PYTORCH_DATA_LOC=/disk/scratch_ssd/
```

This, for instance, is the data location I have set on one of our remote machines. I have it set differently on my personal laptop, so I never need to change config files or pass extra args to move between machines.

Assuming you are in the base directory of the repository, you can then run

```bash
pytest
````

to verify that things are working on your machine.


## Overview

When you first download the repository, it should look something like this (+- a few other bits):
```
  ├── datasets
  |   ├── dataset_loading_hub.py
  |   ├── custom_transforms.py
  ├── models
  │   ├── __init__.py
  │   ├── auto_builder_models.py
  │   ├── densenet.py
  │   ├── resnet.py
  │   └── wresnet.py
  ├── notebooks
  │   ├── plot-results.ipynb
  │   └── utils.py
  ├── train.py
  └── utils
      ├── arg_parsing.py
      ├── cinic_utils.py
      ├── gpu_selection_utils.py
      ├── metric_tracking.py
      └── storage.py
```

The goal is for you to be able to do your research without ever needing to touch anything in the `utils` folder. 

The things you might edit:
* `train.py` - where the training logic and hyperparameters live
* `datasets/` - where all the dataset loading happens (including augmentations)
* `models/` - where the models are built (either via `auto_builder_models` or through the `model_zoo`)

All of the extra neat features in the repository live in the `utils/` folder, such as:
* automated metric tracking
* automated gpu selection
* some useful extensions to argparse

Checkpoints and metric files will be saved in `log`.

Configurations for running experiments via `json` files are stored in `experiment_files`.

We also encourage you to write tests for your code and put them in `test/` (although we ourselves are guilty of not staying on top of this).

### Example runs

To train a ResNet-18 on CIFAR100:
```bash
python train.py --experiment_name cifar100_example --dataset CIFAR100 --model.type ResNet18
```

To train ResNet-50 using the PyTorch example [ImageNet training script](https://github.com/pytorch/examples/blob/master/imagenet/main.py):
```bash
python train.py --experiment_name imagenet_default --dataset ImageNet --model.type ResNet50 --num_gpus_to_use 4 --workers 8 --max_epochs 90 --scheduler MultiStep --milestones [30, 60] 
```

See the [docs](https://pytorch-experiments-template-docs.readthedocs.io/en/latest/) for more complex examples!

## Acknowledgements
Note: the resnets and densenet are adapted from [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and the wide resnet from [https://github.com/xternalz/WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch). 
