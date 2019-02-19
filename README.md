# Pytorch classification experiments template

Template for deep learning projects using pytorch and doing classification.

This template is designed as a fully working experiment starter. That is, simply running `python train.py` will run a small CNN on Cifar-10, while handling logging, checkpointing, neat printing to the terminal, datasets, etc. 

Some notable features include:

- Immediate usability of the following models:
  - A small/shallow CNN
  - A standard Resnet
  - A preactivation Resnet
  - A wide Resnet
  - A densenet
- Immediate usability of the following datasets:
  - [Cinic-10](https://github.com/BayesWatch/cinic-10)
  - [Cinic-10](https://github.com/BayesWatch/cinic-10)-enlarged (i.e., Cinic-10's train+validation as the train set)
  - Cifar-10
  - Cifar-100
  - MNIST
  - Fashion-MNIST
- Built in logging and progress bars
- Built in and extensive data augmentation, including:
  - Random horizontal flips
  - Random crops
  - Cutout
  - Random limited rotations
  - Random scaling
  - Random shearing
  - Colour jitter
- A notebook to exemplify the use of the simple logging features

---

Note: the resnets and densenet are adapted from [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and the wide resnet from [https://github.com/xternalz/WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch). 



