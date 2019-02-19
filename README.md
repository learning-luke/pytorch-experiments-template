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
- A [notebook](notebooks/plot-results.ipynb) to exemplify the use of the simple logging features

### Example runs
1. To train a 4 layer deep (2 convs of widths 64 and 128, 2 fully connected layers of widths 256, 10) CNN on MNIST:  
  `python train.py -en MNIST-cnn-4 -data MNIST -model cnn -fil 64 128 -str 2 2 -ker 3 3 -lin 256`
2. To train a preactivation resnet, depth 18, on [Cinic-10-enlarged](https://github.com/BayesWatch/cinic-10):  
  `python train.py -en cinic10enlarged-preact-18 -data Cinic-10-enlarged -model preact_resnet -dep 18`
3. To train a wide resnet, depth 40 and widen factor 2, on Cifar-10:  
  `python train.py -en Cifar-10-wresnet-40-2 -data Cifar-10 -model wresnet -dep 40 -wf 2`

---

Note: the resnets and densenet are adapted from [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and the wide resnet from [https://github.com/xternalz/WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch). 



