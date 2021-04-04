# Pytorch classification experiments template

Template for deep learning projects using pytorch and doing classification.

This template is designed as a fully working experiment starter. That is, simply running `python train.py` will run a small CNN on cifar-10, while handling logging, checkpointing, neat printing to the terminal, datasets, etc. 

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
  - cifar-10
  - cifar-100
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

### Folder/file structure
`train.py`: main training code, run this  
&rarr;&rarr;&rarr; `models`: neural networks  
&rarr;&rarr;&rarr; `notebooks`: notebooks for plotting results  
&rarr;&rarr;&rarr; `utils`: necessary utilities, including experiment admin and datasets  

### Example runs
1. To train the default CNN on cifar-10:  
	`python train.py`
2. To train a 4 layer deep (2 convs of widths 64 and 128, 2 fully connected layers of widths 256, 10) CNN on MNIST:  
    `python train.py -en MNIST-cnn-4 -data MNIST -model cnn -fil 64 128 -str 2 2 -ker 3 3 -lin 256`
3. To train a preactivation resnet, depth 18, on [Cinic-10-enlarged](https://github.com/BayesWatch/cinic-10):  
    `python train.py -en cinic10enlarged-preact-18 -data Cinic-10-enlarged -model preact_resnet -dep 18`
4. To train the same model using the random horizontal flips, random crops (with padding 4) and cutout:  
    `python train.py -en cinic10enlarged-preact-18 -data Cinic-10-enlarged -model preact_resnet -dep 18 -aug random_h_flip random_crop cutout`
5. To train a wide resnet, depth 40 and widen factor 2, on cifar-10:  
    `python train.py -en cifar-10-wresnet-40-2 -data cifar-10 -model wresnet -dep 40 -wf 2`


### Arguments

- **-data, --dataset (str)**  
   - Which dataset to use 
   - 'cifar-10', 'cifar-100', 'Cinic-10', 'Cinic-10-enlarged', 'Fashion-MNIST, 'MNIST'  
- **-norm, --dataset_norm_type (str)**  
   - How to normalize data 
   - 'sandardize' --> mean of zero, standard deviation of one  
   - 'zeroone' --> image range of [0, 1]  
- **-batch, --batch_size (int)**  
   - Batch Size  
- **-tbatch, --test_batch_size (int)**  
   - Test Batch Size  
- **-x, --max_epochs (int)**  
   - How many epochs to run in total  
- **-s, --seed (int)**  
   - Random seed to use for reproducibility  
- **-aug, --data_aug ([string])**  
   - List of Data augmentation to apply  
   - 'random_h_flip', 'random_v_vlip', 'color_jitter', 'affine', 'random_crop', 'random_order', 'cutout  
   - *NOTE*, if applying 'affine', the following three arguments must be given:  'random_rot_D', 'random_scale_S1_S2', 'random_sheer_P', where D defines the maximum rotation (in degrees), S1 and S2 define the lower and upper bounds for random scaling (between [0, 1]), and P defines the maximum sheer rotation (in degrees).  
- **-en, --experiment_name (str)**
   - Experiment name
- **-o, --logs_path (str)**
   - Directory to save log files, check points, and any images
- **-resume, --resume (int)**
   - Resume training from latest point in training. This is effectively a bool and will be False if resume is zero
- **-save, --save (int)**  
   - Save checkpoint files? This is effectively a bool and will be False if resume is zero
- **-saveimg, --save_images (int)**  
   - Create a folder for saved images in log directory? This is effectively a bool and will be False if resume is zero
- **-model, --model (int)**
   - Which model to train
   - 'resnet', 'preact_resnet', 'densenet', 'wresnet', 'cnn'
- **-dep, --depth (int)**
   - ResNet depth
   - For resnet, options are: 18, 34, 50, 101
   - For preact_resnet, options are: 18, 34, 50, 101
   - For densenet, options are: 121, 161, 169, 201
   - For wresnet, (depth - 4) % 6 = 0
- **-wf, --widen_factor (int)**
   - Wide resnet widen factor
- **-act, --activation**
   - Activation function for CNN, not relevant to resnets
- **-fil --filters ([int])**
   - Filter list for CNN
- **-str, --strides ([int])**
   - Strides list for CNN
- **-ker, --kernel_sizes ([int])**
   - Kernel size list for CNN
- **-lin, --linear_widths ([int])**
   - Additional linear layer widths. If empty, cnn goes from conv outs to single linear layer
- **-bn, --use_batch_norm (int)**
   - Use batch norm for CNN. This is effectively a bool and will be False if resume is zero
- **-l, --learning_rate (float)**
   - Base learning rate
- **-sched, --scheduler (str)**
   - Scheduler for learning rate annealing
   - 'CosineAnnealing','MultiStep'
- **-mile, --milestones ([int])**
   - Milstones (epochs) for multi step scheduler annealing
- **-optim, --optim (str)**
   - Optimizer to use
   - 'Adam', 'SGD'
- **-wd, --weight_decay (float)**
   - Weight decay value
- **-mom, --momentum**
   - Momentum multiplier

### Installation Guide
Simply clone (`git clone git@github.com:learning-luke/overfit-aware-networks.git`) or download this repo and [run one of the commands](#Example-runs).

Additionally, the following are the required packages.

#### Requirements
- [Pytorch](https://pytorch.org/) and torchvision
- [tqdm](https://pypi.org/project/tqdm/)
- [scipy](https://www.scipy.org/)

The suggested and easiest way to get this working quickly is to install [miniconda](https://conda.io/en/latest/miniconda.html) and then run the following two commands:  
	`conda create -n deep-learning`  
	`conda install -n deep-learning pytorch torchvision cudatoolkit=9.0 tqdm scipy -c anaconda -c pytorch`  
This assumes you will be using a GPU version. Once installation is complete, activate the enviroment with:  
	`source activate deep-learning ` 
and then run this repo's code. 



---

Note: the resnets and densenet are adapted from [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) and the wide resnet from [https://github.com/xternalz/WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch). 



