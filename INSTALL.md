To use the Pytorch Experiment Template, one needs to set up a conda environment with the right dependencies. 
This can be achieved as follows:
1. Point your web-browser to [miniforge](https://github.com/conda-forge/miniforge)
2. Find and download a suitable mini forge installer for your OS. The next few steps assume Linux and MacOS
3. Go to the directory in which you have downloaded the minforge installer and run:```bash <mini-forge installer>```
4. Follow the setup to install the environment. 
5. Once done, activate conda using ```conda activate```, note that you might need to resource your .bashrc using 
   ```source ~/.bashrc``` to get access to conda if you are using the same terminal as the one used to install conda.
   
6. Create a new conda environment using ```conda create -n pytorch-experiments python=3.7```
7. Once done, run ```conda install git tqdm rich matplotlib regex ftfy requests seaborn GPUtil scipy;``` Follow through the 
   process to finish installing these packages.
   
8.Then go to the [Pytorch installation page](https://pytorch.org/) and run the appropriate commands to install pytorch 
on your machine. The command is not given here as it depends on your OS, CPU/GPU type and CUDA version (if any). 
This one will take a while.

### Note: PyTorch >= 1.8.0 is required for proper usage of this codebase

9. You should now be able to run the default resnet18, cifar10 experiment using 
   ```python train.py```. 
   
If you have any problems or further questions please post on the slack channel 
"#pytorch-debug"
