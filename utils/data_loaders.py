import os
import random
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import namedtuple

ImSize = namedtuple('ImSize', ['channels', 'width', 'height'])

class MNISTLoader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.1307],
                                         std=[0.3081])
        self.im_size = ImSize(1, 28, 28)

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self.transform_validate = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def get_data(self, data_loc):
        train_set = datasets.MNIST(data_loc, train=True, download=True,
                                   transform=self.transform_train)
        val_set = datasets.MNIST(data_loc, train=False,
                                 transform=self.transform_validate)
        return train_set, val_set


class CINIC10Loader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        self.im_size = ImSize(3, 32, 32)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(self.im_size.width, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.transform_validate = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def get_data(self, data_loc):
        traindir = os.path.join(data_loc, 'train')
        valdir = os.path.join(data_loc, 'test')

        train_set = datasets.ImageFolder(traindir, self.transform_train)
        val_set = datasets.ImageFolder(valdir, self.transform_validate)

        return train_set, val_set


class CIFAR10Loader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        self.im_size = ImSize(3, 32, 32)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(self.im_size.width, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.transform_validate = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def get_data(self, data_loc):
        train_set = datasets.CIFAR10(root=data_loc,
                                     train=True, download=False, transform=self.transform_train)
        val_set = datasets.CIFAR10(root=data_loc,
                                   train=False, download=False, transform=self.transform_validate)

        return train_set, val_set


class CIFAR100Loader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                         std=[0.2009, 0.1984, 0.2023])
        self.im_size = ImSize(3, 32, 32)
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(self.im_size.width, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,

        ])

        self.transform_validate = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def get_data(self, data_loc):
        train_set = datasets.CIFAR100(root=data_loc,
                                      train=True, download=True, transform=self.transform_train)
        val_set = datasets.CIFAR100(root=data_loc,
                                    train=False, download=True, transform=self.transform_validate)
        return train_set, val_set


class ImageNetLoader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.im_size = ImSize(3, 224, 224)
        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(self.im_size.width),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        self.transform_validate = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.im_size.width),
            transforms.ToTensor(),
            normalize,
        ])

    def get_data(self, data_loc):
        traindir = os.path.join(data_loc, 'train')
        valdir = os.path.join(data_loc, 'val')

        train_set = datasets.ImageFolder(traindir, self.transform_train)
        val_set = datasets.ImageFolder(valdir, self.transform_validate)

        return train_set, val_set

class SimCLRTransform(object):
    """
    credit: https://github.com/sthalles/SimCLR/
    """
    def __init__(self, size, s=1, n_views=2):
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        base_transform = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

def load_dataset(dataset, data_loc, batch_size, workers):
    if dataset == 'mnist':
        dataloader = MNISTLoader()
    elif dataset == 'cinic10':
        dataloader = CINIC10Loader()
    elif dataset == 'cifar10':
        dataloader = CIFAR10Loader()
    elif dataset == 'cifar100':
        dataloader = CIFAR100Loader()
    elif dataset == 'imagenet':
        dataloader = ImageNetLoader()

    # custom augmentations can go here if you want them to apply to any dset
    # e.g. dataloader.transform_train.append(Cutout(...))

    ### e.g. ADD SIMCLR
    dataloader.transform_train = SimCLRTransform(size=dataloader.im_size.width)
    ### 

    train_set, val_set = dataloader.get_data(data_loc)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, val_loader, dataloader.im_size
