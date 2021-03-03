import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import namedtuple
from utils.augmentors import SimCLRTransform
from utils.cinic_utils import enlarge_cinic_10, download_cinic

ImageShape = namedtuple('ImageShape', ['channels', 'width', 'height'])

class MNISTLoader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.1307],
                                         std=[0.3081])
        self.im_size = ImageShape(1, 28, 28)

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        self.transform_validate = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    def get_data(self, data_loc, download=False):
        train_set = datasets.MNIST(data_loc, train=True, download=download,
                                   transform=self.transform_train)
        val_set = datasets.MNIST(data_loc, train=False,
                                 transform=self.transform_validate)
        return train_set, val_set


class CINIC10Loader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        self.im_size = ImageShape(3, 32, 32)
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

    def get_data(self, data_loc, download=False):
        if download:
            download_cinic(root_to_cinic.replace('-enlarged',''))
            if '-enlarged' in which_set:
                enlarge_cinic_10(root_to_cinic.replace('-enlarged',''))

        traindir = os.path.join(data_loc, 'train')
        valdir = os.path.join(data_loc, 'test')

        train_set = datasets.ImageFolder(traindir, self.transform_train)
        val_set = datasets.ImageFolder(valdir, self.transform_validate)

        return train_set, val_set


class CIFAR10Loader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        self.im_size = ImageShape(3, 32, 32)
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

    def get_data(self, data_loc, download=False):
        train_set = datasets.CIFAR10(root=data_loc,
                                     train=True, download=download, transform=self.transform_train)
        val_set = datasets.CIFAR10(root=data_loc,
                                   train=False, download=download, transform=self.transform_validate)

        return train_set, val_set


class CIFAR100Loader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                                         std=[0.2009, 0.1984, 0.2023])
        self.im_size = ImageShape(3, 32, 32)
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

    def get_data(self, data_loc, download=False):
        train_set = datasets.CIFAR100(root=data_loc,
                                      train=True, download=download, transform=self.transform_train)
        val_set = datasets.CIFAR100(root=data_loc,
                                    train=False, download=download, transform=self.transform_validate)
        return train_set, val_set


class ImageNetLoader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.im_size = ImageShape(3, 224, 224)
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

    def get_data(self, data_loc, download=False):
        traindir = os.path.join(data_loc, 'train')
        valdir = os.path.join(data_loc, 'val')

        train_set = datasets.ImageFolder(traindir, self.transform_train)
        val_set = datasets.ImageFolder(valdir, self.transform_validate)

        return train_set, val_set

def load_dataset(dataset, data_loc, batch_size=128, num_dataloading_workers=0, download=False):

    datasets = {
        'mnist': MNISTLoader,
        'cinic10': CINIC10Loader,
        'cifar10': CIFAR10Loader,
        'cifar100': CIFAR100Loader,
        'imagenet': ImageNetLoader,
    }

    dataloader = datasets[dataset.lower()]()

    ### e.g. ADD SIMCLR
    dataloader.transform_train = SimCLRTransform(size=dataloader.im_size.width)
    ### 

    train_set, val_set = dataloader.get_data(data_loc, download)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_dataloading_workers=num_dataloading_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size, shuffle=False,
        num_dataloading_workers=num_dataloading_workers, pin_memory=True)

    return train_loader, val_loader, dataloader.im_size
