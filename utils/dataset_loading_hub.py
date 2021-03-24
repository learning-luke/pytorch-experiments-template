import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import namedtuple

from torch.utils.data import Subset

from utils.custom_transforms import SimCLRTransform
from utils.cinic_utils import enlarge_cinic_10, download_cinic

ImageShape = namedtuple("ImageShape", ["channels", "width", "height"])


class MNISTLoader:
    def __init__(self):
        normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])
        self.image_shape = ImageShape(1, 28, 28)

        self.transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.transform_validate = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    def get_data(self, data_filepath, train_val_split_tuple, download=False, test=False):
        train_set = datasets.MNIST(
            data_filepath, train=True, download=download, transform=self.transform_train
        )
        if test:
            eval_set = datasets.MNIST(
                data_filepath, train=False, transform=self.transform_validate
            )
        else:
            splits = int(len(train_set) * 0.9), int(len(train_set)) - int(
                len(train_set) * 0.9
            )
            train_set, eval_set = torch.utils.data.random_split(
                train_set, [splits[0], splits[1]]
            )
        return train_set, eval_set


class CINIC10Loader:
    def __init__(self):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        self.image_shape = ImageShape(3, 32, 32)
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.image_shape.width, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.transform_validate = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    def get_data(self, data_loc, download=False, enlarge=False, test=False):
        if download and not os.path.exists(data_loc):
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context

            download_cinic(data_loc.replace("-enlarged", ""))
            if enlarge:
                enlarge_cinic_10(data_loc.replace("-enlarged", ""))

        traindir = os.path.join(data_loc, "train")
        valdir = os.path.join(data_loc, "valid")
        testdir = os.path.join(data_loc, "test")

        train_set = datasets.ImageFolder(traindir, self.transform_train)
        if test:
            eval_set = datasets.ImageFolder(testdir, self.transform_validate)
        else:
            eval_set = datasets.ImageFolder(valdir, self.transform_validate)

        return train_set, eval_set


class CIFAR10Loader:
    def __init__(self):
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
        )
        self.image_shape = ImageShape(3, 32, 32)
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.image_shape.width, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.transform_validate = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    def get_data(self, data_loc, download=False, test=False):
        train_set = datasets.CIFAR10(
            root=data_loc, train=True, download=download, transform=self.transform_train
        )

        if test:
            eval_set = datasets.CIFAR10(
                root=data_loc,
                train=False,
                download=download,
                transform=self.transform_validate,
            )
        else:
            splits = int(len(train_set) * 0.9), int(len(train_set)) - int(
                len(train_set) * 0.9
            )
            train_set, eval_set = torch.utils.data.random_split(
                train_set, [splits[0], splits[1]]
            )
        return train_set, eval_set


class CIFAR100Loader:
    def __init__(self):
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4866, 0.4409], std=[0.2009, 0.1984, 0.2023]
        )
        self.image_shape = ImageShape(3, 32, 32)
        self.transform_train = transforms.Compose(
            [
                transforms.RandomCrop(self.image_shape.width, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.transform_validate = transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )

    def get_data(self, data_loc, download=False, test=False):
        train_set = datasets.CIFAR100(
            root=data_loc, train=True, download=download, transform=self.transform_train
        )

        if test:
            eval_set = datasets.CIFAR100(
                root=data_loc,
                train=False,
                download=download,
                transform=self.transform_validate,
            )
        else:
            train_set, eval_set = torch.utils.data.random_split(train_set, [45000, 5000])
        return train_set, eval_set


class ImageNetLoader:
    def __init__(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.image_shape = ImageShape(3, 224, 224)
        self.transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(self.image_shape.width),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.transform_validate = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(self.im_size.width),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def get_data(self, data_loc, download=False, test=False):
        traindir = os.path.join(data_loc, "train")
        valdir = os.path.join(data_loc, "val")

        train_set = datasets.ImageFolder(traindir, self.transform_train)

        if test:
            eval_set = datasets.ImageFolder(valdir, self.transform_validate)
        else:
            splits = int(len(train_set) * 0.9), int(len(train_set)) - int(
                len(train_set) * 0.9
            )
            train_set, eval_set = torch.utils.data.random_split(
                train_set, [splits[0], splits[1]]
            )

        return train_set, eval_set


def load_dataset(
    dataset,
    data_filepath,
    batch_size=128,
    test_batch_size=128,
    num_workers=0,
    download=False,
    test=False,
):

    datasets = {
        "mnist": MNISTLoader,
        "cinic10": CINIC10Loader,
        "cifar10": CIFAR10Loader,
        "cifar100": CIFAR100Loader,
        "imagenet": ImageNetLoader,
    }

    dataloader = datasets[dataset.lower()]()

    ### e.g. ADD SIMCLR
    # dataloader.transform_train = SimCLRTransform(size=dataloader.im_size.width)
    ###

    train_set, eval_set = dataloader.get_data(
        data_filepath, download, test=test
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        eval_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        train_set,
        eval_set,
        dataloader.image_shape,
    )

def load_split_datasets(dataset, split_tuple):
    total_length = len(dataset)
    total_idx = [i for i in range(total_length)]

    start_end_index_tuples = [(int(len(total_idx) * sum(split_tuple[:i-1])),
                                int(len(total_idx) * split_tuple[i])) for i in range(len(split_tuple))]

    set_selection_index_lists = [total_idx[start_idx: end_idx] for (start_idx, end_idx) in
                                 start_end_index_tuples]

    split_sets = (Subset(dataset, set_indices) for set_indices in set_selection_index_lists)

    return split_sets