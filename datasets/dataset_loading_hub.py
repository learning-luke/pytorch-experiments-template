import os
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from collections import namedtuple

from torch.utils.data import Subset

from datasets.custom_transforms import SimCLRTransform
from utils.cinic_utils import extend_cinic_10, download_cinic
from rich import print

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

    def get_data(
        self, data_filepath, val_set_percentage, random_split_seed, download=False
    ):
        train_set = datasets.MNIST(
            data_filepath, train=True, download=download, transform=self.transform_train
        )
        num_training_items = int(len(train_set) * (1.0 - val_set_percentage))
        num_val_items = len(train_set) - num_training_items

        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [num_training_items, num_val_items],
            generator=torch.Generator().manual_seed(random_split_seed),
        )

        test_set = datasets.MNIST(
            data_filepath, train=False, transform=self.transform_validate
        )
        num_labels = 10
        return train_set, val_set, test_set, num_labels


class EMNISTLoader:
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

    def get_data(
        self, data_filepath, val_set_percentage, random_split_seed, download=False
    ):
        train_set = datasets.EMNIST(
            root=data_filepath,
            split="balanced",
            train=True,
            download=download,
            transform=self.transform_train,
        )
        num_training_items = int(len(train_set) * (1.0 - val_set_percentage))
        num_val_items = len(train_set) - num_training_items

        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [num_training_items, num_val_items],
            generator=torch.Generator().manual_seed(random_split_seed),
        )

        test_set = datasets.EMNIST(
            root=data_filepath,
            split="balanced",
            train=False,
            transform=self.transform_validate,
        )
        num_labels = 47
        return train_set, val_set, test_set, num_labels


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

    def get_data(self, data_filepath, download=False, enlarge=False, **args):
        if download and not os.path.exists(data_filepath):
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context

            download_cinic(data_filepath.replace("-enlarged", ""))
            if enlarge:
                extend_cinic_10(data_filepath.replace("-enlarged", ""))

        train_set = datasets.ImageFolder(
            os.path.join(data_filepath, "train"), self.transform_train
        )

        val_set = datasets.ImageFolder(
            os.path.join(data_filepath, "valid"), self.transform_validate
        )

        test_set = datasets.ImageFolder(
            os.path.join(data_filepath, "test"), self.transform_validate
        )

        num_labels = 10
        return train_set, val_set, test_set, num_labels


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

    def get_data(
        self, data_filepath, val_set_percentage, random_split_seed, download=False
    ):
        train_set = datasets.CIFAR10(
            root=data_filepath,
            train=True,
            download=download,
            transform=self.transform_train,
        )

        num_training_items = int(len(train_set) * (1.0 - val_set_percentage))
        num_val_items = len(train_set) - num_training_items

        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [num_training_items, num_val_items],
            generator=torch.Generator().manual_seed(random_split_seed),
        )

        test_set = datasets.CIFAR10(
            root=data_filepath,
            train=False,
            download=download,
            transform=self.transform_validate,
        )

        num_labels = 10
        return train_set, val_set, test_set, num_labels


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

    def get_data(
        self, data_filepath, val_set_percentage, random_split_seed, download=False
    ):
        train_set = datasets.CIFAR100(
            root=data_filepath,
            train=True,
            download=download,
            transform=self.transform_train,
        )

        num_training_items = int(len(train_set) * (1.0 - val_set_percentage))
        num_val_items = len(train_set) - num_training_items

        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [num_training_items, num_val_items],
            generator=torch.Generator().manual_seed(random_split_seed),
        )

        test_set = datasets.CIFAR100(
            root=data_filepath,
            train=False,
            download=download,
            transform=self.transform_validate,
        )

        num_labels = 100
        return train_set, val_set, test_set, num_labels


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
                transforms.CenterCrop(self.image_shape.width),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def get_data(self, data_filepath, val_set_percentage, random_split_seed, **kwargs):
        train_dir = os.path.join(data_filepath, "train")
        val_dir = os.path.join(data_filepath, "val")

        train_set = datasets.ImageFolder(train_dir, self.transform_train)

        num_training_items = int(len(train_set) * (1.0 - val_set_percentage))
        num_val_items = len(train_set) - num_training_items

        train_set, val_set = torch.utils.data.random_split(
            train_set,
            [num_training_items, num_val_items],
            generator=torch.Generator().manual_seed(random_split_seed),
        )

        test_set = datasets.ImageFolder(val_dir, self.transform_validate)

        num_labels = 1000
        return train_set, val_set, test_set, num_labels


# build multi view augmentations
def load_dataset(
    dataset,
    data_filepath,
    batch_size=128,
    test_batch_size=128,
    num_workers=0,
    download=False,
    val_set_percentage=0.0,
    random_split_seed=1,
):

    datasets = {
        "mnist": MNISTLoader,
        "emnist": EMNISTLoader,
        "cinic10": CINIC10Loader,
        "cifar10": CIFAR10Loader,
        "cifar100": CIFAR100Loader,
        "imagenet": ImageNetLoader,
    }

    dataloader = datasets[dataset.lower()]()

    # ## e.g. ADD SIMCLR
    # dataloader.transform_train = SimCLRTransform(size=dataloader.im_size.width)
    ###

    train_set, val_set, test_set, num_labels = dataloader.get_data(
        data_filepath,
        val_set_percentage=val_set_percentage,
        random_split_seed=random_split_seed,
        download=download,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_set,
        val_set,
        test_set,
        dataloader.image_shape,
        num_labels,
    )


def load_split_datasets(dataset, split_tuple):
    total_length = len(dataset)
    total_idx = [i for i in range(total_length)]

    start_end_index_tuples = [
        (
            int(len(total_idx) * sum(split_tuple[: i - 1])),
            int(len(total_idx) * split_tuple[i]),
        )
        for i in range(len(split_tuple))
    ]

    set_selection_index_lists = [
        total_idx[start_idx:end_idx] for (start_idx, end_idx) in start_end_index_tuples
    ]

    return (Subset(dataset, set_indices) for set_indices in set_selection_index_lists)
