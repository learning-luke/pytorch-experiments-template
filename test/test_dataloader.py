import argparse
import torch
import torchvision.models as models

from utils import data_loaders

DATA_LOC = "~/datasets/"


def test_cifar10():
    (
        train_loader,
        val_loader,
        train_set,
        val_set,
        image_shape,
    ) = data_loaders.load_dataset("cifar10", DATA_LOC + "cifar10", download=True)

    num_classes = 10

    resnet18 = models.resnet18(num_classes=num_classes)

    with torch.no_grad():
        train_x, train_y = next(iter(train_loader))
        val_x, val_y = next(iter(val_loader))

        y = resnet18(train_x)
        assert y.size()[0] == train_y.size()[0]

        y = resnet18(val_x)
        assert y.size()[0] == val_y.size()[0]


def test_cinic10():
    (
        train_loader,
        val_loader,
        train_set,
        val_set,
        image_shape,
    ) = data_loaders.load_dataset("cinic10", DATA_LOC + "cinic10", download=False)

    num_classes = len(train_set.classes)
    assert num_classes == 10

    resnet18 = models.resnet18(num_classes=num_classes)

    with torch.no_grad():
        train_x, train_y = next(iter(train_loader))
        val_x, val_y = next(iter(val_loader))

        y = resnet18(train_x)
        assert y.size()[0] == train_y.size()[0]

        y = resnet18(val_x)
        assert y.size()[0] == val_y.size()[0]
