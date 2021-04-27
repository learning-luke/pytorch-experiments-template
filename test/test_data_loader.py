import argparse
import torch
import torchvision.models as models
import os
import math
from datasets import dataset_loading_hub as data_loaders

DATA_LOC = os.environ.get("PYTORCH_DATA_LOC")


def test_cifar10_val_split():
    (
        train_loader_no_split,
        val_loader_no_split,
        test_loader_no_split,
        train_set_no_split,
        val_set_no_split,
        test_set_no_split,
        image_shape_no_split,
        num_classes_no_split,
    ) = data_loaders.load_dataset(
        "cifar10",
        os.path.join(DATA_LOC, "cifar10"),
        batch_size=1,
        test_batch_size=1,
        download=False,
        val_set_percentage=0,
    )

    (
        train_loader,
        val_loader,
        test_loader,
        train_set,
        val_set,
        test_set,
        image_shape,
        num_classes,
    ) = data_loaders.load_dataset(
        "cifar10",
        os.path.join(DATA_LOC, "cifar10"),
        batch_size=1,
        test_batch_size=1,
        download=False,
        val_set_percentage=0.2,
    )

    # make sure data is being siphoned off from the train set correctly
    assert len(train_loader_no_split) > len(train_loader)
    assert len(val_loader_no_split) < len(val_loader)
    assert len(train_loader) + len(val_loader) == len(train_loader_no_split)


def test_cifar10():
    batch_size = 128
    val_set_percentage = 0.1
    (
        train_loader,
        val_loader,
        test_loader,
        train_set,
        val_set,
        test_set,
        image_shape,
        num_classes,
    ) = data_loaders.load_dataset(
        "cifar10",
        os.path.join(DATA_LOC, "cifar10"),
        batch_size=batch_size,
        test_batch_size=batch_size,
        download=False,
        val_set_percentage=val_set_percentage,
    )

    assert num_classes == 10
    assert len(train_loader) == math.ceil(
        math.ceil(50000 / batch_size) * (1 - val_set_percentage)
    )
    assert len(val_loader) == math.ceil(
        math.ceil(50000 / batch_size) * (val_set_percentage)
    )

    resnet18 = models.resnet18(num_classes=num_classes)

    with torch.no_grad():
        train_x, train_y = next(iter(train_loader))
        val_x, val_y = next(iter(val_loader))

        assert len(train_x) == batch_size
        assert len(val_x) == batch_size

        y = resnet18(train_x)
        assert y.size()[0] == train_y.size()[0]

        resnet18.eval()

        y = resnet18(val_x)
        assert y.size()[0] == val_y.size()[0]


def test_cinic10():
    (
        train_loader,
        val_loader,
        test_loader,
        train_set,
        val_set,
        test_set,
        image_shape,
        num_classes,
    ) = data_loaders.load_dataset(
        "cinic10", os.path.join(DATA_LOC, "cinic10"), download=False
    )

    assert num_classes == 10

    resnet18 = models.resnet18(num_classes=num_classes)

    with torch.no_grad():
        _extracted_from_test_cinic10_20(train_loader, val_loader, resnet18)

def _extracted_from_test_cinic10_20(train_loader, val_loader, resnet18):
    train_x, train_y = next(iter(train_loader))
    val_x, val_y = next(iter(val_loader))

    y = resnet18(train_x)
    assert y.size()[0] == train_y.size()[0]

    resnet18.eval()

    y = resnet18(val_x)
    assert y.size()[0] == val_y.size()[0]
