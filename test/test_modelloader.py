import torch
import argparse

from models import model_zoo


def test_resnet():
    res18 = model_zoo["ResNet18"](
        num_classes=10, dataset_name="cifar10", in_channels=100
    )
    cifar_data = torch.rand((64, 100, 32, 32))
    _ = res18(cifar_data)

    res18 = model_zoo["ResNet18"](num_classes=1000, dataset_name="imagenet")
    imagenet_data = torch.rand((64, 3, 224, 224))
    _ = res18(imagenet_data)


def test_densenet():
    dense121 = model_zoo["DenseNet121"](num_classes=10, dataset_name="cifar10")

    cifar_data = torch.rand((64, 3, 32, 32))
    _ = dense121(cifar_data)


def test_wrn():
    wrn_40_2 = model_zoo["WideResNet_40_2"](num_classes=10, dataset_name="cifar")

    cifar_data = torch.rand((64, 3, 32, 32))

    _ = wrn_40_2(cifar_data)
