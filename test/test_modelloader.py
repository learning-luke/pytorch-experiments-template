import torch
import argparse
import unittest

from models import get_model


class TestModels(unittest.TestCase):
    def test_resnet(self):
        res18 = get_model(
            "resnet18", num_classes=10, variant="cifar10", in_channels=100
        )
        cifar_data = torch.rand((64, 100, 32, 32))
        _ = res18(cifar_data)

        res18 = get_model("resnet18", num_classes=1000, variant="imagenet")
        imagenet_data = torch.rand((64, 3, 224, 224))
        _ = res18(imagenet_data)

    def test_densenet(self):
        dense121 = get_model("densenet121", num_classes=10, variant="cifar10")

        cifar_data = torch.rand((64, 3, 32, 32))
        _ = dense121(cifar_data)

    def test_wrn(self):
        wrn_40_2 = get_model("wrn_40_2", num_classes=10, variant="cifar")

        cifar_data = torch.rand((64, 3, 32, 32))

        _ = wrn_40_2(cifar_data)


if __name__ == "__main__":
    unittest.main()
