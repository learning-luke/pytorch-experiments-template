import torch
import argparse
import unittest

from models import get_model

class TestModels(unittest.TestCase):
    def test_resnet(self):
        res18 = get_model('resnet18', num_classes=10, dataset='cifar10', in_channels=100)
        cifar_data = torch.rand((64,3,32,32))
        _ = res18(cifar_data)

        res18 = get_model('resnet18', num_classes=1000, dataset='imagenet')
        imagenet_data = torch.rand((64,3,224,224))
        _ = res18(imagenet_data)


if __name__ == '__main__':
    unittest.main()
