import unittest
import argparse
import torch
import torchvision.models as models

from utils import data_loaders

DATA_LOC = "~/datasets/"


class TestLoaders(unittest.TestCase):
    def test_cifar10(self):
        (
            train_loader,
            val_loader,
            train_set,
            val_set,
            image_shape,
        ) = data_loaders.load_dataset("cifar10", DATA_LOC + "cifar10", download=True)

        num_classes = len(train_set.classes)
        self.assertEqual(num_classes, 10)

        resnet18 = models.resnet18(num_classes=num_classes)

        with torch.no_grad():
            train_x, train_y = next(iter(train_loader))
            val_x, val_y = next(iter(val_loader))

            y = resnet18(train_x)
            self.assertEqual(y.size()[0], train_y.size()[0])

            y = resnet18(val_x)
            self.assertEqual(y.size()[0], val_y.size()[0])

    def test_cinic10(self):
        (
            train_loader,
            val_loader,
            train_set,
            val_set,
            image_shape,
        ) = data_loaders.load_dataset("cinic10", DATA_LOC + "cinic10", download=True)

        num_classes = len(train_set.classes)
        self.assertEqual(num_classes, 10)

        resnet18 = models.resnet18(num_classes=num_classes)

        with torch.no_grad():
            train_x, train_y = next(iter(train_loader))
            val_x, val_y = next(iter(val_loader))

            y = resnet18(train_x)
            self.assertEqual(y.size()[0], train_y.size()[0])

            y = resnet18(val_x)
            self.assertEqual(y.size()[0], val_y.size()[0])


if __name__ == "__main__":
    unittest.main()
