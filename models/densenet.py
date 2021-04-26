"""DenseNet in PyTorch."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print

__all__ = ["DenseNet121", "DenseNet169", "DenseNet201", "DenseNet161"]


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        block,
        nblocks,
        growth_rate=12,
        reduction=0.5,
        num_classes=10,
        in_channels=3,
    ):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        self.num_blocks = 0

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(
            in_channels, num_planes, kernel_size=3, padding=1, bias=False
        )

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for _ in range(nblock):
            self.num_blocks += 1
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):

        layer0 = self.conv1(x)
        layer1 = self.trans1(self.dense1(layer0))
        layer2 = self.trans2(self.dense2(layer1))
        layer3 = self.trans3(self.dense3(layer2))
        layer4 = self.dense4(layer3)
        pool = F.avg_pool2d(F.relu(self.bn(layer4)), 4)
        pool = pool.view(pool.size(0), -1)
        logits = self.linear(pool)
        return logits, (layer0, layer1, layer2, layer3, layer4, pool)


def DenseNet121(growth_rate=32, num_classes=10, **kwargs):
    return DenseNet(
        Bottleneck,
        [6, 12, 24, 16],
        growth_rate=growth_rate,
        num_classes=num_classes,
        **kwargs
    )


def DenseNet169(growth_rate=32, num_classes=10, **kwargs):
    return DenseNet(
        Bottleneck,
        [6, 12, 32, 32],
        growth_rate=growth_rate,
        num_classes=num_classes,
        **kwargs
    )


def DenseNet201(growth_rate=32, num_classes=10, **kwargs):
    return DenseNet(
        Bottleneck,
        [6, 12, 48, 32],
        growth_rate=growth_rate,
        num_classes=num_classes,
        **kwargs
    )


def DenseNet161(growth_rate=48, num_classes=10, **kwargs):
    return DenseNet(
        Bottleneck,
        [6, 12, 36, 24],
        growth_rate=growth_rate,
        num_classes=num_classes,
        **kwargs
    )
