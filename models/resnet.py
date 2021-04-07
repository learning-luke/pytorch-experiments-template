"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print


__all__ = [
    "ResNet9",
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "ResNet152",
    "PreActResNet18",
    "PreActResNet34",
    "PreActResNet50",
    "PreActResNet101",
    "PreActResNet152",
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    """Pre-activation version of the original Bottleneck module."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, variant=None, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        if variant == "imagenet":
            self.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7 if variant == "imagenet" else 3,
                stride=2 if variant == "imagenet" else 1,
                padding=3 if variant == "imagenet" else 1,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )

        self.bn1 = nn.BatchNorm2d(64)

        #  this pooling is only needed for imagenet-sized images
        self.maxpool = (
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if variant == "imagenet"
            else nn.Identity()
        )

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        layer0 = F.relu(self.bn1(self.conv1(x)))
        maxpool = self.maxpool(layer0)
        layer1 = self.layer1(maxpool)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        avgpool = torch.flatten(self.avgpool(layer4), 1)
        logits = self.linear(avgpool)
        return logits, (layer0, maxpool, layer1, layer2, layer3, layer4, avgpool)


def ResNet9(num_classes=10, variant="cifar10", **kwargs):
    return ResNet(
        BasicBlock, [1, 1, 1, 1], num_classes=num_classes, variant=variant, **kwargs
    )


def ResNet18(num_classes=10, variant="cifar10", **kwargs):
    return ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes=num_classes, variant=variant, **kwargs
    )


def ResNet34(num_classes=10, variant="cifar10", **kwargs):
    return ResNet(
        BasicBlock, [3, 4, 6, 3], num_classes=num_classes, variant=variant, **kwargs
    )


def ResNet50(num_classes=10, variant="cifar10", **kwargs):
    return ResNet(
        Bottleneck, [3, 4, 6, 3], num_classes=num_classes, variant=variant, **kwargs
    )


def ResNet101(num_classes=10, variant="cifar10", **kwargs):
    return ResNet(
        Bottleneck, [3, 4, 23, 3], num_classes=num_classes, variant=variant, **kwargs
    )


def ResNet152(num_classes=10, variant="cifar10", **kwargs):
    return ResNet(
        Bottleneck, [3, 8, 36, 3], num_classes=num_classes, variant=variant, **kwargs
    )


def PreActResNet18(num_classes=10, **kwargs):
    return ResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)


def PreActResNet34(num_classes=10, **kwargs):
    return ResNet(PreActBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def PreActResNet50(num_classes=10, **kwargs):
    return ResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


def PreActResNet101(num_classes=10, **kwargs):
    return ResNet(PreActBottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


def PreActResNet152(num_classes=10, **kwargs):
    return ResNet(PreActBottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
