"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import math

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

from torch import Tensor

from torch.nn import Parameter, init


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


class LinearWithFancyAttention(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, H_{in})` where :math:`*` means any number of
          additional dimensions and :math:`H_{in} = \text{in\_features}`
        - Output: :math:`(N, *, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(LinearWithFancyAttention, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight_attention = nn.Linear(
            in_features=in_features, out_features=out_features * in_features, bias=True
        )
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        batch_weight_shape = [input.shape[0]] + list(self.weight.shape)
        attended_weights = self.weight * F.sigmoid(
            self.weight_attention(input).view(batch_weight_shape)
        )
        return torch.bmm(input.unsqueeze(dim=1), attended_weights).squeeze() + self.bias

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class ResNetWithFancyAttentionalLinear(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, variant=None, in_channels=3):
        super(ResNetWithFancyAttentionalLinear, self).__init__()
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
