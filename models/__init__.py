from .auto_builder_models import EasyPeasyResNet
from .densenet import *
from .resnet import *
from .wresnet import *
from .auto_builder_transformers import *

model_zoo = {
    "ResNet9": ResNet9,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "PreActResNet18": PreActResNet18,
    "PreActResNet34": PreActResNet34,
    "PreActResNet50": PreActResNet50,
    "PreActResNet101": PreActResNet101,
    "PreActResNet152": PreActResNet152,
    "DenseNet121": DenseNet121,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
    "DenseNet161": DenseNet161,
    "WideResNet_16_8": WideResNet_16_8,
    "WideResNet_28_10": WideResNet_28_10,
    "WideResNet_40_2": WideResNet_40_2,
    "ViT32LastTimeStep": EasyPeasyViTLastTimeStep,
    "ViT32Flatten": EasyPeasyViTFlatten,
    "EasyPeasyResNet": EasyPeasyResNet
}
