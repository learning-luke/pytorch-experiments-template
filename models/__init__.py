from .densenet import *
from .resnet import *
from .wresnet import *


def get_model(model, **kwargs):

    model_zoo = {
        # ResNets
        "resnet18": ResNet18,
        "resnet34": ResNet34,
        "resnet50": ResNet50,
        "resnet101": ResNet101,
        "resnet152": ResNet152,
        # DenseNets
        "densenet121": DenseNet121,
        "densenet161": DenseNet161,
        "densenet169": DenseNet169,
        "densenet201": DenseNet201,
        # Preact ResNets
        "preact_resnet18": PreActResNet18,
        "preact_resnet34": PreActResNet34,
        "preact_resnet50": PreActResNet50,
        "preact_resnet101": PreActResNet101,
        "preact_resnet152": PreActResNet152,
        # WideResNets
        "wrn_16_8": WideResNet_16_8,
        "wrn_28_10": WideResNet_28_10,
        "wrn_40_2": WideResNet_40_2,
    }

    return model_zoo[model](**kwargs)
