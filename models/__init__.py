from .densenet import *
from .resnet import *
from .wresnet import *


def get_model(model, num_classes=10, dataset="cifar10", **kwargs):

    ## NOTE: the unfortunate use of lambda here can be dropped as soon as
    ## PEP 622 https://www.python.org/dev/peps/pep-0622/ is released
    ## and replaced with pattern matching
    model_zoo = {
        # ResNets
        "resnet18": lambda kwargs: ResNet18(num_classes=num_classes, variant=dataset, **kwargs),
        "resnet34": lambda kwargs: ResNet34(num_classes=num_classes, variant=dataset, **kwargs),
        "resnet50": lambda kwargs: ResNet50(num_classes=num_classes, variant=dataset, **kwargs),
        "resnet101": lambda kwargs: ResNet101(num_classes=num_classes, variant=dataset, **kwargs),
        "resnet152": lambda kwargs: ResNet152(num_classes=num_classes, variant=dataset, **kwargs),
        # DenseNets
        "densenet121": lambda kwargs: DenseNet121(growth_rate=12, num_classes=num_classes, **kwargs),
        "densenet161": lambda kwargs: DenseNet161(growth_rate=12, num_classes=num_classes, **kwargs),
        "densenet169": lambda kwargs: DenseNet169(growth_rate=12, num_classes=num_classes, **kwargs),
        "densenet201": lambda kwargs: DenseNet201(growth_rate=12, num_classes=num_classes, **kwargs),
        # Preact ResNets
        "preact_resnet18": lambda kwargs: PreActResNet18(num_classes, **kwargs),
        "preact_resnet34": lambda kwargs: PreActResNet34(num_classes, **kwargs),
        "preact_resnet50": lambda kwargs: PreActResNet50(num_classes, **kwargs),
        "preact_resnet101": lambda kwargs: PreActResNet101(num_classes, **kwargs),
        "preact_resnet152": lambda kwargs: PreActResNet152(num_classes, **kwargs),
        # WideResNets
        "wrn_16_8": lambda kwargs: WideResNet(depth=16, num_classes=num_classes, widen_factor=8, **kwargs),
        "wrn_28_10": lambda kwargs: WideResNet(depth=28, num_classes=num_classes, widen_factor=10, **kwargs),
        "wrn_40_2": lambda kwargs: WideResNet(depth=40, num_classes=num_classes, widen_factor=2, **kwargs),
    }

    return model_zoo[model](kwargs)
