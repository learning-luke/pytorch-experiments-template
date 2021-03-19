from .densenet import *
from .preact_resnet import *
from .resnet import *
from .wresnet import *

def get_model(model, num_classes=10, dataset='cifar10', **kwargs):

    model_zoo = {
        # ResNets
        'resnet18': ResNet18(num_classes=num_classes, variant=dataset, **kwargs),
        'resnet34': ResNet34(num_classes=num_classes, variant=dataset, **kwargs),
        'resnet50': ResNet50(num_classes=num_classes, variant=dataset, **kwargs),
        'resnet101': ResNet101(num_classes=num_classes, variant=dataset, **kwargs),
        'resnet152': ResNet152(num_classes=num_classes, variant=dataset, **kwargs),

        # DenseNets
        'densenet121': DenseNet121(growth_rate=12, num_classes=num_classes),
        'densenet161': DenseNet161(growth_rate=12, num_classes=num_classes),
        'densenet169': DenseNet169(growth_rate=12, num_classes=num_classes),
        'densenet201': DenseNet201(growth_rate=12, num_classes=num_classes),

        # Preact ResNets
        'preact_resnet18': PreActResNet18(num_classes),
        'preact_resnet34': PreActResNet34(num_classes),
        'preact_resnet50': PreActResNet50(num_classes),
        'preact_resnet101': PreActResNet101(num_classes),
        'preact_resnet152': PreActResNet152(num_classes),

        # WideResNets
        'wrn_16_8': WideResNet(depth=16, num_classes=num_classes, widen_factor=8),
        'wrn_28_10': WideResNet(depth=28, num_classes=num_classes, widen_factor=10),
        'wrn_40_2': WideResNet(depth=40, num_classes=num_classes, widen_factor=2),
    }

    return model_zoo[model]
