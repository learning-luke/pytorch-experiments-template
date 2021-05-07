from .auto_builder_models import AutoResNet
from .densenet import *
from .resnet import *
from .wresnet import *
from .auto_builder_transformers import *

import inspect
from inspect import Parameter
import functools
from typing import Callable, Any


def ignore_unexpected_kwargs(func: Callable[..., Any]) -> Callable[..., Any]:
    sig = inspect.signature(func)
    params = sig.parameters.values()

    def filter_kwargs(kwargs: dict):
        _params = filter(
            lambda p: p.kind
            in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY},
            params,
        )

        res_kwargs = {
            param.name: kwargs[param.name] for param in _params if param.name in kwargs
        }
        return res_kwargs

    def contain_var_keyword():
        return len(params) >= 1 and any(
            filter(lambda p: p.kind == Parameter.VAR_KEYWORD, params)
        )

    def contain_var_positional():
        return len(params) >= 1 and any(
            filter(lambda p: p.kind == Parameter.VAR_POSITIONAL, params)
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        kwargs = filter_kwargs(kwargs)
        return func(*args, **kwargs)

    ret_func = func
    if not contain_var_keyword():
        if contain_var_positional():
            raise RuntimeError("*args not supported")
        ret_func = wrapper

    return ret_func

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
    "ViT32LastTimeStep": AutoViTLastTimeStep,
    "ViT32Flatten": AutoViTFlatten,
    "AutoResNet": AutoResNet
}
