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