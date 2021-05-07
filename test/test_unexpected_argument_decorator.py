from utils.decorators import ignore_unexpected_kwargs


def test_ignore_unexpected_kwargs():
    @ignore_unexpected_kwargs
    def foo(a, b=0, c=3):
        return a, b, c

    assert foo(0, 0, 0) == (0, 0, 0)
    assert foo(0, b=0, c=0) == (0, 0, 0)
    assert foo(a=0, b=0, c=0) == (0, 0, 0)
    dct = {"a": 1, "b": 2}
    assert foo(**dct) == (1, 2, 3)

    @ignore_unexpected_kwargs
    def bar(a, b, **kwargs):
        return a, b, kwargs.get("c", 3)

    assert bar(**{"a": 1, "b": 2, "c": 4}) == (1, 2, 4)
