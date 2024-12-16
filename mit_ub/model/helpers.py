import functools
import os


def compile_is_disabled() -> bool:
    """Gets state of ``torch.compile`` from environment variable.

    Set ``TORCH_COMPILE=0`` to disable ``torch.compile``.
    """
    return os.getenv("TORCH_COMPILE", "1").lower() == "0"


def compile_backend() -> str:
    """Gets state of ``torch.compile`` from environment variable.

    Set ``TORCH_COMPILE=0`` to disable ``torch.compile``.
    """
    return os.getenv("TORCH_COMPILE_BACKEND", "inductor")


class reprable:
    """Decorates a function with a repr method."""

    def __init__(self, wrapped):
        self._wrapped = wrapped
        functools.update_wrapper(self, wrapped)

    def __call__(self, *args, **kwargs):
        return self._wrapped(*args, **kwargs)

    def __repr__(self):
        return f"{self._wrapped.__name__}"
