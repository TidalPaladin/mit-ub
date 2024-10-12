import os


def compile_is_disabled() -> bool:
    """Gets state of ``torch.compile`` from environment variable.

    Set ``TORCH_COMPILE=0`` to disable ``torch.compile``.
    """
    return os.getenv("TORCH_COMPILE", "1").lower() == "0"
