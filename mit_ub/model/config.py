import json
from abc import abstractmethod
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Self, Sequence, Type, TypeVar

import torch.nn as nn
import yaml

from .activations import ACTIVATIONS, get_activation


@dataclass(frozen=True)
class ModelConfig:

    def __post_init__(self) -> None:
        convert_str_to_activation(self)
        convert_sequences(self, tuple)

    @classmethod
    def from_file(cls, path: Path) -> Self:
        match path.suffix.lower():
            case ".json":
                with open(path, "r") as f:
                    data = json.load(f)
            case ".yaml" | ".yml":
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
            case _:
                raise ValueError(f"Unsupported file extension: {path.suffix}")

        return cls(**data)

    def _prepare_for_save(self) -> Self:
        config = copy(self)
        config = convert_activation_to_str(config)
        config = convert_sequences(config, list)
        return config

    def save(self, path: Path) -> None:
        if path.suffix.lower() not in (".json", ".yaml", ".yml"):
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        config = self._prepare_for_save()
        with open(path, "w") as f:
            if path.suffix.lower() == ".json":
                json.dump(asdict(config), f)
            elif path.suffix.lower() in (".yaml", ".yml"):
                yaml.dump(asdict(config), f)

    @abstractmethod
    def instantiate(self) -> nn.Module:
        raise NotImplementedError


T = TypeVar("T", bound=ModelConfig)


def get_activation_fields(config: ModelConfig) -> List[str]:
    return [name for name in config.__dataclass_fields__.keys() if "activation" in name.replace(" ", "")]


def convert_activation_to_str(config: T) -> T:
    r"""Convert any field of type Activation to a string."""
    # Convert activation and gate_activation to strings.
    # First try using the name of the activation function. If that isnt in ACTIVATIONS,
    # traverse the dict to find the key.
    for key in get_activation_fields(config):
        fn = getattr(config, key)
        if fn is None:
            continue

        # Try using function name directly
        replacement = None
        if fn.__name__ in ACTIVATIONS:
            replacement = fn.__name__
        else:
            # Search through ACTIVATIONS dict
            for name, func in ACTIVATIONS.items():
                if func == fn:
                    replacement = name
                    break

        if replacement is not None:
            object.__setattr__(config, key, replacement)
            assert isinstance(getattr(config, key), str)
        else:
            raise ValueError(f"Activation function {fn} not found in ACTIVATIONS")

    return config


def convert_str_to_activation(config: T) -> T:
    r"""Convert any field of type Activation to a string."""
    for key in get_activation_fields(config):
        val = getattr(config, key)
        if val is None:
            continue

        act = get_activation(val)
        object.__setattr__(config, key, act)
        assert isinstance(getattr(config, key), Callable)

    return config


def convert_sequences(config: T, container: Type[Sequence]) -> T:
    r"""Convert any field of type Sequence to the container type."""
    # Get all fields that are sequences
    fields = config.__dataclass_fields__
    sequence_fields = [name for name, field in fields.items() if "Sequence" in str(field.type).replace(" ", "")]

    # Convert activation and gate_activation to strings.
    # First try using the name of the activation function. If that isnt in ACTIVATIONS,
    # traverse the dict to find the key.
    for key in sequence_fields:
        val = getattr(config, key)
        if val is None:
            continue

        object.__setattr__(config, key, container(val))  # type: ignore
        assert isinstance(getattr(config, key), container)

    return config
