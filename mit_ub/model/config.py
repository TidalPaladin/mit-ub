import json
import tarfile
import tempfile
from abc import abstractmethod
from copy import copy
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Callable, ClassVar, List, Protocol, Self, Sequence, Type, TypeVar, cast, runtime_checkable

import torch.nn as nn
import yaml
from deep_helpers.helpers import load_checkpoint
from safetensors.torch import load_file

from .activations import ACTIVATIONS, get_activation


@dataclass(frozen=True)
class ModelConfig:

    def __post_init__(self) -> None:
        convert_str_to_activation(self)
        convert_sequences(self, tuple)

    @classmethod
    def from_file(cls, path: Path) -> Self:
        r"""Read a config from a JSON or YAML file."""
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

    @classmethod
    def from_tar(cls, path: Path) -> Self:
        r"""Read a config from `config.{yaml,json,yml}` inside a tar.gz file.

        Args:
            path: Path to a tar.gz file containing a a config file.

        """
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found at {path}")  # pragma: no cover

        POSSIBLE_FILES = (Path("config.yaml"), Path("config.json"), Path("config.yml"))
        is_yaml = False
        with tarfile.open(path, "r:gz") as tar:
            for file in POSSIBLE_FILES:
                try:
                    config_file = tar.extractfile(str(file))
                    if config_file is not None:
                        is_yaml = file.suffix in (".yaml", ".yml")
                        break
                except KeyError:
                    continue
            else:
                raise IOError(f"Could not load {POSSIBLE_FILES} from checkpoint")  # pragma: no cover

            if is_yaml:
                config = yaml.safe_load(config_file)
            else:
                config = json.load(config_file)

        return cls(**config)

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


DEFAULT_CHECKPOINT_NAME = "checkpoint.safetensors"


@runtime_checkable
class SupportsSafeTensors(Protocol):
    CONFIG_TYPE: ClassVar[Type[ModelConfig]]

    def load_safetensors(self, path: PathLike, strict: bool = True) -> Self:
        r"""Load a model from a safetensors file.

        Args:
            path: Path to a safetensors file.
            strict: When `False`, try to coerce parameter shapes and ignore missing/extra keys.

        Returns:
            The model with the checkpoint loaded.
        """
        assert isinstance(self, nn.Module), "This method is only supported for nn.Module instances"
        path = Path(path)
        for p in self.parameters():
            device = p.device
            break
        else:
            raise ValueError("No parameters found in model")

        state_dict = load_file(path, device=str(device))
        load_checkpoint(self, state_dict, strict)
        return self

    @classmethod
    def load_tar(
        cls,
        path: PathLike,
        strict: bool = True,
        checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    ) -> Self:
        r"""Load a model from a tar.gz file.

        Args:
            path: Path to a tar.gz file containing a `config.{yaml,json,yml}` and a `checkpoint.safetensors` file.
            strict: When `False`, try to coerce parameter shapes and ignore missing/extra keys.
            checkpoint_name: Name of the checkpoint file inside the tar.gz file, if different from `checkpoint.safetensors`.

        Returns:
            The model with the checkpoint loaded.
        """
        # First read the config and use it to instantiate the model
        path = Path(path)
        config = cls.CONFIG_TYPE.from_tar(path)
        model = config.instantiate()
        assert isinstance(model, SupportsSafeTensors)

        # Then load the checkpoint
        with tarfile.open(path, "r:gz") as tar:
            # Reading from tar gives a byte stream, which we put into a temp file for safe_open
            stream = tar.extractfile(checkpoint_name)
            if stream is None:
                raise IOError(f"Could not load {checkpoint_name} from tar")
            with tempfile.NamedTemporaryFile() as tmp:
                tmp.write(stream.read())
                tmp.flush()
                return cast(Any, model).load_safetensors(Path(tmp.name), strict)
