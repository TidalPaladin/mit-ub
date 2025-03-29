import json
import tarfile
import tempfile
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar, Protocol, Self, Sequence, Type, TypeVar, cast, runtime_checkable

import torch.nn as nn
import yaml
from deep_helpers.helpers import load_checkpoint
from safetensors.torch import load_file


@dataclass(frozen=True)
class ModelConfig(ABC):

    def __post_init__(self) -> None:
        convert_sequences(self, tuple)

    @property
    def isotropic_output_dim(self) -> int:
        r"""The output dimension of the model when interpreted as an isotropic model."""
        raise NotImplementedError

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

        for field_name, field_type in cls.__annotations__.items():
            if isinstance(field_type, type) and issubclass(field_type, ModelConfig) and field_name in data:
                data[field_name] = field_type(**data[field_name])  # type: ignore
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

        for field_name, field_type in cls.__annotations__.items():
            if isinstance(field_type, type) and issubclass(field_type, ModelConfig) and field_name in config:
                config[field_name] = field_type(**config[field_name])  # type: ignore
        return cls(**config)

    def _prepare_for_save(self) -> Self:
        config = copy(self)
        config = convert_sequences(config, list)
        for field_name, field_type in self.__class__.__annotations__.items():
            if isinstance(field_type, type) and issubclass(field_type, ModelConfig) and field_name in config.__dict__:
                new_val = asdict(getattr(config, field_name)._prepare_for_save())
                object.__setattr__(config, field_name, new_val)
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

        # Recursive conversion up to depth 2
        if len(val) > 0 and isinstance(val[0], Sequence):
            recursive_converted = container([container(v) for v in val])  # type: ignore
            object.__setattr__(config, key, recursive_converted)

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
