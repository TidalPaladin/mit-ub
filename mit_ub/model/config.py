import json
from abc import abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

import torch.nn as nn
import yaml


@dataclass(frozen=True)
class ModelConfig:

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
        return self

    def save(self, path: Path) -> None:
        self._prepare_for_save()
        with open(path, "w") as f:
            if path.suffix.lower() == ".json":
                json.dump(asdict(self), f)
            elif path.suffix.lower() in (".yaml", ".yml"):
                yaml.dump(asdict(self), f)

    @abstractmethod
    def instantiate(self) -> nn.Module:
        raise NotImplementedError
