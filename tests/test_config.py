from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pytest
import torch.nn as nn

from mit_ub.model.activations import ACTIVATIONS, Activation
from mit_ub.model.config import (
    ModelConfig,
    convert_activation_to_str,
    convert_sequences,
    convert_str_to_activation,
    get_activation_fields,
)


@dataclass(frozen=True)
class DummyConfig(ModelConfig):
    activation: str | Activation = "gelu"
    gate_activation: str | Activation | None = None
    sequence_field: Sequence[int] = field(default_factory=list)

    def instantiate(self) -> nn.Module:
        return nn.Identity()


class TestModelConfig:
    @pytest.fixture
    def config(self) -> DummyConfig:
        return DummyConfig()

    @pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
    def test_save_load(self, tmp_path: Path, config: DummyConfig, suffix: str):
        path = tmp_path / f"config{suffix}"
        config.save(path)
        loaded = DummyConfig.from_file(path)
        assert loaded == config

    def test_invalid_extension(self, tmp_path: Path, config: DummyConfig):
        path = tmp_path / "config.invalid"
        with pytest.raises(ValueError, match="Unsupported file extension"):
            config.save(path)


def test_get_activation_fields():
    config = DummyConfig()
    fields = get_activation_fields(config)
    assert fields == ["activation", "gate_activation"]


def test_convert_activation_to_str():
    config = DummyConfig(activation=ACTIVATIONS["gelu"])
    converted = convert_activation_to_str(config)
    assert isinstance(converted.activation, str)
    assert converted.activation == "gelu"


def test_convert_str_to_activation():
    config = DummyConfig(activation="gelu")
    converted = convert_str_to_activation(config)
    assert callable(converted.activation)
    assert converted.activation == ACTIVATIONS["gelu"]


@pytest.mark.parametrize("container", [list, tuple])
def test_convert_sequences(container):
    config = DummyConfig(sequence_field=[1, 2, 3])
    converted = convert_sequences(config, container)
    assert isinstance(converted.sequence_field, container)
    assert list(converted.sequence_field) == [1, 2, 3]
