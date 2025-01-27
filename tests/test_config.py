import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pytest
import torch.nn as nn
from safetensors.torch import save_file

from mit_ub.model.activations import ACTIVATIONS, Activation
from mit_ub.model.config import (
    ModelConfig,
    SupportsSafeTensors,
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

    @pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
    def test_from_tar(self, tmp_path: Path, config: DummyConfig, suffix: str):
        path = tmp_path / f"config{suffix}"
        config.save(path)
        tar_path = tmp_path / "config.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(path, arcname="config.yaml")

        loaded = DummyConfig.from_tar(tar_path)
        assert loaded == config


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


@dataclass(frozen=True)
class SimpleModelConfig(ModelConfig):
    dim: int = 10

    def instantiate(self) -> nn.Module:
        return SimpleModel(self)


class SimpleModel(nn.Sequential, SupportsSafeTensors):
    CONFIG_TYPE = SimpleModelConfig

    def __init__(self, config: SimpleModelConfig):
        super().__init__(
            nn.Linear(config.dim, config.dim),
            nn.LayerNorm(config.dim),
            nn.ReLU(),
            nn.Linear(config.dim, 1),
        )
        self.config = config


class TestSupportsSafeTensors:
    @pytest.fixture
    def model(self) -> nn.Module:
        return SimpleModel(SimpleModelConfig(dim=10))

    @pytest.fixture
    def safetensors_checkpoint(self, tmp_path, model):
        checkpoint_path = tmp_path / "checkpoint.safetensors"
        state_dict = model.state_dict()
        save_file(state_dict, checkpoint_path)
        return checkpoint_path

    @pytest.fixture
    def tar_checkpoint(self, tmp_path, safetensors_checkpoint, model):
        model.config.save(tmp_path / "config.yaml")

        tar_path = tmp_path / "checkpoint.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(safetensors_checkpoint, arcname="checkpoint.safetensors")
            tar.add(tmp_path / "config.yaml", arcname="config.yaml")

        return tar_path

    def test_load_safetensors(self, model: nn.Module, safetensors_checkpoint: Path):
        # Fill with an irregular value
        for param in model.parameters():
            param.data.fill_(3.0)

        # Load should update the irregular value back to normal
        loaded = model.load_safetensors(safetensors_checkpoint)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()

    def test_load_tar(self, model: nn.Module, tar_checkpoint: Path):
        # Fill with an irregular value
        for param in model.parameters():
            param.data.fill_(3.0)

        # Load should update the irregular value back to normal
        loaded = SimpleModel.load_tar(tar_checkpoint)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()
