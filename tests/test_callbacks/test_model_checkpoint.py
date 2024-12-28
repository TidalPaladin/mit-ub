from unittest.mock import Mock

import pytest
import pytorch_lightning as pl

from mit_ub.callbacks.model_checkpoint import ModelCheckpoint
from mit_ub.tasks.jepa import JEPA


@pytest.fixture
def mock_trainer():
    trainer = Mock(spec=pl.Trainer)
    trainer.lightning_module = Mock(spec=pl.LightningModule)
    return trainer


def test_on_fit_end(mock_trainer, tmp_path, mocker):
    """Test that on_fit_end properly sets save_weights_only and calls parent methods."""
    callback = ModelCheckpoint(dirpath=tmp_path)

    mock_monitor = mocker.patch.object(ModelCheckpoint, "_monitor_candidates")
    mock_save = mocker.patch.object(ModelCheckpoint, "_save_last_checkpoint")
    mock_monitor.return_value = {"val_loss": 0.5}

    callback.on_fit_end(mock_trainer, mock_trainer.lightning_module)

    assert callback.save_weights_only
    mock_monitor.assert_called_once_with(mock_trainer)
    mock_save.assert_called_once_with(mock_trainer, {"val_loss": 0.5})


def test_synchronize_ema_weights(mock_trainer, tmp_path, mocker):
    """Test that _save_checkpoint synchronizes weights for JEPA tasks."""
    callback = ModelCheckpoint(dirpath=tmp_path)
    mock_trainer.lightning_module = Mock(spec=JEPA)

    mock_parent_save = mocker.patch("pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint._save_checkpoint")
    callback._save_checkpoint(mock_trainer, str(tmp_path / "checkpoint.ckpt"))

    mock_trainer.lightning_module.synchronize_ema_weights.assert_called_once()
    mock_parent_save.assert_called_once()
