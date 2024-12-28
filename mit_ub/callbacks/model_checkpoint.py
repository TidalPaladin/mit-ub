import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint as BaseModelCheckpoint

from ..tasks.jepa import JEPA


class ModelCheckpoint(BaseModelCheckpoint):

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        monitor_candidates = self._monitor_candidates(trainer)
        self.save_weights_only = True
        self._save_last_checkpoint(trainer, monitor_candidates)

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        # Synchronize teacher weights before saving checkpoint
        task = trainer.lightning_module
        if isinstance(task, JEPA):
            task.synchronize_ema_weights()
        super()._save_checkpoint(trainer, filepath)
