import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore


def transfer_batch_to_device(model, batch, device, dataloader_idx):
    # PL complains because FileRecord is a frozen dataclass. Pop it and then reattach
    records = batch.pop("record", None)
    batch = pl.LightningModule.transfer_batch_to_device(model, batch, device, dataloader_idx)
    if records is not None:
        batch["record"] = records
    return batch
