from copy import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10 as CIFAR10Base


# NOTE: jsonargparse has trouble if os.PathLike is in the union
PathLike = Union[str, Path]


class CIFAR10(CIFAR10Base):

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, target = super().__getitem__(index)
        return {"img": img, "label": torch.tensor(target)}


class CycledCIFAR10(CIFAR10):
    def __init__(self, *args, cycle: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle = cycle

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = index % len(self.data)
        return super().__getitem__(index)

    def __len__(self):
        return len(self.data) * self.cycle


class CIFAR10DataModule(LightningDataModule):
    r"""Data module for CIFAR10 dataset.

    Note that this DataModule uses the "test" split for both testing and validation.
    CIFAR-10 is only part of this library for testing and ablation purposes.

    .. note::
        It is recommended to use `torchvision.transforms.v2` for transforms.

    Args:
        root: Root directory of the dataset.
        batch_size: Size of the batches.
        seed: Seed for random number generation.
        train_transforms: Transformations to apply to the training images.
        train_gpu_transforms: GPU transformations to apply to the training images.
        val_transforms: Transformations to apply to the validation images.
        test_transforms: Transformations to apply to the test images.
        train_dataset_kwargs: Additional keyword arguments for the training dataset.
        dataset_kwargs: Additional keyword arguments for inference datasets.
        num_workers: Number of workers for data loading.
        pin_memory: Whether to pin memory.
        prefetch_factor: Prefetch factor for data loading.
        persistent_workers: Whether to use persistent workers for the training dataloader.

    Keyword Args:
        Forwarded to :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        root: PathLike,
        batch_size: int = 4,
        seed: int = 42,
        train_transforms: Optional[Callable] = None,
        train_gpu_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        train_dataset_kwargs: Dict[str, Any] = {},
        dataset_kwargs: Dict[str, Any] = {},
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = True,
        cycle: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.seed = seed
        # NOTE: Callable[[E], E] generic seems to break jsonargparse
        # Accept transforms as Callable and cast to Transform
        self.train_transforms = train_transforms
        self.train_gpu_transforms = train_gpu_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.train_dataset_kwargs = train_dataset_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers if num_workers > 0 else False
        self.dataloader_config = kwargs
        self.cycle = cycle

    def prepare_data(self):
        CIFAR10(root=self.root, train=True, download=True)
        CIFAR10(root=self.root, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        match stage:
            case "fit":
                # prepare training dataset
                rank_zero_info("Preparing training datasets")
                if self.cycle is not None:
                    self.dataset_train = CycledCIFAR10(
                        self.root, cycle=self.cycle, train=True, transform=self.train_transforms
                    )
                else:
                    self.dataset_train = CIFAR10(self.root, train=True, transform=self.train_transforms)
                self.dataset_val = CIFAR10(self.root, train=False, transform=self.val_transforms)

            case None:
                pass  # pragma: no cover

            case _:
                raise ValueError(f"Unknown stage: {stage}")  # pragma: no cover

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        if not hasattr(self, "dataset_train"):
            raise RuntimeError("setup() must be called before train_dataloader()")  # pragma: no cover
        return self._data_loader(
            self.dataset_train, shuffle=True, drop_last=True, persistent_workers=self.persistent_workers
        )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        if not hasattr(self, "dataset_val"):
            raise RuntimeError("setup() must be called before val_dataloader()")  # pragma: no cover
        return self._data_loader(self.dataset_val, shuffle=False)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        if not hasattr(self, "dataset_val"):
            raise RuntimeError("setup() must be called before test_dataloader()")  # pragma: no cover
        return self._data_loader(self.dataset_val, shuffle=False)

    def _data_loader(self, dataset: Dataset, **kwargs) -> DataLoader:
        config = copy(self.dataloader_config)
        config.update(kwargs)
        config["batch_size"] = self.batch_size

        # Torch forces us to pop these arguments when using a batch_sampler
        if config.get("batch_sampler", None) is not None:
            config.pop("batch_size", None)
            config.pop("shuffle", None)
            config.pop("sampler", None)

        return DataLoader(
            dataset,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            **config,
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        assert self.trainer is not None
        # TODO: Should we consider allowing GPU transforms for val/test?
        # This was originally added to speed up training which is more augmentation intensive
        if self.trainer.training and self.train_gpu_transforms is not None:
            batch = self.train_gpu_transforms(batch)
        return batch
