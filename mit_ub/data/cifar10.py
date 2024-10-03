from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch
from lightning_fabric.utilities.rank_zero import rank_zero_info
from torchvision.datasets import CIFAR10 as CIFAR10Base

from .datamodule import BaseDataModule


# NOTE: jsonargparse has trouble if os.PathLike is in the union
PathLike = Union[str, Path]


class CIFAR10(CIFAR10Base):

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, target = super().__getitem__(index)
        return {"img": img, "label": torch.tensor(target)}


class CIFAR10DataModule(BaseDataModule):
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
        **kwargs,
    ) -> None:
        super().__init__(
            batch_size,
            seed,
            train_transforms,
            train_gpu_transforms,
            val_transforms,
            test_transforms,
            train_dataset_kwargs,
            dataset_kwargs,
            num_workers,
            pin_memory,
            prefetch_factor,
            **kwargs,
        )
        self.root = Path(root)

    def prepare_data(self):
        CIFAR10(root=self.root, train=True, download=True)
        CIFAR10(root=self.root, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        match stage:
            case "fit":
                # prepare training dataset
                rank_zero_info("Preparing training datasets")
                self.dataset_train = CIFAR10(self.root, train=True, transform=self.train_transforms)
                self.dataset_val = CIFAR10(self.root, train=False, transform=self.val_transforms)

            case "test":
                self.dataset_test = CIFAR10(self.root, train=False, transform=self.test_transforms)

            case None:
                pass  # pragma: no cover

            case _:
                raise ValueError(f"Unknown stage: {stage}")  # pragma: no cover
