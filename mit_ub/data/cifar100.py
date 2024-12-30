from typing import Any, Dict, Optional

import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from torchvision.datasets import CIFAR100 as CIFAR100Base

from .cifar10 import CIFAR10DataModule


class CIFAR100(CIFAR100Base):

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, target = super().__getitem__(index)
        return {"img": img, "label": torch.tensor(target)}


class CIFAR100DataModule(CIFAR10DataModule):
    r"""Data module for CIFAR100 dataset.

    Note that this DataModule uses the "test" split for both testing and validation.
    CIFAR-100 is only part of this library for testing and ablation purposes.

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

    def prepare_data(self):
        CIFAR100(root=self.root, train=True, download=True)
        CIFAR100(root=self.root, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        match stage:
            case "fit":
                # prepare training dataset
                rank_zero_info("Preparing training datasets")
                self.dataset_train = CIFAR100(self.root, train=True, transform=self.train_transforms)
                self.dataset_val = CIFAR100(self.root, train=False, transform=self.val_transforms)

            case None:
                pass  # pragma: no cover

            case _:
                raise ValueError(f"Unknown stage: {stage}")  # pragma: no cover
