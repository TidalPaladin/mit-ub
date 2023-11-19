import os
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Final, Iterator, List, Optional, Sequence, Sized, Tuple, Union, cast

import torch
from dicom_utils.volume import VOLUME_HANDLERS, VolumeHandler
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms.v2 import Compose
from torch_dicom.datasets import collate_fn, DicomPathDataset, ImagePathDataset
from ..model import BACKBONES
from ..helpers import transfer_batch_to_device

TRAIN_VOLUME_HANDLER: Final = cast(VolumeHandler, VOLUME_HANDLERS.get("max-1-5").instantiate_with_metadata().fn)
PathLike = Union[str, os.PathLike, Path]

def _prepare_inputs(inputs: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    return [
        Path(i) 
        for i in ([inputs] if isinstance(inputs, (str, os.PathLike, Path)) else inputs)
    ]


class DicomDataModule(LightningDataModule):

    def __init__(
        self,
        train_inputs: Union[str, Sequence[str]],
        val_inputs: Union[str, Sequence[str]] = [],
        test_inputs: Union[str, Sequence[str]] = [],
        batch_size: int = 4,
        seed: int = 42,
        shuffle: bool = True,
        train_transforms: Optional[Union[Callable, Compose]] = None,
        val_transforms: Optional[Union[Callable, Compose]] = None,
        test_transforms: Optional[Union[Callable, Compose]] = None,
        train_dataset_kwargs: Dict[str, Any] = {},
        dataset_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.train_inputs = _prepare_inputs(train_inputs)
        self.val_inputs = val_inputs if isinstance(val_inputs, (int, float)) else _prepare_inputs(val_inputs)
        self.test_inputs = _prepare_inputs(test_inputs)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.train_dataset_kwargs = train_dataset_kwargs
        self.dataset_kwargs = dataset_kwargs

        self.dataloader_config = kwargs
        self.dataloader_config.setdefault("num_workers", 0)
        self.dataloader_config.setdefault("pin_memory", True)

    def create_dataset(
        self,
        target: Union[str, os.PathLike],
        **kwargs,
    ) -> DicomPathDataset:
        target = Path(target)
        if not target.is_dir():
            raise NotADirectoryError(target)  # pragma: no cover

        dicom_images = target.rglob("*.dcm")
        dataset = DicomPathDataset(dicom_images, **kwargs)
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""

        if stage == "fit" or stage is None:
            # prepare training dataset
            train_dataset_config = self.train_dataset_kwargs
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            train_datasets = [
                self.create_dataset(inp, transform=train_transforms, **train_dataset_config)
                for inp in self.train_inputs
            ]
            self.dataset_train = ConcatDataset(train_datasets)
            assert isinstance(self.dataset_train, Sized)
            if not len(self.dataset_train):
                raise RuntimeError(f"Empty training dataset from inputs: {self.train_inputs}")  # pragma: no cover

            # prepare validation dataset
            infer_dataset_config = self.dataset_kwargs
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms
            val_datasets = [
                self.create_dataset(inp, transform=val_transforms, **infer_dataset_config)
                for inp in self.val_inputs
            ]
            self.dataset_val = ConcatDataset(val_datasets)
            assert isinstance(self.dataset_val, Sized)
            if not len(self.dataset_val):
                raise RuntimeError(f"Empty validation dataset from inputs: {self.val_inputs}")  # pragma: no cover

        if stage == "test" or stage is None:
            infer_dataset_config = self.dataset_kwargs
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            test_datasets = [
                self.create_dataset(inp, transform=test_transforms, **infer_dataset_config) for inp in self.test_inputs
            ]
            self.dataset_test = ConcatDataset(test_datasets)
            assert isinstance(self.dataset_test, Sized)
            if not len(self.dataset_test):
                raise RuntimeError(f"Empty test dataset from inputs: {self.test_inputs}")  # pragma: no cover

    @abstractmethod
    def default_transforms(self) -> Optional[Callable]:
        """Default transform for the dataset."""
        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, train=True)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        return self._data_loader(self.dataset_val, train=False)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        return self._data_loader(self.dataset_test, train=False)

    def _data_loader(self, dataset: Dataset, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            drop_last=train,
            collate_fn=collate_fn,
            **self.dataloader_config,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx) -> Any:
        return transfer_batch_to_device(self, batch, device, dataloader_idx)


class ImageDataModule(LightningDataModule):

    def __init__(
        self,
        train_inputs: Union[str, Sequence[str]],
        val_inputs: Union[str, Sequence[str]] = [],
        test_inputs: Union[str, Sequence[str]] = [],
        batch_size: int = 4,
        seed: int = 42,
        shuffle: bool = True,
        train_transforms: Optional[Union[Callable, Compose]] = None,
        val_transforms: Optional[Union[Callable, Compose]] = None,
        test_transforms: Optional[Union[Callable, Compose]] = None,
        train_dataset_kwargs: Dict[str, Any] = {},
        dataset_kwargs: Dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super().__init__()
        self.train_inputs = _prepare_inputs(train_inputs)
        self.val_inputs = val_inputs if isinstance(val_inputs, (int, float)) else _prepare_inputs(val_inputs)
        self.test_inputs = _prepare_inputs(test_inputs)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.train_dataset_kwargs = train_dataset_kwargs
        self.dataset_kwargs = dataset_kwargs

        self.dataloader_config = kwargs
        self.dataloader_config.setdefault("num_workers", 0)
        self.dataloader_config.setdefault("pin_memory", True)

    def create_dataset(
        self,
        target: Union[str, os.PathLike],
        normalize=False,
        **kwargs,
    ) -> ImagePathDataset:
        target = Path(target)
        if not target.is_dir():
            raise NotADirectoryError(target)  # pragma: no cover

        images = target.rglob("*.png")
        dataset = ImagePathDataset(images, **kwargs, normalize=normalize)
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""

        if stage == "fit" or stage is None:
            # prepare training dataset
            train_dataset_config = self.train_dataset_kwargs
            train_transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            train_datasets = [
                self.create_dataset(inp, transform=train_transforms, **train_dataset_config)
                for inp in self.train_inputs
            ]
            self.dataset_train = ConcatDataset(train_datasets)
            assert isinstance(self.dataset_train, Sized)
            if not len(self.dataset_train):
                raise RuntimeError(f"Empty training dataset from inputs: {self.train_inputs}")  # pragma: no cover

            # prepare validation dataset
            infer_dataset_config = self.dataset_kwargs
            val_transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms
            val_datasets = [
                self.create_dataset(inp, transform=val_transforms, **infer_dataset_config)
                for inp in self.val_inputs
            ]
            self.dataset_val = ConcatDataset(val_datasets)
            assert isinstance(self.dataset_val, Sized)
            if not len(self.dataset_val):
                raise RuntimeError(f"Empty validation dataset from inputs: {self.val_inputs}")  # pragma: no cover

        if stage == "test" or stage is None:
            infer_dataset_config = self.dataset_kwargs
            test_transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms
            test_datasets = [
                self.create_dataset(inp, transform=test_transforms, **infer_dataset_config) for inp in self.test_inputs
            ]
            self.dataset_test = ConcatDataset(test_datasets)
            assert isinstance(self.dataset_test, Sized)
            if not len(self.dataset_test):
                raise RuntimeError(f"Empty test dataset from inputs: {self.test_inputs}")  # pragma: no cover

    @abstractmethod
    def default_transforms(self) -> Optional[Callable]:
        """Default transform for the dataset."""
        return

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        return self._data_loader(self.dataset_train, train=True)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        return self._data_loader(self.dataset_val, train=False)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        return self._data_loader(self.dataset_test, train=False)

    def _data_loader(self, dataset: Dataset, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            drop_last=train,
            collate_fn=collate_fn,
            **self.dataloader_config,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx) -> Any:
        return transfer_batch_to_device(self, batch, device, dataloader_idx)