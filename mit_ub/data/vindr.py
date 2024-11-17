from copy import copy
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import numpy as np
import pandas as pd
from lightning_fabric.utilities.rank_zero import rank_zero_info
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_dicom.datasets import ImagePathDataset, collate_fn
from torch_dicom.datasets.helpers import Transform


# NOTE: jsonargparse has trouble if os.PathLike is in the union
PathLike = Union[str, Path]


class Split(StrEnum):
    TRAIN = "training"
    VAL = "val"
    TEST = "test"


class VinDrMammo(ImagePathDataset):

    def __init__(self, root: Path, split: Split, *args, **kwargs) -> None:
        # Load metadata
        self.metadata = pd.read_csv(root / "metadata.csv", index_col="SOP Instance UID")
        self.breast = pd.read_csv(root / "breast-level_annotations.csv", index_col="image_id")
        self.finding = pd.read_csv(root / "finding_annotations.csv", index_col="image_id")

        # Apply split
        if split == Split.TEST:
            self.breast = self.breast[self.breast["split"] == "test"]
        elif split in (Split.TRAIN, Split.VAL):
            self.breast = self.breast[self.breast["split"] == "training"]
            # Split by distinct study_id, 10% val and remaining for train
            study_ids = self.breast["study_id"].unique()
            np.random.seed(42)
            val_study_ids = np.random.choice(study_ids, size=int(len(study_ids) * 0.1), replace=False)
            if split == Split.VAL:
                self.breast = self.breast[self.breast["study_id"].isin(val_study_ids)]
            else:
                self.breast = self.breast[~self.breast["study_id"].isin(val_study_ids)]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Determine paths
        study_ids = self.breast["study_id"]
        image_ids = self.breast.index
        paths = set(root / f"{study_id}" / f"{image_id}.tiff" for study_id, image_id in zip(study_ids, image_ids))
        paths = [p for p in paths if p.exists()]
        super().__init__(iter(paths), *args, **kwargs)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = cast(Dict[str, Any], super().__getitem__(index))
        path = example["path"]
        image_id = path.stem
        breast = self.breast.loc[image_id]
        example["view_position"] = breast["view_position"].lower()
        example["density"] = breast["breast_density"][-1].lower()
        example["birads"] = int(breast["breast_birads"][-1])
        return example


class VinDrDataModule(LightningDataModule):
    r"""Data module for VinDr Mammo dataset.

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
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.seed = seed
        # NOTE: Callable[[E], E] generic seems to break jsonargparse
        # Accept transforms as Callable and cast to Transform
        self.train_transforms = cast(Transform, train_transforms)
        self.train_gpu_transforms = cast(Transform, train_gpu_transforms)
        self.val_transforms = cast(Transform, val_transforms)
        self.test_transforms = cast(Transform, test_transforms)
        self.train_dataset_kwargs = train_dataset_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.dataloader_config = kwargs

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        match stage:
            case "fit":
                # prepare training dataset
                rank_zero_info("Preparing training datasets")
                self.dataset_train = VinDrMammo(self.root, Split.TRAIN, transform=self.train_transforms)
                self.dataset_val = VinDrMammo(self.root, Split.VAL, transform=self.val_transforms)

            case "test":
                self.dataset_test = VinDrMammo(self.root, Split.TEST, transform=self.test_transforms)

            case None:
                pass  # pragma: no cover

            case _:
                raise ValueError(f"Unknown stage: {stage}")  # pragma: no cover

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        if not hasattr(self, "dataset_train"):
            raise RuntimeError("setup() must be called before train_dataloader()")  # pragma: no cover
        return self._data_loader(self.dataset_train, shuffle=True, drop_last=True)

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
            collate_fn=partial(collate_fn, default_fallback=False),
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


if __name__ == "__main__":
    train = VinDrMammo(Path("/mnt/data/vindr/lowres"), Split.TRAIN)
    print(f"Train: {len(train)}")
    val = VinDrMammo(Path("/mnt/data/vindr/lowres"), Split.VAL)
    print(f"Val: {len(val)}")
    test = VinDrMammo(Path("/mnt/data/vindr/lowres"), Split.TEST)
    print(f"Test: {len(test)}")
