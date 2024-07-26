from copy import copy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Sized, Union, cast

from deep_helpers.data.sampler import ConcatSampler
from deep_helpers.structs import Mode
from lightning_fabric.utilities.rank_zero import rank_zero_info
from pytorch_lightning import LightningDataModule
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)
from torch_dicom.datasets import ImagePathDataset, MetadataDatasetWrapper, collate_fn
from torch_dicom.datasets.helpers import Transform


# NOTE: jsonargparse has trouble if os.PathLike is in the union
PathLike = Union[str, Path]


def _prepare_inputs(inputs: Union[PathLike, Sequence[PathLike]]) -> List[Path]:
    return [Path(i) for i in ([inputs] if isinstance(inputs, PathLike) else inputs)]


def _unwrap_dataset(dataset: Union[ImagePathDataset, MetadataDatasetWrapper]) -> ImagePathDataset:
    _dataset = dataset
    while isinstance(_dataset, MetadataDatasetWrapper):
        _dataset = _dataset.dataset
    assert isinstance(_dataset, ImagePathDataset)
    return _dataset


def _prepare_sopuid_exclusions(sopuid_exclusions: PathLike | Iterable[str] | None) -> Set[str]:
    if isinstance(sopuid_exclusions, (str, Path)):
        sopuid_exclusions = Path(sopuid_exclusions)
        if not sopuid_exclusions.is_file():
            raise FileNotFoundError(sopuid_exclusions)  # pragma: no cover
        with open(sopuid_exclusions, "r") as f:
            return set(f.read().splitlines())

    elif isinstance(sopuid_exclusions, Iterable):
        return set(sopuid_exclusions)

    else:
        return set()


class PNGDataModule(LightningDataModule):
    r"""Data module for preprocessed PNG images.

    .. note::
        It is recommended to use `torchvision.transforms.v2` for transforms.

    .. note::
        This DataModule may not interact well with lightning's automated distributed sampling wrapper when
        using a custom batch sampler.

    Args:
        train_inputs: Paths to the training images.
        val_inputs: Paths to the validation images.
        test_inputs: Paths to the test images.
        batch_size: Size of the batches.
        seed: Seed for random number generation.
        train_transforms: Transformations to apply to the training images.
        train_gpu_transforms: GPU transformations to apply to the training images.
        val_transforms: Transformations to apply to the validation images.
        test_transforms: Transformations to apply to the test images.
        train_dataset_kwargs: Additional keyword arguments for the training dataset.
        dataset_kwargs: Additional keyword arguments for inference datasets.
        metadata_filenames: Dictionary mapping metadata keys to filenames. For example,
            ``{"manifest": "manifest.csv", "annotation": "annotation.csv"}``.
        boxes_filename: Filename of the boxes file to read with :class:`BoundingBoxMetadata`.
            If ``None``, no bounding box metadata will be applied.
        boxes_extra_keys: Extra keys for the boxes. See :class:`BoundingBoxMetadata` for more information.
        num_workers: Number of workers for data loading.
        pin_memory: Whether to pin memory.
        prefetch_factor: Prefetch factor for data loading.
        train_sopuid_exclusions: SOPInstanceUIDs or path to a file of such to exclude from the training set.
        val_sopuid_exclusions: SOPInstanceUIDs or path to a file of such to exclude from the validation set.
        test_sopuid_exclusions: SOPInstanceUIDs or path to a file of such to exclude from the test set.

    Keyword Args:
        Forwarded to :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        train_inputs: Union[PathLike, Sequence[PathLike]] = [],
        val_inputs: Union[PathLike, Sequence[PathLike]] = [],
        test_inputs: Union[PathLike, Sequence[PathLike]] = [],
        batch_size: int = 4,
        seed: int = 42,
        train_transforms: Optional[Callable] = None,
        train_gpu_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        train_dataset_kwargs: Dict[str, Any] = {},
        dataset_kwargs: Dict[str, Any] = {},
        metadata_filenames: Dict[str, str] = {},
        boxes_filename: Optional[str] = None,
        boxes_extra_keys: Iterable[str] = [],
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: Optional[int] = None,
        train_sopuid_exclusions: PathLike | Iterable[str] | None = None,
        val_sopuid_exclusions: PathLike | Iterable[str] | None = None,
        test_sopuid_exclusions: PathLike | Iterable[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.train_inputs = _prepare_inputs(train_inputs)
        self.val_inputs = _prepare_inputs(val_inputs)
        self.test_inputs = _prepare_inputs(test_inputs)
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
        self.metadata_filenames = metadata_filenames
        self.boxes_filename = boxes_filename
        self.boxes_extra_keys = boxes_extra_keys
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.dataloader_config = kwargs
        self.train_sopuid_exclusions = _prepare_sopuid_exclusions(train_sopuid_exclusions)
        self.val_sopuid_exclusions = _prepare_sopuid_exclusions(val_sopuid_exclusions)
        self.test_sopuid_exclusions = _prepare_sopuid_exclusions(test_sopuid_exclusions)

    def create_dataset(
        self,
        target: PathLike,
        mode: Mode,
        sopuid_exclusions: Set[str] = set(),
        **kwargs,
    ) -> MetadataDatasetWrapper:
        """
        Creates a dataset of preprocessed images and applies metadata wrappers.

        Args:
            target: Path to the directory containing the preprocessed images.
            mode: The mode of the dataset. One of "train", "val", or "test".
            sopuid_exclusions: Set of SOPInstanceUIDs to exclude out.

        Keyword Args:
            Forwarded to :class:`ImagePathDataset`.

        Returns:
            Dataset: The preprocessed dataset with metadata wrappers applied.
        """
        target = Path(target)
        if not target.is_dir():
            raise NotADirectoryError(target)  # pragma: no cover

        images = filter(lambda p: p.stem not in sopuid_exclusions, target.rglob("*.png"))
        dataset = ImagePathDataset(images, **kwargs)

        return dataset

    def create_sampler(
        self,
        dataset: ImagePathDataset,
        example_paths: List[Path],
        root: Path,
        mode: Mode,
    ) -> Sampler[int]:
        """
        Creates a sampler for the given dataset. By default, a random sampler is created for the training dataset
        and a sequential sampler is created for the validation and test datasets.

        Args:
            dataset: The dataset for which the sampler is to be created.
            example_paths: The paths to the examples in the dataset.
            root: The root path of the dataset.
            mode: The mode of the dataset. One of "train", "val", or "test".

        Returns:
            The created sampler for the dataset.
        """
        return RandomSampler(dataset) if mode == Mode.TRAIN else SequentialSampler(dataset)

    def create_batch_sampler(
        self,
        dataset: ConcatDataset,
        sampler: Sampler,
        example_paths: List[Path],
        roots: List[Path],
        mode: Mode,
    ) -> Optional[BatchSampler]:
        """
        Creates a batch sampler for aggregate dataset. By default returns ``None``, meaning
        the default batch sampler.

        Args:
            dataset: The :class:`ConcatDataset` for which the sampler is to be created.
            sampler: The :class:`Sampler` for the aggregate dataset.
            example_paths: The paths to the examples in the aggregate dataset.
            roots: The root paths of the datasets.
            mode: The mode of the dataset. One of "train", "val", or "test".

        Returns:
            The created batch sampler for the dataset.
        """
        return  # pragma: no cover

    def _prepare_datasets(
        self,
        inputs: Iterable[Path],
        mode: Mode,
        **kwargs,
    ) -> List[MetadataDatasetWrapper]:
        # Look up associated SOPInstanceUIDs to filter out
        match mode:
            case Mode.TRAIN:
                sopuid_exclude = self.train_sopuid_exclusions
            case Mode.VAL:
                sopuid_exclude = self.val_sopuid_exclusions
            case Mode.TEST:
                sopuid_exclude = self.test_sopuid_exclusions
            case _:
                sopuid_exclude = set()

        datasets = [self.create_dataset(inp, mode, sopuid_exclude, **kwargs) for inp in inputs]
        assert all(isinstance(ds, Sized) for ds in datasets)
        if not any(isinstance(ds, Sized) and len(ds) for ds in datasets):
            raise FileNotFoundError("Loaded empty dataset")

        return datasets

    def _prepare_samplers(
        self,
        roots: Iterable[Path],
        datasets: Iterable[MetadataDatasetWrapper],
        mode: Mode,
    ) -> List[Sampler]:
        samplers: List[Sampler] = [
            self.create_sampler((unwrapped := _unwrap_dataset(dataset)), list(unwrapped.files), root, mode)
            for dataset, root in zip(datasets, roots)
        ]
        return samplers

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        match stage:
            case "fit":
                # prepare training dataset
                rank_zero_info("Preparing training datasets")
                train_datasets = self._prepare_datasets(self.train_inputs, Mode.TRAIN, transform=self.train_transforms)
                self.dataset_train = ConcatDataset(train_datasets)

                # prepare training sampler
                rank_zero_info("Preparing training samplers")
                train_samplers = self._prepare_samplers(self.train_inputs, train_datasets, Mode.TRAIN)
                # if all datasets use simple random sampler, use a single random sampler.
                # TODO: ConcatSampler will randomly sample from each dataset sequentially. Fix this so that samples
                # are also random between datasets and not just within a dataset
                all_simple_random_samplers = all(
                    isinstance(sampler, RandomSampler) and len(sampler) == len(dataset)
                    for sampler, dataset in zip(train_samplers, train_datasets)
                )
                self.train_sampler = (
                    RandomSampler(self.dataset_train) if all_simple_random_samplers else ConcatSampler(train_samplers)
                )

                # prepare training batch sampler
                self.train_batch_sampler = self.create_batch_sampler(
                    self.dataset_train,
                    self.train_sampler,
                    [path for dataset in train_datasets for path in list(_unwrap_dataset(dataset).files)],
                    self.train_inputs,
                    Mode.TRAIN,
                )

                # Val inputs are optional in fit stage
                if self.val_inputs:
                    # prepare validation dataset
                    rank_zero_info("Preparing validation datasets")
                    val_datasets = self._prepare_datasets(self.val_inputs, Mode.VAL, transform=self.val_transforms)
                    self.dataset_val = ConcatDataset(val_datasets)

                    # prepare validation sampler
                    rank_zero_info("Preparing validation samplers")
                    val_samplers = self._prepare_samplers(self.val_inputs, val_datasets, Mode.VAL)
                    self.val_sampler = ConcatSampler(val_samplers)

                    # prepare validation batch sampler
                    self.val_batch_sampler = self.create_batch_sampler(
                        self.dataset_val,
                        self.val_sampler,
                        [path for dataset in val_datasets for path in list(_unwrap_dataset(dataset).files)],
                        self.val_inputs,
                        Mode.VAL,
                    )

            case "test":
                # prepare test dataset
                rank_zero_info("Preparing test datasets")
                test_datasets = self._prepare_datasets(self.test_inputs, Mode.TEST, transform=self.test_transforms)
                self.dataset_test = ConcatDataset(test_datasets)

                # prepare test sampler
                rank_zero_info("Preparing test samplers")
                test_samplers = self._prepare_samplers(self.test_inputs, test_datasets, Mode.TEST)
                self.test_sampler = ConcatSampler(test_samplers)

                # prepare test batch sampler
                self.test_batch_sampler = self.create_batch_sampler(
                    self.dataset_test,
                    self.test_sampler,
                    [path for dataset in test_datasets for path in list(_unwrap_dataset(dataset).files)],
                    self.test_inputs,
                    Mode.TEST,
                )

            case None:
                pass  # pragma: no cover

            case _:
                raise ValueError(f"Unknown stage: {stage}")  # pragma: no cover

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """The train dataloader."""
        if not hasattr(self, "dataset_train"):
            raise RuntimeError("setup() must be called before train_dataloader()")  # pragma: no cover
        return self._data_loader(self.dataset_train, sampler=self.train_sampler, batch_sampler=self.train_batch_sampler)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The val dataloader."""
        if not self.val_inputs:
            return DataLoader(cast(Any, []))
        elif not hasattr(self, "dataset_val"):
            raise RuntimeError("setup() must be called before val_dataloader()")  # pragma: no cover
        return self._data_loader(self.dataset_val, sampler=self.val_sampler, batch_sampler=self.val_batch_sampler)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """The test dataloader."""
        if not hasattr(self, "dataset_test"):
            raise RuntimeError("setup() must be called before test_dataloader()")  # pragma: no cover
        return self._data_loader(self.dataset_test, sampler=self.test_sampler, batch_sampler=self.test_batch_sampler)

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
