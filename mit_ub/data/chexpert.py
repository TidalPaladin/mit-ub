from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as tv2
from lightning_fabric.utilities.rank_zero import rank_zero_info
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_dicom.datasets import collate_fn
from torch_dicom.datasets.helpers import Transform
from torch_dicom.datasets.image import ImagePathDataset, load_image, save_image
from torch_dicom.preprocessing import MinMaxCrop, Resize
from tqdm_multiprocessing import ConcurrentMapper

from .datamodule import BaseDataModule


# NOTE: jsonargparse has trouble if os.PathLike is in the union
PathLike = Union[str, Path]

LABELS: Final = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


class CheXpert(Dataset):

    def __init__(self, root: PathLike, train: bool = True, transform: Optional[Callable] = None):
        super().__init__()
        root = Path(root)
        if not root.is_dir():
            raise NotADirectoryError(root)

        csv_path = root / ("train.csv" if train else "valid.csv")
        self.root = root / ("train" if train else "valid")
        self.df = pd.read_csv(csv_path)
        self.df["Path"] = self.df["Path"].apply(lambda x: root / x.replace("CheXpert-v1.0/", ""))
        self.df.fillna(0.0, inplace=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        path = Path(row["Path"])
        if path.is_file():
            path = path
        elif path.with_suffix(".tiff").is_file():
            path = path.with_suffix(".tiff")
        else:
            raise FileNotFoundError(path)

        x = load_image(path).contiguous()
        finding, dense_label = self.label_to_tensor(row[self.df.columns[1:]].to_dict())
        if self.transform is not None:
            x = self.transform(x)
        return {"img": x, "finding": finding, "dense_label": dense_label}

    @classmethod
    def label_to_tensor(cls, label: Dict[str, Any]) -> Tuple[Tensor, Tensor]:
        r"""Convert a dictionary of labels to a tensor of dense labels and a 'any-finding' label."""
        dense_label = torch.stack([torch.tensor(label[label_name], dtype=torch.float32) for label_name in LABELS])
        min_label, max_label = dense_label.aminmax()
        if max_label == 1.0:
            finding = dense_label.new_tensor(1.0)
        elif min_label == -1.0:
            finding = dense_label.new_tensor(-1.0)
        else:
            finding = dense_label.new_tensor(0.0)
        return finding, dense_label


class CheXpertDataModule(BaseDataModule):
    r"""Data module for CheXpert dataset.

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

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        match stage:
            case "fit":
                # prepare training dataset
                rank_zero_info("Preparing training datasets")
                self.dataset_train = CheXpert(self.root, train=True, transform=self.train_transforms)
                self.dataset_val = CheXpert(self.root, train=False, transform=self.val_transforms)

            case None:
                pass  # pragma: no cover

            case _:
                raise ValueError(f"Unknown stage: {stage}")  # pragma: no cover


def _preprocess_image(example: Dict[str, Any], root: Path, dest: Path, compression: str | None = None):
    image = example["img"].squeeze()
    source = example["path"][0]
    dest_path = dest / source.relative_to(root)
    dest_path = dest_path.with_suffix(".tiff")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    # Source images are 8 bit grayscale JPEG so we save in 8 bits
    save_image(image, dest_path, dtype=cast(Any, np.uint8), compression=compression)


def preprocess_chexpert(
    source: Path,
    dest: Path,
    size: Tuple[int, int] | None = None,
    num_workers: int = 8,
):
    if not source.is_dir():
        raise NotADirectoryError(source)
    if not dest.is_dir():
        raise NotADirectoryError(dest)

    # Prepare crop and resize
    transforms: List[Transform] = [
        MinMaxCrop(),
    ]
    if size is not None:
        transforms.append(Resize(size, mode="max"))
    transform = tv2.Compose(transforms)

    # Prepare dataset and dataloader
    sources = source.rglob("*.jpg")
    ds = ImagePathDataset(sources, transform=transform)
    dl = DataLoader(ds, num_workers=num_workers, batch_size=1, collate_fn=collate_fn)

    with ConcurrentMapper(jobs=num_workers) as mapper:
        mapper.create_bar(total=len(ds), desc="Preprocessing images")
        func = partial(_preprocess_image, root=source, dest=dest, compression="packbits")
        mapper(func, dl)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Preprocess CheXpert dataset")
    parser.add_argument("source", type=Path, help="Source directory")
    parser.add_argument("dest", type=Path, help="Destination directory")
    parser.add_argument("--size", type=int, nargs=2, default=None, help="Size of the images")
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main(args: Namespace):
    preprocess_chexpert(args.source, args.dest, args.size, args.num_workers)


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
