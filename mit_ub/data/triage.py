from pathlib import Path
from typing import List

from deep_helpers.structs import Mode
from torch.utils.data import RandomSampler, Sampler, SequentialSampler
from torch_dicom.datasets import ImagePathDataset
from torch_dicom.datasets.sampler import WeightedCSVSampler
from torch_dicom.preprocessing.datamodule import PreprocessedPNGDataModule as BasePreprocessedPNGDataModule


class PreprocessedPNGDataModule(BasePreprocessedPNGDataModule):

    def __init__(self, *args, malign_weight: float | None = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.malign_weight = malign_weight

    def create_sampler(
        self,
        dataset: ImagePathDataset,
        example_paths: List[Path],
        root: Path,
        mode: Mode,
    ) -> Sampler[int]:
        if mode == Mode.TRAIN:
            if self.malign_weight is None:
                return RandomSampler(dataset)
            else:
                # Enables balanced sampling by density
                if "annotation" not in self.metadata_filenames:
                    raise KeyError("`annotation` not found in `metadata_filenames` keys")
                weights = {
                    "True": self.malign_weight,
                    "False": 1 - self.malign_weight,
                }
                metadata_filename = self.metadata_filenames["annotation"]
                sampler = WeightedCSVSampler(
                    root / metadata_filename,
                    example_paths,
                    colname="malignant",
                    weights=weights,
                )
                return sampler

        else:
            return SequentialSampler(dataset)
