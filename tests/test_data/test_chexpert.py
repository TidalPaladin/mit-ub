import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest
from PIL import Image
from torch.testing import assert_close
from torch_dicom.datasets.image import load_image

from mit_ub.data.chexpert import CheXpert, entrypoint


@pytest.fixture(scope="module")
def image_factory():
    def func(root: Path, train: bool, patient: str, study: str, filename: str) -> Path:
        path = root / ("train" if train else "valid") / f"patient{patient}" / f"study{study}" / f"{filename}.jpg"
        H, W = 128, 128
        img = Image.new("L", (W, H), (255,))
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return path

    return func


@pytest.fixture(scope="module")
def csv_factory():
    def func(root: Path, train: bool, patients: List[str], studies: List[str], filenames: List[str]) -> Path:
        header = [
            "Path",
            "Sex",
            "Age",
            "Frontal/Lateral",
            "AP/PA",
            "No Finding",
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

        # Values can be 1.0, 0.0, -1.0 or missing. -1.0 indicates unknown. Authors treat missing as 0.0
        data = []
        for patient, study, filename in zip(patients, studies, filenames):
            # Randomly choose values for this row
            row = [
                f"CheXpert-v1.0/{'train' if train else 'valid'}/patient{patient}/study{study}/{filename}.jpg",
                "Female" if int(patient) % 2 == 0 else "Male",
                str(20 + int(patient) % 60),
                "Frontal" if int(study) % 2 == 0 else "Lateral",
                "AP" if int(study) % 2 == 0 else "PA",
                *[str([-1.0, 0.0, 1.0, ""][int(patient) % 4]) for i in range(len(header) - 5)],
            ]
            data.append(row)
        df = pd.DataFrame(data, columns=header)
        csv_path = root / ("train.csv" if train else "valid.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    return func


@pytest.fixture
def data_factory(tmp_path, csv_factory, image_factory):
    def func(train: bool):
        patients = [0, 1, 2]
        studies = [1, 2, 1]
        filenames = ["view1_frontal", "view1_frontal", "view1_frontal"]

        csv_path = csv_factory(tmp_path, train=train, patients=patients, studies=studies, filenames=filenames)
        for patient, study, filename in zip(patients, studies, filenames):
            image_factory(tmp_path, train=train, patient=patient, study=study, filename=filename)
        return csv_path

    return func


class TestCheXpertDataset:

    def test_train(self, tmp_path, data_factory):
        data_factory(train=True)
        dataset = CheXpert(tmp_path, train=True)
        assert len(dataset) == 3
        e1 = dataset[0]
        assert e1["img"].shape == (3, 128, 128)
        assert e1["finding"] in [-1.0, 0.0, 1.0]
        assert e1["dense_label"].shape == (13,)

    def test_valid(self, tmp_path, data_factory):
        data_factory(train=False)
        dataset = CheXpert(tmp_path, train=False)
        assert len(dataset) == 3
        e1 = dataset[0]
        assert e1["img"].shape == (3, 128, 128)
        assert e1["finding"] in [-1.0, 0.0, 1.0]
        assert e1["dense_label"].shape == (13,)


@pytest.mark.parametrize("size", [None, (32, 32)])
def test_preprocess_chexpert(tmp_path, data_factory, size):
    output = tmp_path / "output"
    output.mkdir()
    data_factory(train=True)
    data_factory(train=False)
    sys.argv = ["mit_ub.data.chexpert", str(tmp_path), str(output), "--num_workers", "0"]
    if size is not None:
        sys.argv.extend(["--size", str(size[0]), str(size[1])])

    entrypoint()

    for source_path in tmp_path.rglob("*.jpg"):
        relpath = source_path.relative_to(tmp_path)
        dest_path = (output / relpath).with_suffix(".tiff")
        assert dest_path.is_file()
        if size is None:
            source = load_image(source_path)
            dest = load_image(dest_path)
            assert_close(source, dest)
