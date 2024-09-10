import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest
import pytorch_lightning as pl
import yaml
from deep_helpers.testing import checkpoint_factory
from dicom_utils.dicom_factory import CompleteMammographyStudyFactory

from mit_ub.tasks.triage import BreastTriage, entrypoint


@pytest.fixture(scope="module")
def dicom_target(tmpdir_factory):
    tmp_path = Path(tmpdir_factory.mktemp("dicom_target"))
    fact = CompleteMammographyStudyFactory()
    paths = []
    for i, dcm in enumerate(fact()):
        path = tmp_path / f"{i}.dcm"
        dcm.save_as(path)
        paths.append(path)

    text_file = tmp_path / "paths.txt"
    with open(text_file, "w") as f:
        for path in paths:
            f.write(f"{path}\n")
    return text_file


@pytest.fixture
def png_target(datamodule, tmp_path):
    path = tmp_path / "paths.txt"
    files = datamodule.train_inputs[0].rglob("*.png")
    with open(path, "w") as f:
        for file in files:
            f.write(f"{file}\n")
    return path


@pytest.fixture(params=["dicom_target", "png_target"])
def target(request, dicom_target, png_target):
    return request.getfixturevalue(request.param)


@pytest.fixture
def config(backbone):
    return {
        "backbone": backbone,
    }


@pytest.fixture
def checkpoint(tmp_path, task, config):
    path = tmp_path / "model.safetensors"
    checkpoint_factory(task, tmp_path, path.name)

    config = {
        "class_path": task.__class__.__module__ + "." + task.__class__.__name__,
        "init_args": config,
    }
    with open(path.parent / "config.yaml", "w") as f:
        yaml.dump(config, f)
    return path


class TestBreastTriage:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return BreastTriage(backbone, optimizer_init=optimizer_init)

    @pytest.mark.skip(reason="'malignant' key will be renamed soon")
    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_inference_cli(self, config, device, target, checkpoint, capsys, batch_size):
        sys.argv = [
            sys.argv[0],
            str(target),
            str(checkpoint),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
            "-i",
            "64",
            "32",
        ]
        entrypoint()
        out = capsys.readouterr().out
        df = pd.read_csv(StringIO(out))
        num_lines_expected = open(target).read().count("\n")
        assert len(df) == num_lines_expected
