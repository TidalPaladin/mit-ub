#!/usr/bin/env python
# -*- coding: utf-8 -*-
from typing import Sized, cast

import pytest
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import RandomVerticalFlip

from mit_ub.data import DicomDataModule


@pytest.mark.usefixtures("datamodule_input", "mock_dcmread")
class TestDicomDataModule:
    @pytest.fixture
    def datamodule(self, manager, stage, preprocessed_data):
        dm = DicomDataModule(preprocessed_data, test_inputs=preprocessed_data, manager=manager)
        dm.setup(stage=stage or "fit")
        return dm

    def test_multiple_inputs(self, manager, preprocessed_data):
        dm1 = DicomDataModule(preprocessed_data, manager=manager)
        dm2 = DicomDataModule([preprocessed_data, preprocessed_data], manager=manager)
        dm1.setup(stage="fit")
        dm2.setup(stage="fit")
        assert len(cast(Sized, dm2.dataset_train)) == 2 * len(cast(Sized, dm1.dataset_train))

    @pytest.mark.parametrize(
        "stage,dataloader",
        [
            ("fit", DicomDataModule.train_dataloader),
            ("fit", DicomDataModule.val_dataloader),
            ("test", DicomDataModule.test_dataloader),
        ],
    )
    def test_dataloader(self, datamodule, dataloader):
        result = dataloader(datamodule)
        assert isinstance(result, DataLoader)
        assert result.batch_size == datamodule.batch_size

    @pytest.mark.parametrize(
        "val_inputs, train_len, val_len",
        [
            (0.1, 108, 12),
            (5, 115, 5),
            (None, 120, 120),
        ],
    )
    def test_val_split(self, manager, preprocessed_data, val_inputs, train_len, val_len):
        if val_inputs is None:
            val_inputs = preprocessed_data
        dm = DicomDataModule(preprocessed_data, val_inputs=val_inputs, batch_size=1, manager=manager)
        dm.setup(stage="fit")

        assert len(cast(Sized, dm.dataset_train)) == train_len
        assert len(cast(Sized, dm.train_dataloader())) == train_len
        assert len(cast(Sized, dm.dataset_val)) == val_len
        assert len(cast(Sized, dm.val_dataloader())) == val_len
