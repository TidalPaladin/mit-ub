import sys

import pytest

from mit_ub.model.backbone import ViTConfig
from mit_ub.profile import entrypoint


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("training", [False, True])
def test_profile(tmp_path, capsys, device, training):
    config = ViTConfig(
        in_channels=3,
        dim=128,
        dim_feedforward=256,
        patch_size=(16, 16),
        depth=3,
        nhead=128 // 16,
    )
    config.save(tmp_path / "config.json")

    sys.argv = [
        "profile",
        "vit",
        str(tmp_path / "config.json"),
        "32",
        "32",
        "-b",
        "1",
        "-c",
        "3",
        "-d",
        device,
    ]
    if training:
        sys.argv.append("-t")
    entrypoint()
    captured = capsys.readouterr()
    assert "Calls" in captured.out
