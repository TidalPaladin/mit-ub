import sys

import pytest
from vit import ViTConfig

from mit_ub.profile import entrypoint


@pytest.mark.cuda
@pytest.mark.parametrize("training", [False, True])
def test_profile(tmp_path, capsys, training):
    config = ViTConfig(
        in_channels=3,
        patch_size=(16, 16),
        depth=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=128 // 16,
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
        "cuda:0",
    ]
    if training:
        sys.argv.append("-t")
    entrypoint()
    captured = capsys.readouterr()
    assert "Calls" in captured.out
