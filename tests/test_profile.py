import sys

import pytest

from mit_ub.profile import entrypoint


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("training", [False, True])
def test_profile(capsys, device, training):
    sys.argv = [
        "profile",
        "vit-cifar10",
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
