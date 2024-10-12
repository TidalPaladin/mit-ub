import pytest
import pytorch_lightning as pl

from mit_ub.callbacks.lora import LoRACallback, LoRATarget, SupportsLoRA
from mit_ub.model.backbone import AdaptiveViT, ViT


class DummyModule(pl.LightningModule):
    backbone: ViT | AdaptiveViT

    def __init__(self, backbone: ViT | AdaptiveViT):
        super().__init__()
        self.backbone = backbone


class TestLoRACallback:

    @pytest.fixture
    def trainer(self, mocker):
        return mocker.MagicMock(spec_set=pl.Trainer)

    @pytest.fixture(params=[AdaptiveViT, ViT])
    def pl_module(self, request):
        if request.param == ViT:
            backbone = ViT(
                in_channels=3,
                dim=32,
                patch_size=4,
                depth=2,
                nhead=1,
            )
        elif request.param == AdaptiveViT:
            backbone = AdaptiveViT(
                in_channels=3,
                dim=32,
                kv_dim=32,
                patch_size=4,
                target_shape=(8, 8),
                depth=2,
                high_res_depth=2,
                nhead=1,
            )
        else:
            raise ValueError(f"Invalid backbone type: {request.param}")
        return DummyModule(backbone)

    @pytest.mark.parametrize(
        "targets",
        [
            (LoRATarget.ATTENTION, LoRATarget.FEEDFORWARD, LoRATarget.POSITION),
            (LoRATarget.ATTENTION,),
            (LoRATarget.FEEDFORWARD,),
            (LoRATarget.POSITION,),
        ],
    )
    @pytest.mark.parametrize("rank,alpha,dropout", [(8, 16, 0.0), (16, 8, 0.1)])
    @pytest.mark.parametrize("quantize_base", [False, True])
    @pytest.mark.parametrize("freeze_stem", [False, True])
    @pytest.mark.parametrize("freeze_norm", [False, True])
    def test_on_fit_start(
        self, mocker, trainer, pl_module, targets, rank, alpha, dropout, quantize_base, freeze_stem, freeze_norm
    ):
        # Spy each LoRA capable module
        spies = {
            name: mocker.spy(module, "apply_lora")
            for name, module in pl_module.backbone.named_modules()
            if isinstance(module, SupportsLoRA)
        }
        # assert len(spies) > 0

        # Run the callback
        callback = LoRACallback(
            targets=targets,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            quantize_base=quantize_base,
            freeze_stem=freeze_stem,
            freeze_norm=freeze_norm,
        )
        callback.on_fit_start(trainer, pl_module)

        # Check that each LoRA capable module has an apply_lora call
        for name, spy in spies.items():
            spy.assert_called_with(targets, rank, alpha, dropout, quantize_base)

        # Check that the norm layer is frozen/unfrozen correctly
        for name, module in pl_module.backbone.named_modules():
            if "stem" in name:
                continue
            if isinstance(module, pl_module.backbone.norm_layer):
                for param in module.parameters():
                    assert param.requires_grad == (not freeze_norm)

        # Check that the stem is frozen/unfrozen correctly
        for name, param in pl_module.backbone.stem.named_parameters():
            module_name = ".".join(name.split(".")[:-1])
            module = pl_module.backbone.stem.get_submodule(module_name)
            assert param.requires_grad == (not freeze_stem or ("lora_a" in name or "lora_b" in name))
