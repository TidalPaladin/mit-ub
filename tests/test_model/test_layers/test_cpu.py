import pytest
import torch
import transformer_engine.pytorch as te
from torch.testing import assert_close

from mit_ub.model.layers.cpu import LayerNormLinear, LayerNormMLP


class TestLayerNormLinear:

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("zero_centered_gamma", [False, True])
    def test_weights_compatible(self, bias, normalization, zero_centered_gamma):
        hidden_size = 128
        baseline = te.LayerNormLinear(
            hidden_size,
            hidden_size,
            bias=bias,
            normalization=normalization,
            zero_centered_gamma=zero_centered_gamma,
            device="cpu",
        )
        cpu = LayerNormLinear(
            hidden_size, hidden_size, bias=bias, normalization=normalization, zero_centered_gamma=zero_centered_gamma
        )

        for name, param in baseline.named_parameters():
            other_param = cpu.get_parameter(name)
            assert other_param.shape == param.shape
            assert other_param.dtype == param.dtype

    @pytest.mark.cuda
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("zero_centered_gamma", [False, True])
    def test_forward_equal(self, bias, normalization, zero_centered_gamma):
        hidden_size = 128
        baseline = te.LayerNormLinear(
            hidden_size, hidden_size, bias=bias, normalization=normalization, zero_centered_gamma=zero_centered_gamma
        ).cuda()
        cpu = LayerNormLinear(
            hidden_size, hidden_size, bias=bias, normalization=normalization, zero_centered_gamma=zero_centered_gamma
        )

        for name, param in baseline.named_parameters():
            cpu_param = cpu.get_parameter(name)
            cpu_param.data = param.data.cpu()

        B, L, D = 8, 16, hidden_size

        x = torch.randn(B, L, D, device="cuda")
        y_baseline = baseline(x)
        y_cpu = cpu(x.cpu())

        assert_close(y_baseline, y_cpu, check_device=False, atol=1e-3, rtol=0)


class TestLayerNormMLP:

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("zero_centered_gamma", [False, True])
    def test_weights_compatible(self, bias, normalization, zero_centered_gamma):
        hidden_size = 128
        ffn_hidden_size = 256
        baseline = te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            normalization=normalization,
            zero_centered_gamma=zero_centered_gamma,
            device="cpu",
        )
        cpu = LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            normalization=normalization,
            zero_centered_gamma=zero_centered_gamma,
        )

        for name, param in baseline.named_parameters():
            other_param = cpu.get_parameter(name)
            assert other_param.shape == param.shape
            assert other_param.dtype == param.dtype

    @pytest.mark.cuda
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("zero_centered_gamma", [False, True])
    @pytest.mark.parametrize("activation", ["gelu", "relu", "srelu"])
    def test_forward_equal(self, bias, normalization, zero_centered_gamma, activation):
        hidden_size = 128
        ffn_hidden_size = 256
        baseline = te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            normalization=normalization,
            zero_centered_gamma=zero_centered_gamma,
            activation=activation,
        ).cuda()
        cpu = LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            bias=bias,
            normalization=normalization,
            zero_centered_gamma=zero_centered_gamma,
            activation=activation,
        )

        for name, param in baseline.named_parameters():
            cpu_param = cpu.get_parameter(name)
            cpu_param.data = param.data.cpu()

        B, L, D = 8, 16, hidden_size

        x = torch.randn(B, L, D, device="cuda")
        y_baseline = baseline(x)
        y_cpu = cpu(x.cpu())

        assert_close(y_baseline, y_cpu, check_device=False, atol=1e-3, rtol=0)
