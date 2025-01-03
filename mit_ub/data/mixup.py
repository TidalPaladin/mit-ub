import torch
import torch.nn.functional as F
from torch import Tensor


def _right_broadcast(inp: Tensor, proto: Tensor) -> Tensor:
    return inp.view(-1, *(1,) * len(proto.shape[1:]))


@torch.no_grad()
def sample_mixup_parameters(
    size: int,
    mixup_prob: float = 0.2,
    mixup_alpha: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    r"""Samples an interpolation weight tensor for MixUp.

    Args:
        size: The size of the tensor to sample.
        mixup_prob: The probability of applying mixup to the input and target.
        mixup_alpha: The alpha parameter for the Beta distribution used to sample the mixup weight.
        device: The device to sample the tensor on.

    Returns:
        A tensor of size ``size`` sampled from a Beta distribution with parameters ``mixup_alpha`` and ``mixup_alpha``.
    """
    # Generate mixup weight
    u = torch.empty(size, device=device).uniform_(0, 1).pow_(1.0 / mixup_alpha)
    v = torch.empty(size, device=device).uniform_(0, 1).pow_(1.0 / mixup_alpha)
    weight = u.div_(u + v)

    # Generate mask of mixup samples
    mixup_mask = torch.rand_like(weight) < mixup_prob

    # Update the weight tensor with the mixup mask
    weight = torch.where(mixup_mask, weight, 0.0)
    return weight


@torch.no_grad()
def mixup(x: Tensor, weight: Tensor) -> Tensor:
    r"""Apply MixUp to an input tensor using a given weight.

    The input tensor is rolled along the first dimension and linearly interpolated
    with the original input using the provided weight.

    Args:
        x: The input tensor.
        weight: The weight tensor.

    Returns:
        ``x.lerp(x.roll(1, 0), weight)``
    """
    x = x.float()
    return x.lerp(x.roll(1, 0), _right_broadcast(weight, x))


@torch.no_grad()
def mixup_dense_label(x: Tensor, weight: Tensor, num_classes: int) -> Tensor:
    r"""Applies mixup weights to a dense label tensor.

    Args:
        x: The input tensor.
        weight: The mixup weights.
        num_classes: The number of classes.

    Shapes:
        x: :math:`(B,)`
        weight: :math:`(B,)`
        Output: :math:`(B, N)` where :math:`N` is the number of classes

    Returns:
        The mixed label tensor.
    """
    y = x.new_zeros(x.shape[0], num_classes, dtype=torch.float)
    y[x >= 0] = F.one_hot(x[x >= 0], num_classes=num_classes).float()
    return mixup(y, weight)


@torch.no_grad()
def is_mixed(weight: Tensor) -> Tensor:
    r"""Returns a mask of samples with mixup.

    Args:
        weight: The mixup weights.

    Shapes:
        weight: :math:`(B,)`
        Output: :math:`(B,)`

    Returns:
        The mixed label tensor.
    """
    return weight != 0


@torch.no_grad()
def is_mixed_with_unknown(weight: Tensor, mask: Tensor) -> Tensor:
    r"""Checks which examples have mixup applied and the mixed counterpart has unknown label.

    Args:
        weight: The mixup weights.
        mask: The mask of valid labels.

    Shapes:
        weight: :math:`(B,)`
        mask: :math:`(B,)`
        Output: :math:`(B,)`
    """
    mixed = is_mixed(weight)
    counterpart = mask.roll(1, 0)
    return mixed & ~counterpart
