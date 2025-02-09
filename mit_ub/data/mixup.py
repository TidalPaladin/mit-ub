import torch
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import Tuple
from torch.utils.cpp_extension import load
from argparse import ArgumentParser, Namespace
from PIL import Image
import numpy as np
import timeit
from torchvision.utils import make_grid
from torch.autograd import Function

if torch.cuda.is_available():
    _mixup_cuda = load(
        name="mixup_cuda",
        sources=[str(Path(__file__).parents[2] / "csrc" / "mixup.cu")],
        extra_cuda_cflags=["-O3"],
    )
else:
    _mixup_cuda = None










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


@torch.no_grad()
def fused_mixup(x: Tensor, mixup_prob: float = 0.2, mixup_alpha: float = 1.0, seed: int | None = None) -> Tensor:
    r"""Apply MixUp to an input tensor using a given weight.

    The input tensor is rolled along the first dimension and linearly interpolated
    with the original input using the provided weight.

    Args:
        x: The input tensor.
        mixup_prob: The probability of applying mixup to the input and target.
        mixup_alpha: The alpha parameter for the Beta distribution used to sample the mixup weight.
        seed: The seed for the random number generator.

    Returns:
        ``x.lerp(x.roll(1, 0), weight)``
    """
    if _mixup_cuda is None:
        raise RuntimeError("MixUp is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _mixup_cuda.mixup(x, mixup_prob, mixup_alpha, seed)


class CrossEntropyMixup(Function):
    @staticmethod
    def forward(ctx, logits: Tensor, labels: Tensor, mixup_prob: float, mixup_alpha: float, seed: int) -> Tensor:
        return _mixup_cuda.cross_entropy_mixup_fwd(logits, labels, mixup_prob, mixup_alpha, seed)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        raise NotImplementedError("Backward pass is not implemented for CrossEntropyMixup")


def cross_entropy_mixup(logits: Tensor, labels: Tensor, mixup_prob: float, mixup_alpha: float, seed: int | None = None) -> Tensor:
    return CrossEntropyMixup.apply(logits, labels, mixup_prob, mixup_alpha, seed)




def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to the image to apply mixup to")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mixup-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-c", "--cuda", default=False, action="store_true", help="Use CUDA kernel")
    parser.add_argument("-f", "--fused", default=False, action="store_true", help="Use fused CUDA kernel")
    parser.add_argument("-p", "--prob", type=float, default=0.2, help="Probability of applying mixup")
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="Alpha parameter for the Beta distribution")
    return parser.parse_args()


def main(args: Namespace):
    image = Image.open(args.image)
    image = np.array(image)
    image = torch.from_numpy(image)
    image = image.to(torch.float32) / torch.iinfo(image.dtype).max
    if image.ndim == 2:
        image.unsqueeze_(-1)
    image = image.permute(2, 0, 1)

    # Create a length-2 batch by flipping the image horizontally
    image = torch.stack([image, image.flip(2)], dim=0)
    image = image.repeat(args.batch_size, 1, 1, 1)

    torch.manual_seed(args.seed)
    if args.cuda:
        image = image.cuda()

    torch.random.manual_seed(args.seed)
    torch.cuda.synchronize()
    start = timeit.default_timer()
    if args.cuda and args.fused:
        out = fused_mixup(image, args.prob, args.alpha, args.seed)
    else:
        weight = sample_mixup_parameters(image.shape[0], mixup_prob=args.prob, mixup_alpha=args.alpha, device=image.device)
        out = mixup(image, weight)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    print(f"Time taken: {end - start} seconds")
    out = out.cpu()
    out = out.mul_(255).clamp_(0, 255).to(torch.uint8)
    grid = make_grid(out)
    grid = grid.permute(1, 2, 0)
    grid = Image.fromarray(grid.numpy())
    grid.save("mixup_preview.png")


if __name__ == "__main__":
    main(parse_args())
