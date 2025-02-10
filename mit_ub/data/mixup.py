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



@torch.no_grad()
def _get_weights(batch_size: int, mixup_prob: float = 0.2, mixup_alpha: float = 1.0, seed: int | None = None) -> Tensor:
    if _mixup_cuda is None:
        raise RuntimeError("MixUp is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _mixup_cuda.get_weights(batch_size, mixup_prob, mixup_alpha, seed)


@torch.no_grad()
def mixup(x: Tensor, mixup_prob: float = 0.2, mixup_alpha: float = 1.0, seed: int | None = None) -> Tensor:
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
        ctx.mixup_prob = mixup_prob
        ctx.mixup_alpha = mixup_alpha
        ctx.seed = seed
        loss, denom, max_val = _mixup_cuda.cross_entropy_mixup_fwd(logits, labels, mixup_prob, mixup_alpha, seed)
        ctx.save_for_backward(logits, labels, denom, max_val)
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        logits, labels, denom, max_val = ctx.saved_tensors
        mixup_prob = ctx.mixup_prob
        mixup_alpha = ctx.mixup_alpha
        seed = ctx.seed
        grad = _mixup_cuda.cross_entropy_mixup_bwd(logits, labels, denom, max_val, grad_output, mixup_prob, mixup_alpha, seed)
        return grad, None, None, None, None


def cross_entropy_mixup(logits: Tensor, labels: Tensor, seed: int, mixup_prob: float = 0.2, mixup_alpha: float = 1.0) -> Tensor:
    """Cross entropy loss with MixUp.

    Applies MixUp to the target labels by mixing them with a shifted version of the batch.
    The mixing weight is sampled from a Beta distribution. If a label is -1 (unknown),
    that sample is excluded from the loss calculation.

    This implementation avoids materializing the one-hot encoded labels, instead using
    a single kernel with online softmax to compute the loss.

    Args:
        logits: The predicted class logits
        labels: The target class labels
        seed: Random seed for reproducibility. Should match the seed used when applying MixUp
            to the input.
        mixup_prob: Probability of applying mixup to each sample
        mixup_alpha: Alpha parameter for Beta distribution used to sample mixup weight

    Returns:
        The cross entropy loss for each sample in the batch. Samples with unknown
        labels (-1) will have a loss value of -1.

    Shapes:
        - logits: :math:`(N, C)`
        - labels: :math:`(N,)`
        - Output: :math:`(N,)`
    """
    if _mixup_cuda is None:
        raise RuntimeError("MixUp is not available on this system")
    return CrossEntropyMixup.apply(logits, labels, mixup_prob, mixup_alpha, seed)


class BCEMixup(Function):
    @staticmethod
    def forward(ctx, logits: Tensor, labels: Tensor, mixup_prob: float, mixup_alpha: float, seed: int) -> Tensor:
        ctx.mixup_prob = mixup_prob
        ctx.mixup_alpha = mixup_alpha
        ctx.seed = seed
        loss = _mixup_cuda.bce_mixup_fwd(logits, labels, mixup_prob, mixup_alpha, seed)
        ctx.save_for_backward(logits, labels)
        return loss

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None, None, None, None]:
        logits, labels = ctx.saved_tensors
        mixup_prob = ctx.mixup_prob
        mixup_alpha = ctx.mixup_alpha
        seed = ctx.seed
        grad = _mixup_cuda.bce_mixup_bwd(logits, labels, grad_output, mixup_prob, mixup_alpha, seed)
        return grad, None, None, None, None


def bce_mixup(logits: Tensor, labels: Tensor, seed: int, mixup_prob: float = 0.2, mixup_alpha: float = 1.0) -> Tensor:
    """BCE loss with MixUp.

    Applies MixUp to the target labels by mixing them with a shifted version of the batch.
    The mixing weight is sampled from a Beta distribution. If a label is -1 (unknown),
    that sample is excluded from the loss calculation.

    This implementation avoids materializing the one-hot encoded labels, instead using
    a single kernel with online softmax to compute the loss.

    Args:
        logits: The predicted class logits
        labels: The target class labels
        seed: Random seed for reproducibility. Should match the seed used when applying MixUp
            to the input.
        mixup_prob: Probability of applying mixup to each sample
        mixup_alpha: Alpha parameter for Beta distribution used to sample mixup weight

    Returns:
        The cross entropy loss for each sample in the batch. Samples with unknown
        labels (-1) will have a loss value of -1.

    Shapes:
        - logits: :math:`(N, ...)`
        - labels: :math:`(N, ...)`
        - Output: :math:`(N, ...)`
    """
    if _mixup_cuda is None:
        raise RuntimeError("MixUp is not available on this system")
    return BCEMixup.apply(logits, labels, mixup_prob, mixup_alpha, seed)


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
