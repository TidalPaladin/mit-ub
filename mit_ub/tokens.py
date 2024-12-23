from typing import Sequence, cast

import torch
import torch.nn.functional as F
from torch import Tensor


def mask_is_ragged(mask: Tensor) -> bool:
    r"""Checks if the mask is ragged.

    A mask is ragged if the number of unmasked tokens is not the same for all batch elements.

    Args:
        mask: Mask tensor to check

    Shapes:
        mask - :math:`(N, L)` where :math:`L` is the number of tokens
    """
    counts = mask.sum(dim=-1)
    return cast(bool, (counts != counts[0]).any())


def _apply_with_fill(mask: Tensor, x: Tensor, fill_value: float | Tensor) -> Tensor:
    N, L, _ = x.shape
    fill_value = fill_value.type_as(x) if isinstance(fill_value, Tensor) else fill_value
    mask = mask.view(N, L, 1)
    return torch.where(mask, x, fill_value)


def _apply_non_ragged(mask: Tensor, x: Tensor) -> Tensor:
    N, _, D = x.shape
    return x[mask].view(N, -1, D)


def _apply_ragged(mask: Tensor, x: Tensor, padding_value: float | Tensor) -> Tensor:
    N, _, D = x.shape

    # Build indices where we want to put non-padding values
    unmasked_count = mask.sum(dim=-1)
    max_tokens = cast(int, unmasked_count.max())
    indices = torch.stack(
        [
            torch.arange(N, device=x.device).view(N, 1).expand(-1, max_tokens),
            torch.arange(max_tokens, device=x.device).view(1, max_tokens).expand(N, -1),
        ],
        dim=-1,
    )
    indices = indices[indices[..., -1] < unmasked_count.view(-1, 1)]

    if isinstance(padding_value, Tensor):
        o = padding_value.type_as(x).broadcast_to((N, max_tokens, D))
    else:
        o = x.new_full((N, max_tokens, D), padding_value)
    return torch.index_put(o, indices.unbind(-1), x[mask])


def apply_mask(
    mask: Tensor,
    x: Tensor,
    fill_value: float | Tensor | None = None,
    padding_value: float | Tensor = 0,
) -> Tensor:
    r"""Apply the mask to tokens.

    It is expected that ``True`` indicates an unmasked token and ``False`` indicates a masked token.
    When ``fill_value=None`` and the mask is ragged, the result is padded to match the number of tokens in the
    largest batch element. Padding is done using ``padding_value`` and is applied to the end of each batch sequence.

    Args:
        mask: Mask tensor
        x: Input tensor
        fill_value: Value to fill the masked tokens with. If ``None``, the masked tokens are removed.
        padding_value: Padding value used when the mask is ragged.

    Shapes:
        mask - :math:`(N, L)` where :math:`L` is the number of tokens
        x - :math:`(N, L, D)`
        Output - :math:`(N, L', D)` where :math:`L'` is the number of output tokens

    Returns:
        Tensor with the mask applied
    """
    if x.shape[:-1] != mask.shape:
        raise ValueError(
            f"Mask and input must match in all dimensions except the last: {x.shape} != {mask.shape}"
        )  # pragma: no cover

    if fill_value is not None:
        return _apply_with_fill(mask, x, fill_value)
    elif not mask_is_ragged(mask):
        return _apply_non_ragged(mask, x)
    else:
        return _apply_ragged(mask, x, padding_value)


def unapply_mask(mask: Tensor, x: Tensor) -> Tensor:
    assert not mask_is_ragged(mask), "Cannot unapply ragged mask"
    B, L = mask.shape
    D = x.shape[-1]
    result = x.new_zeros((B, L, D))
    result.masked_scatter_(mask.view(B, L, 1), x)
    return result


def create_mask(
    size: Sequence[int],
    mask_ratio: float,
    batch_size: int = 1,
    scale: int = 1,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    r"""Create a token mask for an input.

    Args:
        size: Size of the token grid
        mask_ratio: Ratio of tokens to mask
        batch_size: Size of the batch
        scale: Dilates the mask by this factor. For example, if ``scale == 2`` and ``len(size) == 2``,
            masking will be done in (2x2) contiguous blocks.
        device: Device to create the mask on

    Shapes:
        Output - :math:`(N, L)` where :math:`L` is the product of ``size`` and ``N`` is ``batch_size``

    Raises:
        ValueError: If ``mask_ratio`` is not in the range (0, 1)
        ValueError: If ``scale`` is less than 1

    Returns:
        Token mask tensor, with ``True`` indicating an unmasked token and ``False`` indicating a masked token
    """
    if not 0 < mask_ratio < 1.0:
        raise ValueError(f"Invalid `mask_ratio` {mask_ratio}")  # pragma: no cover
    if scale < 1:
        raise ValueError(f"Invalid `scale` {scale}")  # pragma: no cover
    if any(dim <= scale for dim in size):
        raise ValueError(
            f"Invalid `size` {size} for `scale` {scale}. Size must be greater than scale"
        )  # pragma: no cover

    # When scale > 1, reformulate the problem as a recursive call over smaller mask and upsample
    if scale > 1:
        scaled_size = tuple(s // scale for s in size)
        mask = create_mask(scaled_size, mask_ratio, batch_size, scale=1, device=device)
        mask = mask.view(batch_size, 1, *scaled_size).float()
        mask = F.interpolate(mask, scale_factor=scale, mode="nearest")
        mask = mask.view(batch_size, -1).bool()
        return mask

    # Compute the total number of tokens and number of masked tokens
    Lmask = cast(int, torch.tensor(size, dtype=torch.long).prod())
    num_masked_tokens = cast(Tensor, (Lmask * mask_ratio)).round_().long()
    num_masked_tokens = cast(int, num_masked_tokens)

    # initialize empty mask
    mask = torch.full((batch_size, Lmask), True, device=device, dtype=torch.bool)

    # select exactly num_masked_tokens random locations, with unique locations for each batch element
    token_idx = torch.randperm(Lmask).view(1, Lmask).expand(batch_size, -1)
    indices = torch.argsort(torch.rand_like(token_idx, dtype=torch.float32), dim=-1)[..., :num_masked_tokens]
    token_idx = torch.gather(token_idx, dim=-1, index=indices)
    assert token_idx.shape == (batch_size, num_masked_tokens)
    batch_idx = torch.arange(batch_size).view(batch_size, 1).expand(-1, num_masked_tokens)

    # update mask based on chosen locations
    mask[batch_idx.flatten(), token_idx.flatten()] = False
    return mask


def generate_non_overlapping_mask(mask1: Tensor, p1: float, p2: float) -> Tensor:
    """Generates a second mask that does not overlap with the first mask.

    Args:
        mask1: First mask tensor of shape (B, L) where B is batch size and L is sequence length
        p1: Ratio of tokens masked in first mask, must be in range (0,1)
        p2: Ratio of tokens to mask in second mask, must be in range (0,1)

    Shapes:
        mask1 - :math:`(B, L)` where :math:`B` is batch size and :math:`L` is sequence length
        Output - :math:`(B, L)` with same shape as input mask

    Raises:
        ValueError: If p1 + p2 > 1, indicating masks cannot be non-overlapping

    Returns:
        Second mask tensor with same shape as input, with True indicating masked tokens
    """
    B, L = mask1.shape
    n = int(L * p1)
    m = int(L * p2)
    if n + m > L:
        raise ValueError("Cannot satisfy non-overlap constraint.")

    # Generate random values in the shape of the mask
    x = torch.rand(B, L)

    # Fill values in the first mask with a high value
    x[mask1] = 2.0

    # Sort the values and take the top m values
    idx = x.argsort(dim=1)[:, :m]

    # Create the second mask
    mask2 = torch.zeros_like(mask1)
    rows = torch.arange(B).unsqueeze_(-1)
    mask2[rows, idx] = True
    return mask2
