from typing import Any, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from triton_helpers.ops import relu2 as relu2_op
from triton_helpers.ops import relu2_bwd as relu2_bwd_op


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 16}),
        triton.Config({"BLOCK": 32}),
        triton.Config({"BLOCK": 64}),
        triton.Config({"BLOCK": 128}),
        triton.Config({"BLOCK": 256}),
        triton.Config({"BLOCK": 512}),
        triton.Config({"BLOCK": 1024}),
    ],
    key=["L"],
)
@triton.jit
def relu2_kernel(
    # fmt: off
    x_p, o_p,
    stride_x: int, stride_o: int,
    L,
    BLOCK: tl.constexpr,
    # fmt: on
):
    pid = tl.program_id(0)
    x_p += pid * BLOCK * stride_x
    o_p += pid * BLOCK * stride_o

    X_block_ptr = tl.make_block_ptr(
        x_p,
        shape=(L,),
        strides=(stride_x,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    x = tl.load(X_block_ptr, boundary_check=(0,))
    x = relu2_op(x).to(x.dtype)

    O_block_ptr = tl.make_block_ptr(
        o_p,
        shape=(L,),
        strides=(stride_o,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tl.store(O_block_ptr, x, boundary_check=(0,))


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 16}),
        triton.Config({"BLOCK": 32}),
        triton.Config({"BLOCK": 64}),
        triton.Config({"BLOCK": 128}),
        triton.Config({"BLOCK": 256}),
        triton.Config({"BLOCK": 512}),
        triton.Config({"BLOCK": 1024}),
    ],
    key=["L"],
)
@triton.jit
def relu2_bwd_kernel(
    # fmt: off
    x_p, do_p, dx_p,
    stride_x: int, stride_do: int, stride_dx: int,
    L,
    BLOCK: tl.constexpr,
    # fmt: on
):
    pid = tl.program_id(0)
    x_p += pid * BLOCK * stride_x
    do_p += pid * BLOCK * stride_do
    dx_p += pid * BLOCK * stride_dx

    X_block_ptr = tl.make_block_ptr(
        x_p,
        shape=(L,),
        strides=(stride_x,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    DO_block_ptr = tl.make_block_ptr(
        do_p,
        shape=(L,),
        strides=(stride_do,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )

    x = tl.load(X_block_ptr, boundary_check=(0,))
    do = tl.load(DO_block_ptr, boundary_check=(0,))
    dx = relu2_bwd_op(x, do).to(x.dtype)

    DX_block_ptr = tl.make_block_ptr(
        dx_p,
        shape=(L,),
        strides=(stride_dx,),
        offsets=(0,),
        block_shape=(BLOCK,),
        order=(0,),
    )
    tl.store(DX_block_ptr, dx, boundary_check=(0,))


class ReLU2(Function):

    @torch.amp.custom_fwd(device_type="cuda")  # type: ignore
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        L = x.numel()

        def grid(META):
            return (triton.cdiv(L, META["BLOCK"]),)

        o = torch.empty_like(x)
        cast(Any, relu2_kernel)[grid](
            # fmt: off
            x, o,
            x.stride(-1), o.stride(-1),
            L,
            # fmt: on
        )

        ctx.save_for_backward(x)
        return o

    @torch.amp.custom_bwd(device_type="cuda")  # type: ignore
    @staticmethod
    def backward(ctx, do) -> Tensor:
        x = ctx.saved_tensors[0]
        L = x.numel()

        def grid(META):
            return (triton.cdiv(L, META["BLOCK"]),)

        dx = torch.empty_like(x)
        cast(Any, relu2_bwd_kernel)[grid](
            # fmt: off
            x, do, dx,
            x.stride(-1), do.stride(-1), dx.stride(-1),
            L,
            # fmt: on
        )
        return dx


def relu2(x: Tensor) -> Tensor:
    r"""Computes squared ReLU of an input."""
    if x.device.type == "cuda":
        return cast(Tensor, ReLU2.apply(x))
    else:
        y = F.relu(x)
        return y * y
