import torch
import triton
import triton.language as tl


@triton.jit
def _scale_shift_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    scale,
    shift,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * scale + shift
    tl.store(y_ptr + offsets, y, mask=mask)


def run(x, scale, shift, y):
    """Compute ``y = x * scale + shift`` into a preallocated CUDA tensor."""
    torch.cuda.set_device(x.device)
    n_elements = y.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    _scale_shift_kernel[grid](
        x,
        y,
        n_elements,
        scale,
        shift,
        BLOCK_SIZE=block_size,
    )
