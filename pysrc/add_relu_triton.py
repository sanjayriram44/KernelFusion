import torch
import triton
import triton.language as tl

@triton.jit
def add_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    
    val = x + bias
    out = tl.maximum(val, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)

def triton_add_relu(x: torch.Tensor, bias: torch.Tensor):
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA"
    assert x.is_contiguous() and bias.is_contiguous(), "Tensors must be contiguous"
    assert x.shape == bias.shape, "Tensors must have the same shape"

    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_relu_kernel[grid](
        x, bias, out, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out
