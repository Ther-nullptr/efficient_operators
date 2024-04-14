import torch
import triton
import triton.language as tl

MAX_FUSED_SIZE = 65536
ROPE_GROUP_SIZE = 4
next_power_of_2 = triton.next_power_of_2


def rope_forward(x: torch.Tensor, cos, sin, position_ids):
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    
    half = x.shape[-1] // 2
    RH_x = torch.cat((-x[..., half:], x[..., :half]), dim = -1)
    x *= cos
    x.addcmul_(RH_x, sin)
    return x


def rope_backward(dx: torch.Tensor, cos, sin):
    half = dx.shape[-1] // 2
    RH_dx = torch.cat((-dx[..., half:], dx[..., :half]), dim = -1)
    dx *= cos
    dx.addcmul_(RH_dx, sin)
    return dx
