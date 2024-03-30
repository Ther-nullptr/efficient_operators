import math
import torch
import bitsandbytes.functional as F

class MixedSparseAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor, 
        sparsity_ratio: float, maintain_heads: int, quantization: bool, iteration: int, layer_id: int
    ):
        # q,k,v: [bsz, num_heads, q_len, head_dim]
        # notice forward process no need to drop heads
        bsz, num_heads, q_len, head_dim = q.shape

        # forward: S = Q @ K.T / sqrt(d_k)
        s = q @ k.transpose(-2, -1)
        # apply mask
        if attention_mask is not None:
            s = s + attention_mask

        # forward: softmax 
        a = torch.softmax(s, dim=-1) # [bsz, num_heads, q_len, q_len]

        # forward: O = A @ V
        o = a @ v

        # save for backward: q, k, v, a
        # firstly, compute the norm of each attention head
        norm = torch.norm(o.reshape(bsz, num_heads, -1), dim=-1)
        actual_maintain_heads = min(int(sparsity_ratio * num_heads), maintain_heads)
        # record the heads with minimum norm
        norm_data, min_indices = norm.topk(actual_maintain_heads, dim=-1, largest=True)

        # save the selected heads
        q_save = torch.zeros((bsz, actual_maintain_heads, q_len, head_dim), dtype=q.dtype, device=q.device)
        k_save = torch.zeros((bsz, actual_maintain_heads, q_len, head_dim), dtype=k.dtype, device=k.device)
        v_save = torch.zeros((bsz, actual_maintain_heads, q_len, head_dim), dtype=v.dtype, device=v.device)
        a_save = torch.zeros((bsz, actual_maintain_heads, q_len, q_len), dtype=a.dtype, device=a.device)

        for i in range(bsz):
            batch_idx = i
            _min_indices = min_indices[i]
            q_save[i] = q[batch_idx, _min_indices, :, :]
            k_save[i] = k[batch_idx, _min_indices, :, :]
            v_save[i] = v[batch_idx, _min_indices, :, :]
            a_save[i] = a[batch_idx, _min_indices, :, :]

        if quantization:
            q_save, q_quant_state = F.quantize_nf4(q_save)
            k_save, k_quant_state = F.quantize_nf4(k_save)
            v_save, v_quant_state = F.quantize_nf4(v_save)
            a_save, a_quant_state = F.quantize_nf4(a_save)
            ctx.quant_state_activation = q_quant_state, k_quant_state, v_quant_state, a_quant_state

        ctx.save_for_backward(q_save, k_save, v_save, a_save)
        ctx.quantization = quantization
        ctx.min_indices = min_indices
        ctx.q_shape = q.shape
        ctx.a_shape = a.shape
        ctx.q_device = q.device
        ctx.iteration = iteration
        ctx.layer_id = layer_id

        return o
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        min_indices = ctx.min_indices

        q_save, k_save, v_save, a_save = ctx.saved_tensors

        if ctx.quantization:
            q_quant_state, k_quant_state, v_quant_state, a_quant_state = ctx.quant_state_activation
            q_save = F.dequantize_nf4(q_save, q_quant_state)
            k_save = F.dequantize_nf4(k_save, k_quant_state)
            v_save = F.dequantize_nf4(v_save, v_quant_state)
            a_save = F.dequantize_nf4(a_save, a_quant_state)

        # convert the q, k, v, a to the original shape
        q = torch.zeros(ctx.q_shape, dtype=q_save.dtype, device=ctx.q_device)
        k = torch.zeros(ctx.q_shape, dtype=k_save.dtype, device=ctx.q_device)
        v = torch.zeros(ctx.q_shape, dtype=v_save.dtype, device=ctx.q_device)
        a = torch.zeros(ctx.a_shape, dtype=a_save.dtype, device=ctx.q_device)

        for i in range(q_save.shape[0]):
            q[i, min_indices[i], :, :] = q_save[i]
            k[i, min_indices[i], :, :] = k_save[i]
            v[i, min_indices[i], :, :] = v_save[i]
            a[i, min_indices[i], :, :] = a_save[i]

        # backward of second GEMM: O = A @ V
        # d L / d V = A.T @ d L / d O
        grad_v = a.transpose(-2, -1) @ grad_output
        grad_a = grad_output @ v.transpose(-2, -1)

        # backward of softmax
        grad_s = (grad_a - (grad_a * a).sum(dim=-1, keepdims=True)) * a

        # backward of first GEMM: S = Q @ K.T / sqrt(d_k)
        # d L / d K = (d L / d S)^T @ Q
        grad_k = grad_s.transpose(-2, -1) @ q
        # d L / d Q = d L / d S @ K
        grad_q = grad_s @ k

        # if ctx.iteration % 100 == 1:
        #     print(f'drop iteration {ctx.iteration}')
        #     torch.save(grad_output, f'gradient/grad_output_{ctx.layer_id}_{ctx.iteration}.pt')

        return grad_q, grad_k, grad_v, None, None, None, None, None, None


class MixedSparseAttention(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, quantization: bool = False, layer_id: int = 0):
        super(MixedSparseAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.quantization = quantization
        self.iteration = 0
        self.layer_id = layer_id

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor, sparsity_ratio: float, maintain_heads: int):
        self.iteration += 1
        return MixedSparseAttentionFunc.apply(q, k, v, attention_mask, sparsity_ratio, maintain_heads, self.quantization, self.iteration, self.layer_id)
    