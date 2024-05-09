import torch
from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward
from .compress_function import (
    true_divide_outliner_suboutlinear_svd_compress,
    true_divide_outliner_suboutlinear_svd_decompress,
    get_statistics
)

class EfficientFlashAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        # for calibration
        outliner_ratio,
        sub_outliner_ratio,
        sub_outliner_bit,
        sub_outlier_quantize_method,
        iteration,
        q_static_value,
        k_static_value,
        v_static_value,
        o_static_value
    ):
        
        # qkv: [B, L, 3, NH, HD]
        num_heads = qkv.shape[-2]
        ctx.num_heads = num_heads
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            qkv[:, 0],
            qkv[:, 1],
            qkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        
        if iteration < 10:
            q_outliner, _, scale_q = get_statistics(q, iteration, outliner_ratio, sub_outliner_ratio, sub_outliner_bit, sub_outlier_quantize_method)
            k_outliner, _, scale_k = get_statistics(k, iteration, outliner_ratio, sub_outliner_ratio, sub_outliner_bit, sub_outlier_quantize_method)
            v_outliner, _, scale_v = get_statistics(v, iteration, outliner_ratio, sub_outliner_ratio, sub_outliner_bit, sub_outlier_quantize_method)
            o_outliner, _, scale_o = get_statistics(out_padded, iteration, outliner_ratio, sub_outliner_ratio, sub_outliner_bit, sub_outlier_quantize_method)
        else:
            q_outliner, _, scale_q = q_static_value
            k_outliner, _, scale_k = k_static_value
            v_outliner, _, scale_v = v_static_value
            o_outliner, _, scale_o = o_static_value
        
        q_outliner_compressed, q_sub_outliner_compressed, q_scale = true_divide_outliner_suboutlinear_svd_compress(q, q_outliner, q_scale, sub_outliner_bit, sub_outliner_ratio)
        k_outliner_compressed, k_sub_outliner_compressed, k_scale = true_divide_outliner_suboutlinear_svd_compress(k, k_outliner, k_scale, sub_outliner_bit, sub_outliner_ratio)
        v_outliner_compressed, v_sub_outliner_compressed, v_scale = true_divide_outliner_suboutlinear_svd_compress(v, v_outliner, v_scale, sub_outliner_bit, sub_outliner_ratio)
        o_outliner_compressed, o_sub_outliner_compressed, o_scale = true_divide_outliner_suboutlinear_svd_compress(out, o_outliner, o_scale, sub_outliner_bit, sub_outliner_ratio)
        
        ctx.dropout_p = dropout_p
        ctx.max_seqlen = max_seqlen
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.sub_outliner_bit = sub_outliner_bit
        
        ctx.mark_non_differentiable(q_outliner, k_outliner, v_outliner, o_outliner)
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens, rng_state)
        ctx.save_for_backward(q_outliner_compressed, q_sub_outliner_compressed, q_scale, k_outliner_compressed, k_sub_outliner_compressed, k_scale, v_outliner_compressed, v_sub_outliner_compressed, v_scale, o_outliner_compressed, o_sub_outliner_compressed, o_scale, softmax_lse, cu_seqlens, rng_state)
        return out, q_outliner, k_outliner, v_outliner, o_outliner, q_scale, k_scale, v_scale, o_scale

    @staticmethod
    def backward(ctx, dout, *args):
        q_outliner_compressed, q_sub_outliner_compressed, q_scale, k_outliner_compressed, k_sub_outliner_compressed, k_scale, v_outliner_compressed, v_sub_outliner_compressed, v_scale, o_outliner_compressed, o_sub_outliner_compressed, o_scale, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        q = true_divide_outliner_suboutlinear_svd_decompress(q_outliner_compressed, q_sub_outliner_compressed, ctx.sub_outliner_bit, q_scale, True, ctx.num_heads)
        k = true_divide_outliner_suboutlinear_svd_decompress(k_outliner_compressed, k_sub_outliner_compressed, ctx.sub_outliner_bit, k_scale, True, ctx.num_heads)
        v = true_divide_outliner_suboutlinear_svd_decompress(v_outliner_compressed, v_sub_outliner_compressed, ctx.sub_outliner_bit, v_scale, True, ctx.num_heads)
        out = true_divide_outliner_suboutlinear_svd_decompress(o_outliner_compressed, o_sub_outliner_compressed, ctx.sub_outliner_bit, o_scale, True, ctx.num_heads)
        
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        _flash_attn_varlen_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dqkv[:, 0],
            dqkv[:, 1],
            dqkv[:, 2],
            cu_seqlens,
            cu_seqlens,
            ctx.max_seqlen,
            ctx.max_seqlen,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
    
    
class EfficientFlashAttnVarlenQKVPacked(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio: float = 0.01,
        sub_outliner_ratio: float = 0.2, #! initialize
        sub_outliner_bit: int = 8,
        sub_outlier_quantize_method: str = 'per-tensor',
        rank: int = 16,
    ):
        super(EfficientFlashAttnVarlenQKVPacked, self).__init__()
        self.outliner_ratio = outliner_ratio
        self.sub_outliner_ratio = sub_outliner_ratio
        self.sub_outliner_bit = sub_outliner_bit
        self.sub_outlier_quantize_method = sub_outlier_quantize_method
        self.rank = rank
        self.iteration = 0
        
        self.q_static_value = [None, None, None]
        self.k_static_value = [None, None, None]
        self.v_static_value = [None, None, None]
        self.o_static_value = [None, None, None]
        
    def forward(
        self,
        qkv, 
        cu_seqlens,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ):
        out, q_outliner, k_outliner, v_outliner, o_outliner, q_scale, k_scale, v_scale, o_scale = EfficientFlashAttnVarlenQKVPackedFunc.apply(
            qkv,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            # for calibration
            self.outliner_ratio,
            self.sub_outliner_ratio,
            self.sub_outliner_bit,
            self.sub_outlier_quantize_method,
            self.iteration,
            self.q_static_value,
            self.k_static_value,
            self.v_static_value,
            self.o_static_value
        )
        if self.iteration <= 10:
            self.q_static_value[0] = (
                q_outliner
                if self.q_static_value[0] is None
                else (self.iteration * self.q_static_value[0] + q_outliner)
                / (self.iteration + 1)
            )
            self.k_static_value[0] = (
                k_outliner
                if self.k_static_value[0] is None
                else (self.iteration * self.k_static_value[0] + k_outliner)
                / (self.iteration + 1)
            )
            self.v_static_value[0] = (
                v_outliner
                if self.v_static_value[0] is None
                else (self.iteration * self.v_static_value[0] + v_outliner)
                / (self.iteration + 1)
            )
            self.o_static_value[0] = (
                o_outliner
                if self.o_static_value[0] is None
                else (self.iteration * self.o_static_value[0] + o_outliner)
                / (self.iteration + 1)
            )
            
            self.q_static_value[2] = (
                q_scale
                if self.q_static_value[2] is None
                else (self.iteration * self.q_static_value[2] + q_scale)
                / (self.iteration + 1)
            )
            self.k_static_value[2] = (
                k_scale
                if self.k_static_value[2] is None
                else (self.iteration * self.k_static_value[2] + k_scale)
                / (self.iteration + 1)
            )
            self.v_static_value[2] = (
                v_scale
                if self.v_static_value[2] is None
                else (self.iteration * self.v_static_value[2] + v_scale)
                / (self.iteration + 1)
            )
            self.o_static_value[2] = (
                o_scale
                if self.o_static_value[2] is None
                else (self.iteration * self.o_static_value[2] + o_scale)
                / (self.iteration + 1)
            )
        
        self.iteration += 1
        
        return out