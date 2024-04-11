import math
import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F

def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def head_to_hidden_shape(x: torch.Tensor):
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)


class MixedSparseFullAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w_q: torch.Tensor,
        b_q: torch.Tensor,
        w_q_quant_state: tuple,
        w_q_lora_a: torch.Tensor,
        w_q_lora_b: torch.Tensor,
        ####################################
        w_k: torch.Tensor,
        b_k: torch.Tensor,
        w_k_quant_state: tuple,
        w_k_lora_a: torch.Tensor,
        w_k_lora_b: torch.Tensor,
        ####################################
        w_v: torch.Tensor,
        b_v: torch.Tensor,
        w_v_quant_state: tuple,
        w_v_lora_a: torch.Tensor,
        w_v_lora_b: torch.Tensor,
        ####################################
        w_o: torch.Tensor,
        b_o: torch.Tensor,
        w_o_quant_state: tuple,
        w_o_lora_a: torch.Tensor,
        w_o_lora_b: torch.Tensor,
        ####################################
        attention_mask: torch.Tensor,
        use_rotary_pos_enc: bool,
        num_heads: int,
    ):
        # compute q,k,v
        # forward process: q_proj
        w_q_dequant = F.dequantize_nf4(w_q, w_q_quant_state).to(x.dtype).t()
        q_main = x @ w_q_dequant + b_q if b_q is not None else x @ w_q_dequant
        q_lora_a = x @ w_q_lora_a
        q_lora = q_lora_a @ w_q_lora_b
        q = q_main + q_lora

        # forward process: k_proj
        w_k_dequant = F.dequantize_nf4(w_k, w_k_quant_state).to(x.dtype).t()
        k_main = x @ w_k_dequant + b_k if b_k is not None else x @ w_k_dequant
        k_lora_a = x @ w_k_lora_a
        k_lora = k_lora_a @ w_k_lora_b
        k = k_main + k_lora

        # forward process: v_proj
        w_v_dequant = F.dequantize_nf4(w_v, w_v_quant_state).to(x.dtype).t()
        v_main = x @ w_v_dequant + b_v if b_v is not None else x @ w_v_dequant
        v_lora_a = x @ w_v_lora_a
        v_lora = v_lora_a @ w_v_lora_b
        v = v_main + v_lora
        
        # reshape
        q = hidden_to_head_shape(q, num_heads)
        k = hidden_to_head_shape(k, num_heads)
        v = hidden_to_head_shape(v, num_heads)

        # TODO: apply pos_emb to q & k
        if use_rotary_pos_enc:
            pass

        # q,k,v: [bsz, num_heads, q_len, head_dim]
        # notice forward process no need to drop heads
        bsz, num_heads, q_len, head_dim = q.shape

        # forward: S = Q @ K.T / sqrt(d_k)
        s = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        # apply mask
        if attention_mask is not None:
            s = s + attention_mask

        # forward: softmax
        a = torch.softmax(s, dim=-1)  # [bsz, num_heads, q_len, q_len]

        # forward: O = A @ V
        o = a @ v
        
        # reshape
        o = head_to_hidden_shape(o)

        # forward process: o_proj
        w_o_dequant = F.dequantize_nf4(w_o, w_o_quant_state).to(x.dtype).t()
        o_main = o @ w_o_dequant + b_o if b_o is not None else o @ w_o_dequant
        o_lora_a = o @ w_o_lora_a
        o_lora = o_lora_a @ w_o_lora_b
        o_final = o_main + o_lora

        ctx.save_for_backward(
            x,
            q_lora_a,
            k_lora_a,
            v_lora_a,
            q,
            k,
            v,
            a,
            o,
            o_lora_a,
            #######################
            w_q,
            b_q,
            w_q_lora_a,
            w_q_lora_b,
            #######################
            w_k,
            b_k,
            w_k_lora_a,
            w_k_lora_b,
            #######################
            w_v,
            b_v,
            w_v_lora_a,
            w_v_lora_b,
            #######################
            w_o,
            b_o,
            w_o_lora_a,
            w_o_lora_b,
        )

        ctx.quant_state = (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
        )

        ctx.q_shape = q.shape
        ctx.a_shape = a.shape
        ctx.q_device = q.device

        return o_final

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            x,
            q_lora_a,
            k_lora_a,
            v_lora_a,
            q,
            k,
            v,
            a,
            o,
            o_lora_a,
            w_q,
            b_q,
            w_q_lora_a,
            w_q_lora_b,
            w_k,
            b_k,
            w_k_lora_a,
            w_k_lora_b,
            w_v,
            b_v,
            w_v_lora_a,
            w_v_lora_b,
            w_o,
            b_o,
            w_o_lora_a,
            w_o_lora_b,
        ) = ctx.saved_tensors

        w_q_quant_state, w_k_quant_state, w_v_quant_state, w_o_quant_state = (
            ctx.quant_state
        )

        bsz, num_heads, q_len, head_dim = ctx.q_shape

        # backward of o_proj
        grad_w_o_lora_a = o.mT @ (grad_output @ w_o_lora_b.T)
        grad_w_o_lora_b = o_lora_a.mT @ grad_output
        w_o_dequant = F.dequantize_nf4(w_o, w_o_quant_state).to(grad_output.dtype).t()
        grad_o = grad_output @ w_o_dequant.T + grad_output @ w_o_lora_b.T @ w_o_lora_a.T
        
        # reshape
        grad_o = hidden_to_head_shape(grad_o, num_heads)

        # backward of second GEMM: O = A @ V
        # d L / d V = A.T @ d L / d O
        grad_v = a.transpose(-2, -1) @ grad_o
        grad_a = grad_o @ v.transpose(-2, -1)

        # backward of softmax
        grad_s = (grad_a - (grad_a * a).sum(dim=-1, keepdims=True)) * a

        # backward of first GEMM: S = Q @ K.T / sqrt(d_k)
        grad_s = grad_s / math.sqrt(head_dim)
        # d L / d K = (d L / d S)^T @ Q
        grad_k = grad_s.transpose(-2, -1) @ q
        # d L / d Q = d L / d S @ K
        grad_q = grad_s @ k

        # TODO apply pos_emb to q & k
        
        # reshape
        grad_q = head_to_hidden_shape(grad_q)
        grad_k = head_to_hidden_shape(grad_k)
        grad_v = head_to_hidden_shape(grad_v)
        
        q = head_to_hidden_shape(q)
        k = head_to_hidden_shape(k)
        v = head_to_hidden_shape(v)

        # backward of q_proj
        grad_w_q_lora_a = q.mT @ (grad_q @ w_q_lora_b.T)
        grad_w_q_lora_b = q_lora_a.mT @ grad_q
        w_q_dequant = F.dequantize_nf4(w_q, w_q_quant_state).to(grad_output.dtype).t()
        grad_x = grad_q @ w_q_dequant.T + grad_q @ w_q_lora_b.T @ w_q_lora_a.T

        # backward of k_proj
        grad_w_k_lora_a = k.mT @ (grad_k @ w_k_lora_b.T)
        grad_w_k_lora_b = k_lora_a.mT @ grad_k
        w_k_dequant = F.dequantize_nf4(w_k, w_k_quant_state).to(grad_output.dtype).t()
        grad_x += grad_k @ w_k_dequant.T + grad_k @ w_k_lora_b.T @ w_k_lora_a.T

        # backward of v_proj
        grad_w_v_lora_a = v.mT @ (grad_v @ w_v_lora_b.T)
        grad_w_v_lora_b = v_lora_a.mT @ grad_v
        w_v_dequant = F.dequantize_nf4(w_v, w_v_quant_state).to(grad_output.dtype).t()
        grad_x += grad_v @ w_v_dequant.T + grad_v @ w_v_lora_b.T @ w_v_lora_a.T

        return (
            grad_x,
            None,
            None,
            None,
            grad_w_q_lora_a,
            grad_w_q_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_k_lora_a,
            grad_w_k_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_v_lora_a,
            grad_w_v_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_o_lora_a,
            grad_w_o_lora_b,
            ####################################
            None,
            None,
            None
        )


class MixedSparseFullAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        super(MixedSparseFullAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.iteration = 0

    def forward(
        self,
        input: torch.Tensor,
        q_proj_base: bnb.nn.modules.Linear4bit,
        q_proj_lora_a: torch.nn.Linear,
        q_proj_lora_b: torch.nn.Linear,
        k_proj_base: bnb.nn.modules.Linear4bit,
        k_proj_lora_a: torch.nn.Linear,
        k_proj_lora_b: torch.nn.Linear,
        v_proj_base: bnb.nn.modules.Linear4bit,
        v_proj_lora_a: torch.nn.Linear,
        v_proj_lora_b: torch.nn.Linear,
        o_proj_base: bnb.nn.modules.Linear4bit,
        o_proj_lora_a: torch.nn.Linear,
        o_proj_lora_b: torch.nn.Linear,
        use_rotary_pos_enc: bool,
        attention_mask: torch.Tensor,
    ):
        self.iteration += 1
        return MixedSparseFullAttentionFunc.apply(
            input,
            q_proj_base.weight,
            q_proj_base.bias,
            q_proj_base.weight.quant_state,
            q_proj_lora_a.default.weight.T,
            q_proj_lora_b.default.weight.T,
            ####################################
            k_proj_base.weight,
            k_proj_base.bias,
            k_proj_base.weight.quant_state,
            k_proj_lora_a.default.weight.T,
            k_proj_lora_b.default.weight.T,
            ####################################
            v_proj_base.weight,
            v_proj_base.bias,
            v_proj_base.weight.quant_state,
            v_proj_lora_a.default.weight.T,
            v_proj_lora_b.default.weight.T,
            ####################################
            o_proj_base.weight,
            o_proj_base.bias,
            o_proj_base.weight.quant_state,
            o_proj_lora_a.default.weight.T,
            o_proj_lora_b.default.weight.T,
            ####################################
            attention_mask,
            use_rotary_pos_enc,
            self.num_heads,
        )
