import math
import torch
import typing
import bitsandbytes as bnb
import torch.nn.functional as F
import bitsandbytes.functional as BF

from layernorm_kernels import layernorm_forward, layernorm_backward
from rmsnorm_kernels import rmsnorm_forward, rmsnorm_backward
from rope_kernels import rope_forward, rope_backward

def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

def head_to_hidden_shape(x: torch.Tensor):
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)

class MixedSparseSingleLayerFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        #############attention part#############
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        ####################################
        cos: torch.Tensor,
        sin: torch.Tensor,
        ####################################
        w_q: torch.Tensor,
        b_q: torch.Tensor,
        w_q_quant_state: typing.Tuple,
        w_q_lora_a: torch.Tensor,
        w_q_lora_b: torch.Tensor,
        ####################################
        w_k: torch.Tensor,
        b_k: torch.Tensor,
        w_k_quant_state: typing.Tuple,
        w_k_lora_a: torch.Tensor,
        w_k_lora_b: torch.Tensor,
        ####################################
        w_v: torch.Tensor,
        b_v: torch.Tensor,
        w_v_quant_state: typing.Tuple,
        w_v_lora_a: torch.Tensor,
        w_v_lora_b: torch.Tensor,
        ####################################
        w_o: torch.Tensor,
        b_o: torch.Tensor,
        w_o_quant_state: typing.Tuple,
        w_o_lora_a: torch.Tensor,
        w_o_lora_b: torch.Tensor,
        #############mlp part#############
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        ####################################
        w_up: torch.Tensor,
        b_up: torch.Tensor,
        w_up_quant_state: typing.Tuple,
        w_up_lora_a: torch.Tensor,
        w_up_lora_b: torch.Tensor,
        ####################################
        w_down: torch.Tensor,
        b_down: torch.Tensor,
        w_down_state: typing.Tuple,
        w_down_lora_a: torch.Tensor,
        w_down_lora_b: torch.Tensor,
        ###############other################
        attention_mask: torch.Tensor,
        norm_mode: str,
        num_heads: int,
        use_rotary_pos_enc: bool,
        small_value_approx: bool,
        activation_forward: str,
        activation_backward: str,
    ):
        x_1_res = x # keep the original input for residual connection
        
        # layernorm or rmsnorm
        if norm_mode == "layernorm":
            x_after_norm_1, mean_1, rstd_1 = layernorm_forward(x, norm_weight_1, norm_bias_1, eps = 1e-10)
        else:
            x_after_norm_1, mean_1, rstd_1 = rmsnorm_forward(x, norm_weight_1, eps = 1e-10)
            
        # compute q,k,v
        # forward process: q_proj
        w_q_dequant = F.dequantize_nf4(w_q, w_q_quant_state).to(x_after_norm_1.dtype).t()
        q_main = x_after_norm_1 @ w_q_dequant + b_q if b_q is not None else x_after_norm_1 @ w_q_dequant
        q_lora_a = x_after_norm_1 @ w_q_lora_a
        q_lora = q_lora_a @ w_q_lora_b
        q = q_main + q_lora

        # forward process: k_proj
        w_k_dequant = F.dequantize_nf4(w_k, w_k_quant_state).to(x_after_norm_1.dtype).t()
        k_main = x_after_norm_1 @ w_k_dequant + b_k if b_k is not None else x_after_norm_1 @ w_k_dequant
        k_lora_a = x_after_norm_1 @ w_k_lora_a
        k_lora = k_lora_a @ w_k_lora_b
        k = k_main + k_lora

        # forward process: v_proj
        w_v_dequant = F.dequantize_nf4(w_v, w_v_quant_state).to(x_after_norm_1.dtype).t()
        v_main = x_after_norm_1 @ w_v_dequant + b_v if b_v is not None else x_after_norm_1 @ w_v_dequant
        v_lora_a = x_after_norm_1 @ w_v_lora_a
        v_lora = v_lora_a @ w_v_lora_b
        v = v_main + v_lora
        
        # reshape
        q = hidden_to_head_shape(q, num_heads)
        k = hidden_to_head_shape(k, num_heads)
        v = hidden_to_head_shape(v, num_heads)
        
        ctx.q_shape = q.shape
        
        # TODO: apply positional encoding
        if use_rotary_pos_enc:
            q = rope_forward(q, cos, sin)
            k = rope_forward(k, cos, sin)
            
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
        
        if small_value_approx:
            # save the main weight, then recompute when backward
            q = q_main
            k = k_main
            v = v_main
            
        # residual connection
        x_1_final = x_1_res + o_final
        x_2_res = x_1_final
        
        # layernorm or rmsnorm
        if norm_mode == "layernorm":
            x_after_norm_2, mean_2, rstd_2 = layernorm_forward(x_1_final, norm_weight_2, norm_bias_2, eps = 1e-10)
        else:
            x_after_norm_2, mean_2, rstd_2 = rmsnorm_forward(x_1_final, norm_weight_2, eps = 1e-10)
        
        # forward process: up_proj
        w_up_dequant = F.dequantize_4bit(w_up, w_up_quant_state).to(x_after_norm_2.dtype).t()
        y1_main = x_after_norm_2 @ w_up_dequant + b_up if b_up is not None else x_after_norm_2 @ w_up_dequant
        y1_lora_a = x_after_norm_2 @ w_up_lora_a
        y1_lora = y1_lora_a @ w_up_lora_b
        y1 = y1_main + y1_lora
        
        # apply activation function
        if activation_forward == "relu":
            x2 = torch.relu(y1)
        elif activation_forward == "silu":
            x2 = torch.silu(y1)
        elif activation_forward == "gelu":
            x2 = torch.gelu(y1)
            
        # forward process: down_proj
        w_down_dequant = F.dequantize_4bit(w_down, w_down_state).to(x1.dtype).t()
        y2_main = (
            x2 @ w_down_dequant + b_down if b_down is not None else x2 @ w_down_dequant
        )
        y2_lora_a = x2 @ w_down_lora_a
        y2_lora = y2_lora_a @ w_down_lora_b
        y2 = y2_main + y2_lora
        
        # residual connection
        x_2_final = x_2_res + y2
        
        ctx.save_for_backward(
            ### activations (attention) ###
            x_1_res,
            x_after_norm_1,
            q_lora_a,
            k_lora_a,
            v_lora_a,
            q,
            k,
            v,
            a,
            o,
            o_lora_a,
            cos,
            sin,
            ### activations (mlp) ###
            x_2_res,
            x_after_norm_2,
            y1,
            y1_lora_a,
            x2,
            y2_lora_a,
            ### weights (attention) ###
            norm_weight_1, 
            norm_bias_1,
            w_q,
            b_q,
            w_q_lora_a,
            w_q_lora_b,
            #**********************
            w_k,
            b_k,
            w_k_lora_a,
            w_k_lora_b,
            #**********************
            w_v,
            b_v,
            w_v_lora_a,
            w_v_lora_b,
            #**********************
            w_o,
            b_o,
            w_o_lora_a,
            w_o_lora_b,
            ### weights (mlp) ###
            norm_weight_2,
            norm_bias_2,
            #**********************
            w_up,
            b_up,
            w_up_lora_a,
            w_up_lora_b,
            #**********************
            w_down,
            b_down,
            w_down_lora_a,
            w_down_lora_b,
        )
        
        return x_2_final
    
    @staticmethod
    def backward(ctx, grad_output):
        pass