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

class MixedSparseSingleLayerNaiveFunc(torch.autograd.Function):
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
        up_main = x_after_norm_2 @ w_up_dequant + b_up if b_up is not None else x_after_norm_2 @ w_up_dequant
        up_lora_a = x_after_norm_2 @ w_up_lora_a
        up_lora = up_lora_a @ w_up_lora_b
        up = up_main + up_lora
        
        # apply activation function
        if activation_forward == "relu":
            fn = torch.relu(up)
        elif activation_forward == "silu":
            fn = torch.silu(up)
        elif activation_forward == "gelu":
            fn = torch.gelu(up)
            
        # forward process: down_proj
        w_down_dequant = F.dequantize_4bit(w_down, w_down_state).to(fn.dtype).t()
        down_main = (
            fn @ w_down_dequant + b_down if b_down is not None else fn @ w_down_dequant
        )
        down_lora_a = fn @ w_down_lora_a
        down_lora = down_lora_a @ w_down_lora_b
        down = down_main + down_lora
        
        # residual connection
        x_2_final = x_2_res + down
        
        ctx.save_for_backward(
            ### activations (attention) ###
            x_1_res,
            x_after_norm_1,
            mean_1,
            rstd_1,
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
            mean_2,
            rstd_2,
            up, #! this part can be a raw mask
            up_lora_a,
            fn,
            down_lora_a,
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
        ctx.quant_state = (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_up_quant_state,
            w_down_state,
        )
        ctx.norm_mode = norm_mode
        ctx.num_heads = num_heads
        
        return x_2_final
    
    @staticmethod
    def backward(ctx, grad_output):
        (
            w_q_quant_state,
            w_k_quant_state,
            w_v_quant_state,
            w_o_quant_state,
            w_up_quant_state,
            w_down_quant_state,
        ) = ctx.quant_state
        
        (
            ### activations (attention) ###
            x_1_res,
            x_after_norm_1,
            mean_1,
            rstd_1,
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
            mean_2,
            rstd_2,
            up, #! this part can be a raw mask
            up_lora_a,
            fn,
            down_lora_a,
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
        ) = ctx.saved_tensors
        
        # down proj part
        # d L / d w_down_lora_a = x2.T @ d L / d y2 @ w_down_lora_b.T
        grad_w_down_lora_a = fn.mT @ (grad_output @ w_down_lora_b.T)
        # d L / d w_down_lora_b = y2_lora_a.T @ d L / d y2
        grad_w_down_lora_b = down_lora_a.mT @ grad_output
        # d L / d x2 = d L / d y2 @ w_down.T + d L / d y2 @ w_down_lora_b.T @ w_down_lora_a.T
        w_down_dequant = (
            F.dequantize_4bit(w_down, w_down_quant_state).to(grad_output.dtype).t()
        )
        grad_fn = (
            grad_output @ w_down_dequant.T
            + grad_output @ w_down_lora_b.T @ w_down_lora_a.T
        )
        
        # TODO: activation backward
        # activation part
        grad_up = grad_fn.clone()
        grad_up[up] = 0
        
        # up proj part
        # d L / d w_up_lora_a = x1.T @ d L / d y1 @ w_up_lora_b.T
        grad_w_up_lora_a = x_after_norm_2.mT @ (grad_up @ w_up_lora_b.T)
        # d L / d w_up_lora_b = y1_lora_a.T @ d L / d y1
        grad_w_up_lora_b = up_lora_a.mT @ grad_up
        # d L / d x1 = d L / d y1 @ w_up.T + d L / d y1 @ w_up_lora_b.T @ w_up_lora_a.T
        w_up_dequant = F.dequantize_4bit(w_up, w_up_quant_state).to(grad_output.dtype).t()
        grad_norm_2 = grad_up @ w_up_dequant.T + grad_up @ w_up_lora_b.T @ w_up_lora_a.T
        
        # layernorm & rmsnorm backward
        if ctx.norm_mode == "layernorm":
            grad_x_before_norm_2, grad_norm_weight_2, grad_norm_bias_2 = layernorm_backward(
                grad_norm_2, x_2_res, norm_weight_2, norm_bias_2, mean_2, rstd_2 # TODO: other params
            )
        else:
            grad_x_before_norm_2, grad_norm_weight_2, grad_norm_bias_2 = rmsnorm_backward(
                grad_norm_2, x_2_res, norm_weight_2, mean_2, rstd_2 # TODO: other params
            )
        
        # residual connection
        grad_x_before_norm_2 += grad_output
        
        # o part
        grad_w_o_lora_a = o.mT @ (grad_x_before_norm_2 @ w_o_lora_b.T)
        grad_w_o_lora_b = o_lora_a.mT @ grad_x_before_norm_2
        w_o_dequant = F.dequantize_nf4(w_o, w_o_quant_state).to(grad_x_before_norm_2.dtype).t()
        grad_o = grad_x_before_norm_2 @ w_o_dequant.T + grad_x_before_norm_2 @ w_o_lora_b.T @ w_o_lora_a.T
        
        # reshape
        grad_o = hidden_to_head_shape(grad_o, ctx.num_heads)
        
        # backward of second GEMM: O = A @ V
        # d L / d V = A.T @ d L / d O
        grad_v = a.transpose(-2, -1) @ grad_o
        grad_a = grad_o @ v.transpose(-2, -1)

        # backward of softmax
        grad_s = (grad_a - (grad_a * a).sum(dim=-1, keepdims=True)) * a

        # backward of first GEMM: S = Q @ K.T / sqrt(d_k)
        grad_s = grad_s / math.sqrt(ctx.head_dim)
        # d L / d K = (d L / d S)^T @ Q
        grad_k = grad_s.transpose(-2, -1) @ q
        # d L / d Q = d L / d S @ K
        grad_q = grad_s @ k
        
        # apply positional encoding
        grad_q = rope_backward(grad_q, cos, sin) # TODO: other params
        grad_k = rope_backward(grad_k, cos, sin)
        grad_v = head_to_hidden_shape(grad_v)
        
        # backward of q_proj
        grad_w_q_lora_a = x_after_norm_1.mT @ (grad_q @ w_q_lora_b.T)
        grad_w_q_lora_b = q_lora_a.mT @ grad_q
        w_q_dequant = F.dequantize_nf4(w_q, w_q_quant_state).to(grad_output.dtype).t()
        grad_norm_1 = grad_q @ w_q_dequant.T + grad_q @ w_q_lora_b.T @ w_q_lora_a.T

        # backward of k_proj
        grad_w_k_lora_a = x_after_norm_1.mT @ (grad_k @ w_k_lora_b.T)
        grad_w_k_lora_b = k_lora_a.mT @ grad_k
        w_k_dequant = F.dequantize_nf4(w_k, w_k_quant_state).to(grad_output.dtype).t()
        grad_norm_1 += grad_k @ w_k_dequant.T + grad_k @ w_k_lora_b.T @ w_k_lora_a.T

        # backward of v_proj
        grad_w_v_lora_a = x_after_norm_1.mT @ (grad_v @ w_v_lora_b.T)
        grad_w_v_lora_b = v_lora_a.mT @ grad_v
        w_v_dequant = F.dequantize_nf4(w_v, w_v_quant_state).to(grad_output.dtype).t()
        grad_norm_1 += grad_v @ w_v_dequant.T + grad_v @ w_v_lora_b.T @ w_v_lora_a.T
        
        # layernorm or rmsnorm backward
        if ctx.norm_mode == "layernorm":
            grad_x_before_norm_1, grad_norm_weight_1, grad_norm_bias_1 = layernorm_backward(
                grad_norm_1, x_1_res, norm_weight_1, norm_bias_1, mean_1, rstd_1 # TODO: other params
            )
        else:
            grad_x_before_norm_1, grad_norm_weight_1, grad_norm_bias_1 = rmsnorm_backward(
                grad_norm_1, x_1_res, norm_weight_1, mean_1, rstd_1 # TODO: other params
            )
            
        # residual connection
        grad_x_before_norm_1 += grad_x_before_norm_2
        
        return (
            grad_x_before_norm_1,
            #############attention part#############
            None,
            None,
            ####################################
            None,
            None,
            ####################################
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
            ####################################
            None,
            None,
            None,
            grad_w_up_lora_a,
            grad_w_up_lora_b,
            ####################################
            None,
            None,
            None,
            grad_w_down_lora_a,
            grad_w_down_lora_b
        ) + (None,) * 7


class MixedSparseSingleLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ):
        super(MixedSparseSingleLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.iteration = 0
        self.static_value = None
        
    def forward(
        self,
        input: torch.Tensor,
        norm_weight_1: torch.Tensor,
        norm_bias_1: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        ############################################
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
        ############################################
        norm_weight_2: torch.Tensor,
        norm_bias_2: torch.Tensor,
        up_proj_base: bnb.nn.modules.Linear4bit,
        up_proj_lora_a: torch.nn.Linear,
        up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit,
        down_proj_lora_a: torch.nn.Linear,
        down_proj_lora_b: torch.nn.Linear,
        ############################################
        attention_mask: torch.Tensor,
        norm_mode: str,
        num_heads: int,
        use_rotary_pos_enc: bool,
        small_value_approx: bool,
        activation_forward: str,
        activation_backward: str,
    ):
        y = MixedSparseSingleLayerNaiveFunc.apply(
            input,
            #############attention part#############
            norm_weight_1,
            norm_bias_1,
            ####################################
            cos,
            sin,
            ####################################
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
            #############mlp part#############
            norm_weight_2,
            norm_bias_2,
            ####################################
            up_proj_base.weight,
            up_proj_base.bias,
            up_proj_base.weight.quant_state,
            up_proj_lora_a.default.weight.T,
            up_proj_lora_b.default.weight.T,
            ####################################
            down_proj_base.weight,
            down_proj_base.bias,
            down_proj_base.weight.quant_state,
            down_proj_lora_a.default.weight.T,
            down_proj_lora_b.default.weight.T,
            ####################################
            attention_mask,
            norm_mode,
            num_heads,
            use_rotary_pos_enc,
            small_value_approx,
            activation_forward,
            activation_backward,
        )
        
        return y