import math
import torch
import typing
import bitsandbytes as bnb
import torch.nn.functional as F
import bitsandbytes.functional as BF

from .layernorm_kernels import layernorm_forward, layernorm_backward
from .rmsnorm_kernels import rmsnorm_forward, rmsnorm_backward
from .rope_kernels import rope_forward, rope_backward
from .dropout_kernels import dropout_forward, dropout_backward

def d_gelu(x):
    val = 1.702 * x
    exp_val = torch.exp(val)
    return exp_val * (exp_val + val + 1) / (exp_val + 1) ** 2

def d_silu(x):
    exp_val = torch.exp(x)
    return exp_val * (x + exp_val + 1) / (exp_val + 1) ** 2

def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)

def head_to_hidden_shape(x: torch.Tensor):
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)

def lora_forward(w, w_quant_state, w_lora_a, w_lora_b, b, x, seed=0):
    w_dequant = BF.dequantize_nf4(w, w_quant_state).t()
    x_main = x.to(w_dequant.dtype) @ w_dequant + b if b is not None else x.to(w_dequant.dtype) @ w_dequant
    x_lora_a = x.to(w_lora_a.dtype) @ w_lora_a
    x_lora = x_lora_a @ w_lora_b
    x = x_main + x_lora
    return x, x_main, x_lora_a

def lora_backward(w, w_quant_state, w_lora_a, w_lora_b, x, x_lora_a, grad_y, seed=0):
    grad_w_lora_a = x.to(w_lora_b.dtype).mT @ (grad_y.to(w_lora_b.dtype) @ w_lora_b.T)
    grad_w_lora_b = x_lora_a.mT @ grad_y.to(w_lora_b.dtype)
    w_dequant = BF.dequantize_nf4(w, w_quant_state).t()
    grad_x = grad_y.to(w_dequant.dtype) @ w_dequant.T 
    grad_x += (grad_y.to(w_lora_b.dtype) @ w_lora_b.T @ w_lora_a.T)
    return grad_w_lora_a, grad_w_lora_b, grad_x

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
        w_down_quant_state: typing.Tuple,
        w_down_lora_a: torch.Tensor,
        w_down_lora_b: torch.Tensor,
        ###############other################
        attention_mask: torch.Tensor,
        norm_mode: str,
        num_heads: int,
        head_dim: int,
        use_rotary_pos_enc: bool,
        small_value_approx: bool,
        activation_forward: str,
        activation_backward: str,
        ###############about sparse################
        iteration: int,
        static_value: float,
        prune: bool,
        prune_ratio: float,
        #########################################
        training: bool,
    ):
        x_1_res = x # keep the original input for residual connection
        
        if b_q is not None:
            b_q, b_k, b_v, b_o, b_up, b_down = b_q.to(torch.float32), b_k.to(torch.float32), b_v.to(torch.float32), b_o.to(torch.float32), b_up.to(torch.float32), b_down.to(torch.float32)
        
        # layernorm or rmsnorm
        if norm_mode == "layernorm":
            x_after_norm_1, mean_1, rstd_1, _, _ = layernorm_forward(x, norm_weight_1, norm_bias_1, eps = 1e-10)
        else:
            x_after_norm_1, mean_1, rstd_1, _, _ = rmsnorm_forward(x, norm_weight_1, eps = 1e-10)

        # compute q,k,v
        # forward process: q_proj
        q, q_main, q_lora_a = lora_forward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, b_q, x_after_norm_1, iteration)
        
        # forward process: k_proj
        k, k_main, k_lora_a = lora_forward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, b_k, x_after_norm_1, iteration)

        # forward process: v_proj
        v, v_main, v_lora_a = lora_forward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, b_v, x_after_norm_1, iteration)
        
        # reshape
        q = hidden_to_head_shape(q, num_heads)
        k = hidden_to_head_shape(k, num_heads)
        v = hidden_to_head_shape(v, num_heads)
        
        ctx.q_shape = q.shape
        
        # TODO: apply positional encoding
        if use_rotary_pos_enc:
            q = rope_forward(q, cos, sin)
            k = rope_forward(k, cos, sin)
        
        q, k, v = q.to(torch.float32), k.to(torch.float32), v.to(torch.float32)

        # q,k,v: [bsz, num_heads, q_len, head_dim]
        # notice forward process no need to drop heads
        bsz, num_heads, q_len, head_dim = q.shape

        # forward: S = Q @ K.T / sqrt(d_k)
        s = q @ k.transpose(-2, -1) / math.sqrt(head_dim)
        # apply mask
        if attention_mask is not None:
            s = s + attention_mask

        # forward: softmax
        a = torch.softmax(s, dim=-1).to(torch.float32)  # [bsz, num_heads, q_len, q_len]

        # forward: O = A @ V
        o = a @ v
        
        # reshape
        o = head_to_hidden_shape(o)

        # forward process: o_proj
        o_final, o_main, o_lora_a = lora_forward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, b_o, o, iteration)
        
        if small_value_approx:
            # save the main weight, then recompute when backward
            q = q_main
            k = k_main
            v = v_main
            
        # residual connection
        x_1_final = x_1_res + o_final
        x_2_res = x_1_final
        
        # prune all the values (attention part)
        if prune and training:
            if iteration < 10:
                kth_val_x_1_res = torch.kthvalue(x_1_res.abs().flatten().to(torch.float32), int(x_1_res.numel() * prune_ratio)).values
                kth_val_x_after_norm_1 = torch.kthvalue(x_after_norm_1.abs().flatten().to(torch.float32), int(x_after_norm_1.numel() * prune_ratio)).values
                kth_val_q = torch.kthvalue(q.abs().flatten().to(torch.float32), int(q.numel() * prune_ratio)).values
                kth_val_k = torch.kthvalue(k.abs().flatten().to(torch.float32), int(k.numel() * prune_ratio)).values
                kth_val_v = torch.kthvalue(v.abs().flatten().to(torch.float32), int(v.numel() * prune_ratio)).values
                #! notice a is all positive, so compute use a instead of a.abs()
                kth_val_a = torch.kthvalue(a.flatten().to(torch.float32), int(a.numel() * prune_ratio)).values
                kth_val_o = torch.kthvalue(o.abs().flatten().to(torch.float32), int(o.numel() * prune_ratio)).values
            else:
                kth_val_x_1_res, kth_val_x_after_norm_1, kth_val_q, kth_val_k, kth_val_v, kth_val_a, kth_val_o = static_value[:7] # static_value is a tuple
            x_1_res = (x_1_res.abs() > kth_val_x_1_res) * x_1_res
            x_after_norm_1 = (x_after_norm_1.abs() > kth_val_x_after_norm_1) * x_after_norm_1
            q = (q.abs() > kth_val_q) * q
            k = (k.abs() > kth_val_k) * k
            v = (v.abs() > kth_val_v) * v
            a = (a > kth_val_a) * a
            o = (o.abs() > kth_val_o) * o
        
        # layernorm or rmsnorm
        if norm_mode == "layernorm":
            x_after_norm_2, mean_2, rstd_2, block_size, num_warps = layernorm_forward(x_1_final, norm_weight_2, norm_bias_2, eps = 1e-10)
        else:
            x_after_norm_2, mean_2, rstd_2, block_size, num_warps = rmsnorm_forward(x_1_final, norm_weight_2, eps = 1e-10)
        
        # forward process: up_proj
        up, up_main, up_lora_a = lora_forward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, b_up, x_after_norm_2, iteration)
        
        # apply activation function
        if activation_forward == "relu":
            fn = torch.relu(up).to(torch.float32)
            # TODO: now only support relu, generate mask
            up = up > 0 # now up is a mask
        elif activation_forward == "silu":
            fn = torch.silu(up).to(torch.float32)
        elif activation_forward == "gelu":
            fn = torch.gelu(up).to(torch.float32)
            up = up > 0

        # forward process: down_proj
        down, down_main, down_lora_a = lora_forward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, b_down, fn, iteration)
        
        # residual connection
        x_2_final = x_2_res + down
        
        if small_value_approx:
            x_2_res = x_2_res - o_lora_a @ w_o_lora_b
            fn = torch.relu(up_main)
            
        if prune and training:
            if iteration < 10:
                kth_val_x_2_res = torch.kthvalue(x_2_res.abs().flatten().to(torch.float32), int(x_2_res.numel() * prune_ratio)).values
                kth_val_x_after_norm_2 = torch.kthvalue(x_after_norm_2.abs().flatten().to(torch.float32), int(x_after_norm_2.numel() * prune_ratio)).values
                kth_val_fn = torch.kthvalue(fn.abs().flatten().to(torch.float32), int(fn.numel() * prune_ratio)).values
            else:
                kth_val_x_2_res, kth_val_x_after_norm_2, kth_val_fn = static_value[7:]
            x_2_res = (x_2_res.abs() > kth_val_x_2_res) * x_2_res
            x_after_norm_2 = (x_after_norm_2.abs() > kth_val_x_after_norm_2) * x_after_norm_2
            fn = (fn > kth_val_fn) * fn
        else:
            kth_val_x_1_res, kth_val_x_after_norm_1, kth_val_q, kth_val_k, kth_val_v, kth_val_a, kth_val_o, kth_val_x_2_res, kth_val_x_after_norm_2, kth_val_fn = [torch.tensor(0.0)] * 10
        ctx.mark_non_differentiable(kth_val_x_1_res, kth_val_x_after_norm_1, kth_val_q, kth_val_k, kth_val_v, kth_val_a, kth_val_o, kth_val_x_2_res, kth_val_x_after_norm_2, kth_val_fn)
        new_static_value = (kth_val_x_1_res, kth_val_x_after_norm_1, kth_val_q, kth_val_k, kth_val_v, kth_val_a, kth_val_o, kth_val_x_2_res, kth_val_x_after_norm_2, kth_val_fn)
        
        ctx.save_for_backward(
            ### activations (attention) ###
            x_1_res.to(torch.float32),
            x_after_norm_1.to(torch.float32),
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
            x_2_res.to(torch.float32),
            x_after_norm_2.to(torch.float32),
            mean_2,
            rstd_2,
            up.to(torch.float32), #! this part can be a raw mask
            up_lora_a,
            fn.to(torch.float32),
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
            w_down_quant_state,
        )
        ctx.norm_mode = norm_mode
        ctx.num_heads = num_heads
        ctx.use_rotary_pos_enc = use_rotary_pos_enc
        ctx.block_size = block_size
        ctx.num_warps = num_warps
        ctx.head_dim = head_dim
        ctx.small_value_approx = small_value_approx
        ctx.iteration = iteration
        ctx.training = training
        ctx.activation_backward = activation_backward
        
        x_2_final = x_2_final.to(torch.float32) #! for lm head
        return x_2_final, new_static_value
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_kth_val):
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
        
        grad_output = grad_output.to(torch.float32)
        
        if ctx.small_value_approx:
            fn = (fn + (up_lora_a @ w_up_lora_b) * up).to(torch.float32)
            x_2_res = (x_2_res + o_lora_a @ w_o_lora_b).to(torch.float32)
        
        # down proj part
        grad_w_down_lora_a, grad_w_down_lora_b, grad_fn = lora_backward(w_down, w_down_quant_state, w_down_lora_a, w_down_lora_b, fn, down_lora_a, grad_output, ctx.iteration)
        
        # TODO: activation backward
        # activation part
        if ctx.activation_backward == "relu":
            grad_up = grad_fn.clone()
            grad_up *= up
        elif ctx.activation_backward == "silu":
            grad_up = grad_fn * d_silu(up)
        elif ctx.activation_backward == "gelu":
            grad_up = grad_fn * d_relu(up)
        
        # up proj part
        grad_w_up_lora_a, grad_w_up_lora_b, grad_norm_2 = lora_backward(w_up, w_up_quant_state, w_up_lora_a, w_up_lora_b, x_after_norm_2, up_lora_a, grad_up, ctx.iteration)
        
        # layernorm & rmsnorm backward
        if ctx.norm_mode == "layernorm":
            grad_x_before_norm_2, grad_norm_weight_2, grad_norm_bias_2 = layernorm_backward(
                grad_norm_2, x_2_res, norm_weight_2, norm_bias_2, mean_2, rstd_2,
                True, 1e-10, ctx.num_warps, ctx.block_size
            )
        else:
            grad_x_before_norm_2, grad_norm_weight_2, grad_norm_bias_2 = rmsnorm_backward(
                grad_norm_2, x_2_res, norm_weight_2, mean_2, rstd_2, # TODO: other params
                True, 1e-10, ctx.num_warps, ctx.block_size
            )
        
        # residual connection
        grad_x_before_norm_2 += grad_output
        
        # o part
        grad_w_o_lora_a, grad_w_o_lora_b, grad_o = lora_backward(w_o, w_o_quant_state, w_o_lora_a, w_o_lora_b, o, o_lora_a, grad_x_before_norm_2, ctx.iteration)
        
        # reshape
        grad_o = hidden_to_head_shape(grad_o, ctx.num_heads)
        grad_o = grad_o.to(torch.float32)
        
        if ctx.small_value_approx:
            q = (q + q_lora_a @ w_q_lora_b).to(torch.float32)
            k = (k + k_lora_a @ w_k_lora_b).to(torch.float32)
            v = (v + v_lora_a @ w_v_lora_b).to(torch.float32)
            q = hidden_to_head_shape(q, ctx.num_heads)
            k = hidden_to_head_shape(k, ctx.num_heads)
            v = hidden_to_head_shape(v, ctx.num_heads)

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
        if ctx.use_rotary_pos_enc:
            grad_q = rope_backward(grad_q, cos, sin) # TODO: other params
            grad_k = rope_backward(grad_k, cos, sin)

        grad_q = head_to_hidden_shape(grad_q)
        grad_k = head_to_hidden_shape(grad_k)
        grad_v = head_to_hidden_shape(grad_v)
        
        # backward of q_proj
        grad_w_q_lora_a, grad_w_q_lora_b, grad_norm_1 = lora_backward(w_q, w_q_quant_state, w_q_lora_a, w_q_lora_b, x_after_norm_1, q_lora_a, grad_q, ctx.iteration)

        # backward of k_proj
        grad_w_k_lora_a, grad_w_k_lora_b, grad_norm_1_temp = lora_backward(w_k, w_k_quant_state, w_k_lora_a, w_k_lora_b, x_after_norm_1, k_lora_a, grad_k, ctx.iteration)
        grad_norm_1 += grad_norm_1_temp

        # backward of v_proj
        grad_w_v_lora_a, grad_w_v_lora_b, grad_norm_1_temp = lora_backward(w_v, w_v_quant_state, w_v_lora_a, w_v_lora_b, x_after_norm_1, v_lora_a, grad_v, ctx.iteration)
        grad_norm_1 += grad_norm_1_temp
        
        # layernorm or rmsnorm backward
        if ctx.norm_mode == "layernorm":
            grad_x_before_norm_1, grad_norm_weight_1, grad_norm_bias_1 = layernorm_backward(
                grad_norm_1, x_1_res, norm_weight_1, norm_bias_1, mean_1, rstd_1, # TODO: other params
                True, 1e-10, ctx.num_warps, ctx.block_size
            )
        else:
            grad_x_before_norm_1, grad_norm_weight_1, grad_norm_bias_1 = rmsnorm_backward(
                grad_norm_1, x_1_res, norm_weight_1, mean_1, rstd_1, # TODO: other params
                True, 1e-10, ctx.num_warps, ctx.block_size
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
        ) + (None,) * 13


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
        head_dim: int,
        use_rotary_pos_enc: bool,
        small_value_approx: bool,
        activation_forward: str,
        activation_backward: str,
        ############################################
        prune: bool,
        prune_ratio: float,
        ############################################
        training: bool,
    ):
        y, static_value = MixedSparseSingleLayerNaiveFunc.apply(
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
            head_dim,
            use_rotary_pos_enc,
            small_value_approx,
            activation_forward,
            activation_backward,
            ############################################
            self.iteration,
            self.static_value,
            prune,
            prune_ratio,
            training
        )
        
        if self.static_value is None:
            self.static_value = static_value
        else:
            self.static_value = list(self.static_value)
            for i in range(len(self.static_value)):
                self.static_value[i] = (self.iteration * self.static_value[i] + static_value[i]) / (self.iteration + 1)
            self.static_value = tuple(self.static_value)
        self.iteration += 1
        
        return y