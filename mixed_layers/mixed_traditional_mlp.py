import math
import torch
import typing
import bitsandbytes as bnb
import bitsandbytes.functional as F


class MixedTraditionalMLPFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x1: torch.Tensor,
        w_up: torch.Tensor,
        b_up: torch.Tensor,
        w_up_state: typing.Tuple,
        w_up_lora_a: torch.Tensor,
        w_up_lora_b: torch.Tensor,
        w_down: torch.Tensor,
        b_down: torch.Tensor,
        w_down_state: typing.Tuple,
        w_down_lora_a: torch.Tensor,
        w_down_lora_b: torch.Tensor,
        activation_forward: str,
        activation_backward: str,
    ):
        # forward process: up_proj
        w_up_dequant = F.dequantize_4bit(w_up, w_up_state).to(x1.dtype).t()
        y1_main = x1 @ w_up_dequant + b_up if b_up is not None else x1 @ w_up_dequant
        y1_lora_a = x1 @ w_up_lora_a
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

        # save: x1, y1_lora_a, y1(for soft activations), mask(for hard activations), x2, y2_lora_a
        if activation_backward == "relu":
            mask = y1 < 0
            if activation_forward != "relu":
                x2 = torch.relu(x2)  # cache the sparse version of x2
            ctx.save_for_backward(
                x1,
                y1_lora_a,
                mask,
                x2,
                y2_lora_a,
                w_up,
                b_up,
                w_up_lora_a,
                w_up_lora_b,
                w_down,
                b_down,
                w_down_lora_a,
                w_down_lora_b,
            )
        else:
            ctx.save_for_backward(
                x1,
                y1_lora_a,
                y1,
                x2,
                y2_lora_a,
                w_up,
                b_up,
                w_up_lora_a,
                w_up_lora_b,
                w_down,
                b_down,
                w_down_lora_a,
                w_down_lora_b,
            )

        ctx.quant_state = w_up_state, w_down_state
        ctx.activation_backward = activation_backward

        return y2

    @staticmethod
    def backward(ctx, grad_output):
        w_up_state, w_down_state = ctx.quant_state
        if ctx.activation_backward == "relu":
            (
                x1,
                y1_lora_a,
                mask,
                x2,
                y2_lora_a,
                w_up,
                b_up,
                w_up_lora_a,
                w_up_lora_b,
                w_down,
                b_down,
                w_down_lora_a,
                w_down_lora_b,
            ) = ctx.saved_tensors
        else:
            (
                x1,
                y1_lora_a,
                y1,
                x2,
                y2_lora_a,
                w_up,
                b_up,
                w_up_lora_a,
                w_up_lora_b,
                w_down,
                b_down,
                w_down_lora_a,
                w_down_lora_b,
            ) = ctx.saved_tensors

        # down proj part
        # d L / d w_down_lora_a = x2.T @ d L / d y2 @ w_down_lora_b.T
        # TODO: x2 maybe sparse
        grad_w_down_lora_a = x2.mT @ (grad_output @ w_down_lora_b.T)
        # d L / d w_down_lora_b = y2_lora_a.T @ d L / d y2
        grad_w_down_lora_b = y2_lora_a.mT @ grad_output
        # d L / d x2 = d L / d y2 @ w_down.T + d L / d y2 @ w_down_lora_b.T @ w_down_lora_a.T
        w_down_dequant = (
            F.dequantize_4bit(w_down, w_down_state).to(grad_output.dtype).t()
        )
        grad_x2 = (
            grad_output @ w_down_dequant.T
            + grad_output @ w_down_lora_b.T @ w_down_lora_a.T
        )

        # activation part
        if ctx.activation_backward == "relu":
            grad_y1 = grad_x2.clone()
            grad_y1[mask] = 0
        elif ctx.activation_backward == "silu":
            sigmoid = torch.sigmoid(y1)
            grad_y1 = grad_x2 * sigmoid * (1 + y1 - y1 * sigmoid)
        elif ctx.activation_backward == "gelu":
            gamma = math.sqrt(2 / math.pi)
            kappa = 0.044715
            y1_extra = gamma * (y1 + kappa * y1**3)
            tanh_y = torch.tanh(y1_extra)
            grad_y1 = (
                grad_x2
                * 0.5
                * (
                    (1 + tanh_y)
                    + y1 * ((1 - tanh_y**2) * gamma * (1 + 3 * kappa * y1**2))
                )
            )

        # up proj part
        # d L / d w_up_lora_a = x1.T @ d L / d y1 @ w_up_lora_b.T
        grad_w_up_lora_a = x1.mT @ (grad_y1 @ w_up_lora_b.T)
        # d L / d w_up_lora_b = y1_lora_a.T @ d L / d y1
        grad_w_up_lora_b = y1_lora_a.mT @ grad_y1
        # d L / d x1 = d L / d y1 @ w_up.T + d L / d y1 @ w_up_lora_b.T @ w_up_lora_a.T
        w_up_dequant = F.dequantize_4bit(w_up, w_up_state).to(grad_output.dtype).t()
        grad_x1 = grad_y1 @ w_up_dequant.T + grad_y1 @ w_up_lora_b.T @ w_up_lora_a.T

        # TODO: add bias support
        return (
            grad_x1,
            None,
            None,
            None,
            grad_w_up_lora_a,
            grad_w_up_lora_b,
            None,
            None,
            None,
            grad_w_down_lora_a,
            grad_w_down_lora_b,
            None,
            None,
        )


class MixedTraditionalMLP(torch.nn.Module):
    def __init__(
        self, activation_forward="relu", activation_backward="relu", bias=False
    ):
        super(MixedTraditionalMLP, self).__init__()

        # activation function method. Now support: ReLU, SiLU, GELU
        self.activation_forward = activation_forward
        self.activation_backward = activation_backward

    def forward(
        self,
        input: torch.Tensor,
        up_proj_base: bnb.nn.modules.Linear4bit,
        up_proj_lora_a: torch.nn.Linear,
        up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit,
        down_proj_lora_a: torch.nn.Linear,
        down_proj_lora_b: torch.nn.Linear,
    ):
        #! Notice we use equation y = xW + b; instead of default y = xW^T + b
        return MixedTraditionalMLPFunc.apply(
            input,
            up_proj_base.weight,
            up_proj_base.bias,
            up_proj_base.weight.quant_state,
            up_proj_lora_a.default.weight.T,
            up_proj_lora_b.default.weight.T,
            ############################
            down_proj_base.weight,
            down_proj_base.bias,
            down_proj_base.weight.quant_state,
            down_proj_lora_a.default.weight.T,
            down_proj_lora_b.default.weight.T,
            ############################
            self.activation_forward,
            self.activation_backward,
        )
