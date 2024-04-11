import math
import torch
import typing
import bitsandbytes as bnb
import bitsandbytes.functional as F


class MixedGatedMLPFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x1: torch.Tensor,
        w_gate: torch.Tensor,
        b_gate: torch.Tensor,
        w_gate_state: typing.Tuple,
        w_gate_lora_a: torch.Tensor,
        w_gate_lora_b: torch.Tensor,
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
        # forward process: gate_proj
        w_gate_dequant = F.dequantize_4bit(w_gate, w_gate_state).to(x1.dtype).t()
        y1_main = (
            x1 @ w_gate_dequant + b_gate if b_gate is not None else x1 @ w_gate_dequant
        )
        y1_lora_a = x1 @ w_gate_lora_a
        y1_lora = y1_lora_a @ w_gate_lora_b
        y1 = y1_main + y1_lora

        # forward process: up_proj
        w_up_dequant = F.dequantize_4bit(w_up, w_up_state).to(x1.dtype).t()
        y2_main = x1 @ w_up_dequant + b_up if b_up is not None else x1 @ w_up_dequant
        y2_lora_a = x1 @ w_up_lora_a
        y2_lora = y2_lora_a @ w_up_lora_b
        y2 = y2_main + y2_lora

        # apply activation function
        if activation_forward == "relu":
            xL = torch.relu(y1)
        elif activation_forward == "silu":
            xL = torch.silu(y1)
        elif activation_forward == "gelu":
            xL = torch.gelu(y1)

        # hadamard product
        xR = y2
        x3 = xL * xR

        # forward process: down_proj
        w_down_dequant = F.dequantize_4bit(w_down, w_down_state).to(x3.dtype).t()
        y3_main = (
            x3 @ w_down_dequant + b_down if b_down is not None else x3 @ w_down_dequant
        )
        y3_lora_a = x3 @ w_down_lora_a
        y3_lora = y3_lora_a @ w_down_lora_b
        y3 = y3_main + y3_lora

        # save: x1, y1_lora_a, y1(for soft activations), mask(for hard activations), x2, y2_lora_a
        if activation_backward == "relu":
            mask = y1 < 0
            if activation_forward != "relu":
                xL = torch.relu(xL)
            ctx.save_for_backward(
                x1,
                y1_lora_a,
                mask,
                xL,
                xR,
                y2_lora_a,
                x3,
                y3_lora_a,
                w_gate,
                b_gate,
                w_gate_lora_a,
                w_gate_lora_b,
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
                xL,
                xR,
                y2_lora_a,
                x3,
                y3_lora_a,
                w_gate,
                b_gate,
                w_gate_lora_a,
                w_gate_lora_b,
                w_up,
                b_up,
                w_up_lora_a,
                w_up_lora_b,
                w_down,
                b_down,
                w_down_lora_a,
                w_down_lora_b,
            )

        ctx.activation_backward = activation_backward
        ctx.quant_state = w_gate_state, w_up_state, w_down_state

        return y3

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        w_gate_state, w_up_state, w_down_state = ctx.quant_state

        if ctx.activation_backward == "relu":
            (
                x1,
                y1_lora_a,
                mask,
                xL,
                xR,
                y2_lora_a,
                x3,
                y3_lora_a,
                w_gate,
                b_gate,
                w_gate_lora_a,
                w_gate_lora_b,
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
                xL,
                xR,
                y2_lora_a,
                x3,
                y3_lora_a,
                w_gate,
                b_gate,
                w_gate_lora_a,
                w_gate_lora_b,
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
        grad_w_down_lora_a = x3.mT @ (grad_output @ w_down_lora_b.T)
        # d L / d w_down_lora_b = y2_lora_a.T @ d L / d y2
        grad_w_down_lora_b = y3_lora_a.mT @ grad_output
        # d L / d x2 = d L / d y2 @ w_down.T + d L / d y2 @ w_down_lora_b.T @ w_down_lora_a.T
        w_down_dequant = (
            F.dequantize_4bit(w_down, w_down_state).to(grad_output.dtype).t()
        )
        grad_x2 = (
            grad_output @ w_down_dequant.T
            + grad_output @ w_down_lora_b.T @ w_down_lora_a.T
        )

        # hadamard product
        # d L / d xL = d L / d x3 * xR
        grad_xL = grad_x2 * xR
        # d L / d xR = d L / d x3 * xL
        grad_xR = grad_x2 * xL

        # activation part
        if ctx.activation_backward == "relu":
            grad_y1 = grad_xL.clone()
            grad_y1[mask] = 0
        elif ctx.activation_backward == "silu":
            sigmoid = torch.sigmoid(y1)
            grad_y1 = sigmoid * (1 + y1 - y1 * sigmoid) * grad_xL
        elif ctx.activation_backward == "gelu":
            gamma = math.sqrt(2 / math.pi)
            kappa = 0.044715
            grad_y1 = gamma * (y1 + kappa * y1**3)
            tanh_y = torch.tanh(y1)
            grad_y1 = (
                0.5
                * (
                    (1 + tanh_y)
                    + y1 * ((1 - tanh_y**2) * gamma * (1 + 3 * kappa * y1**2))
                )
                * grad_xL
            )

        # up proj part
        # d L / d w_up_lora_a = x1.T @ d L / d xL @ w_up_lora_b.T
        grad_w_up_lora_a = xR.mT @ (grad_xL @ w_up_lora_b.T)
        # d L / d w_up_lora_b = y1_lora_a.T @ d L / d xL
        grad_w_up_lora_b = y2_lora_a.mT @ grad_xR
        # d L / d x1 = d L / d xR @ w_up.T + d L / d xR @ w_up_lora_b.T @ w_up_lora_a.T
        w_up_dequant = F.dequantize_4bit(w_up, w_up_state).to(grad_output.dtype).t()
        grad_x1 = grad_xR @ w_up_dequant.T + grad_xR @ w_up_lora_b.T @ w_up_lora_a.T

        # gate proj part
        # d L / d w_gate_lora_a = x1.T @ d L / d xL @ w_gate_lora_b.T
        grad_w_gate_lora_a = x1.mT @ (grad_y1 @ w_gate_lora_b.T)
        # d L / d w_gate_lora_b = y1_lora_a.T @ d L / d xL
        grad_w_gate_lora_b = y1_lora_a.mT @ grad_y1
        # d L / d x1 = d L / d xL @ w_gate.T + d L / d xL @ w_gate_lora_b.T @ w_gate_lora_a.T
        w_gate_dequant = (
            F.dequantize_4bit(w_gate, w_gate_state).to(grad_output.dtype).t()
        )
        grad_x1 += (
            grad_y1 @ w_gate_dequant.T + grad_y1 @ w_gate_lora_b.T @ w_gate_lora_a.T
        )

        return (
            grad_x1,
            None,
            None,
            None,
            grad_w_gate_lora_a,
            grad_w_gate_lora_b,
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


class MixedGatedMLP(torch.nn.Module):
    def __init__(
        self, activation_forward="relu", activation_backward="relu", bias=False
    ):
        super(MixedGatedMLP, self).__init__()

        self.activation_forward = activation_forward
        self.activation_backward = activation_backward

    def forward(
        self,
        input: torch.Tensor,
        gate_proj_base: bnb.nn.modules.Linear4bit,
        gate_proj_lora_a: torch.nn.Linear,
        gate_proj_lora_b: torch.nn.Linear,
        up_proj_base: bnb.nn.modules.Linear4bit,
        up_proj_lora_a: torch.nn.Linear,
        up_proj_lora_b: torch.nn.Linear,
        down_proj_base: bnb.nn.modules.Linear4bit,
        down_proj_lora_a: torch.nn.Linear,
        down_proj_lora_b: torch.nn.Linear,
    ):
        return MixedGatedMLPFunc.apply(
            input,
            gate_proj_base.weight,
            gate_proj_base.bias,
            gate_proj_base.weight.quant_state,
            gate_proj_lora_a.default.weight.T,
            gate_proj_lora_b.default.weight.T,
            ############################
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
