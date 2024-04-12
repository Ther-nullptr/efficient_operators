import math
import torch
import torch.nn.functional as F
import bitsandbytes.functional as BF
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor
from gact.memory_efficient_function import (
    per_block_quantization,
    per_block_dequantization,
    dct_compression,
    jpeg_compression,
    naive_adjustment,
)


class EfficientMemoryGELUFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        compress_type,
        jpeg_processor,
        dct_processor,
        quantization_shape,
        prune_ratio,
        iteration,
        static_value,
    ):
        result = F.gelu(x)
        ctx.needs_inputs_grad = x.requires_grad
        ctx.compress_type = compress_type
        ctx.quantization_shape = quantization_shape

        if compress_type == "NF4":
            # quantize the cached activation
            x, quant_state = BF.quantize_nf4(x)
            ctx.quant_state = quant_state
        elif compress_type == "PRUNE_ROW":
            if iteration < 10:
                kth_val = torch.kthvalue(
                    x.flatten(), int(x.numel() * prune_ratio)
                ).values
            else:
                kth_val = static_value
            mask = x > kth_val # when mask is 0, set them to -10, otherwise keep the original value
            x = (~mask) * -10 + mask * x
            # x = torch.where(, torch.zeros_like(x) - 10, x)
        elif compress_type != "NONE":
            input_shape = x.shape
            ctx.input_shape = input_shape

            x, quant_state = per_block_quantization(x, input_shape, quantization_shape)
            ctx.quant_state = quant_state

            if compress_type == "PRUNE":
                kth_val = torch.kthvalue(x.abs().flatten(), int(x.numel() * 0.1)).values
                x = torch.where(x.abs() < kth_val, torch.zeros_like(x), x)

            if compress_type == "JPEG":
                x = jpeg_compression(x, input_shape, jpeg_processor, quantization_shape)

            elif compress_type == "DCT":
                x = dct_compression(x, input_shape, dct_processor, quantization_shape)

            elif compress_type == "NAIVE":
                x = naive_adjustment(x, input_shape, quantization_shape)
        ctx.mark_non_differentiable(kth_val)
        ctx.save_for_backward(x)
        return result, kth_val

    @staticmethod
    def backward(ctx, grad_output, grad_kth_val):
        (x,) = ctx.saved_tensors
        quantization_shape = ctx.quantization_shape

        gamma = math.sqrt(2 / math.pi)
        kappa = 0.044715
        grad_input = None

        if ctx.needs_inputs_grad:
            if ctx.compress_type == "NF4":
                x = BF.dequantize_nf4(x, ctx.quant_state)
            elif ctx.compress_type != "NONE" and ctx.compress_type != "PRUNE_ROW":
                quant_state = ctx.quant_state
                input_shape = ctx.input_shape
                x = per_block_dequantization(
                    x, input_shape, quant_state, quantization_shape
                )
            y = gamma * (x + kappa * x**3)
            tanh_y = F.tanh(y)
            grad_input = (
                0.5
                * (
                    (1 + tanh_y)
                    + x * ((1 - tanh_y**2) * gamma * (1 + 3 * kappa * x**2))
                )
                * grad_output
            )

        return grad_input, None, None, None, None, None, None, None


class EfficientMemoryGELU(torch.nn.Module):
    def __init__(
        self,
        compress_type: str = "JPEG",
        compress_quality: int = 50,
        quantization_shape: int = 64,
        prune_ratio: float = 0.9,
    ):
        super(EfficientMemoryGELU, self).__init__()
        self.compress_type = compress_type
        self.compress_quality = compress_quality
        self.jpeg_processor = JPEGProcessor(quality=compress_quality)
        self.dct_processor = DCTProcessor(
            quality=compress_quality, interpolation=quantization_shape / 64
        )
        self.quantization_shape = quantization_shape
        self.prune_ratio = prune_ratio
        self.iteration = 0
        self.static_value = None

    def forward(self, input):
        result, static_value = EfficientMemoryGELUFunc.apply(
            input,
            self.compress_type,
            self.jpeg_processor,
            self.dct_processor,
            self.quantization_shape,
            self.prune_ratio,
            self.iteration,
            self.static_value,
        )
        # ema
        self.static_value = (
            static_value
            if self.static_value is None
            else (self.iteration * self.static_value + static_value)
            / (self.iteration + 1)
        )
        self.iteration += 1

        return result
