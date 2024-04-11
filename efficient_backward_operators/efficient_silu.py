import math
import torch
import bitsandbytes.functional as BF
import torch.nn.functional as F
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor
from gact.memory_efficient_function import (
    per_block_quantization,
    per_block_dequantization,
    dct_compression,
    jpeg_compression,
    naive_adjustment,
)


class EfficientMemorySiLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        compress_type,
        jpeg_processor,
        dct_processor,
        quantization_shape,
        prune_ratio,
    ):
        result = F.silu(x)
        ctx.needs_inputs_grad = x.requires_grad
        ctx.compress_type = compress_type
        ctx.quantization_shape = quantization_shape

        if compress_type == "NF4":
            # quantize the cached activation
            x, quant_state = BF.quantize_nf4(x)
            ctx.quant_state = quant_state
        elif compress_type == "PRUNE_ROW":
            kth_val = torch.kthvalue(
                x.abs().flatten(), int(x.numel() * prune_ratio)
            ).values
            x = torch.where(x.abs() < kth_val, torch.zeros_like(x), x)
        elif compress_type != "NONE":
            input_shape = x.shape
            ctx.input_shape = input_shape
            bit = 4 if compress_type == "INT4" else 8

            x, quant_state = per_block_quantization(
                x, input_shape, quantization_shape, bit=bit
            )
            ctx.quant_state = quant_state

            if compress_type == "JPEG":
                x = jpeg_compression(x, input_shape, jpeg_processor, quantization_shape)

            elif compress_type == "DCT":
                x = dct_compression(x, input_shape, dct_processor, quantization_shape)

            elif compress_type == "NAIVE" or compress_type == "INT4":
                x = naive_adjustment(x, input_shape, quantization_shape)

        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        quantization_shape = ctx.quantization_shape
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
            sigmoid = F.sigmoid(x)
            grad_input = sigmoid * (1 + x - x * sigmoid) * grad_output

        return grad_input, None, None, None, None, None


class EfficientMemorySiLU(torch.nn.Module):
    def __init__(
        self,
        compress_type: str = "JPEG",
        compress_quality: int = 50,
        quantization_shape: int = 64,
        prune_ratio: float = 0.75,
    ):
        super(EfficientMemorySiLU, self).__init__()
        self.compress_type = compress_type
        self.compress_quality = compress_quality
        self.jpeg_processor = JPEGProcessor(quality=compress_quality)
        self.dct_processor = DCTProcessor(
            quality=compress_quality, interpolation=quantization_shape / 64
        )
        self.quantization_shape = quantization_shape
        self.prune_ratio = prune_ratio

    def forward(self, input):
        if self.extract_mode:
            torch.save(input, f"output/{self.name}.pt")

        return EfficientMemorySiLUFunc.apply(
            input,
            self.compress_type,
            self.jpeg_processor,
            self.dct_processor,
            self.quantization_shape,
            self.prune_ratio,
        )
