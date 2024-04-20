import math
import torch
import bitsandbytes.functional as F
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor
from gact.memory_efficient_function import (
    per_block_quantization,
    per_block_quantization_4bit,
    per_block_dequantization,
    dct_compression,
    jpeg_compression,
    naive_adjustment,
)


class EfficientMemoryGEMMFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x1,
        x2,
        compress_type,
        jpeg_processor,
        dct_processor,
        quantization_shape,
        attn_first,
        prune_ratio,
        use_8bit,
        iteration,
        static_value1,
        static_value2,
    ):
        if use_8bit:
            s1 = x1.abs().max()
            s2 = x2.abs().max()
            x1 = torch.round(x1 / s1 * 127)
            x2 = torch.round(x2 / s2 * 127)
            result = x1 @ x2 / 127 / 127 * s1 * s2
            ctx.scale = (s1, s2)
        else:
            result = x1 @ x2
        ctx.needs_inputs_grad = [x1.requires_grad, x2.requires_grad]
        ctx.compress_type = compress_type
        ctx.quantization_shape = quantization_shape
        ctx.use_8bit = use_8bit
        
        kth_val_1 = torch.tensor(0.0, device=x1.device)
        kth_val_2 = torch.tensor(0.0, device=x2.device)
        if compress_type == "NF4":
            # quantize the cached activation
            x1, quant_state_1 = F.quantize_nf4(x1)
            x2, quant_state_2 = F.quantize_nf4(x2)
            ctx.quant_state = quant_state_1, quant_state_2
        elif compress_type == "PRUNE_ROW":
            if iteration < 10:
                kth_val_1 = torch.kthvalue(
                    x1.flatten(), int(x1.numel() * prune_ratio)
                ).values
                kth_val_2 = torch.kthvalue(
                    x2.flatten(), int(x2.numel() * prune_ratio)
                ).values
            else:
                kth_val_1, kth_val_2 = static_value1, static_value2
            mask_1 = x1.abs() > kth_val_1
            x1 = x1 * mask_1
            mask_2 = x2.abs() > kth_val_2
            x2 = x2 * mask_2
        elif compress_type != "NONE":
            # shape preparation for DCT
            input_shape = [x1.shape, x2.shape]
            ctx.input_shape = input_shape
            bit = 4 if compress_type == "INT4" else 8
            # quantize the cached activation
            x1, quant_state1 = per_block_quantization(
                x1.contiguous(), input_shape[0], quantization_shape, bit=bit
            )
            x2, quant_state2 = per_block_quantization(
                x2.contiguous(), input_shape[1], quantization_shape, bit=bit
            )
            ctx.quant_state = [quant_state1, quant_state2]

            if compress_type == "PRUNE":
                kth_val = torch.kthvalue(
                    x1.abs().flatten(), int(x1.numel() * 0.1)
                ).values
                x1 = torch.where(x1.abs() < kth_val, torch.zeros_like(x1), x1)
                x1 = naive_adjustment(x1, input_shape[0])

                kth_val = torch.kthvalue(
                    x2.abs().flatten(), int(x2.numel() * 0.1)
                ).values
                x2 = torch.where(x2.abs() < kth_val, torch.zeros_like(x2), x2)
                x2 = naive_adjustment(x2, input_shape[1])

            # compress the cached activation
            if compress_type == "JPEG":
                if attn_first:
                    # x1[x1 < -100] = -128
                    x1 = naive_adjustment(x1, input_shape[0])
                else:
                    x1 = jpeg_compression(x1, input_shape[0], jpeg_processor)
                x2 = jpeg_compression(x2, input_shape[1], jpeg_processor)

            elif compress_type == "DCT":
                if attn_first:
                    # x1[x1 < -100] = -128
                    x1 = naive_adjustment(x1, input_shape[0])
                else:
                    x1 = dct_compression(x1, input_shape[0], dct_processor)
                x2 = dct_compression(x2, input_shape[1], dct_processor)

            elif compress_type == "NAIVE" or compress_type == "INT4":
                # if attn_first:
                #   x1[x1 < -100] = -128
                x1 = naive_adjustment(x1, input_shape[0])
                x2 = naive_adjustment(x2, input_shape[1])
        ctx.mark_non_differentiable(kth_val_1, kth_val_2)
        ctx.save_for_backward(x1, x2)
        return result, kth_val_1, kth_val_2

    def backward(ctx, grad_output, grad_kth_val1, grad_kth_val2):
        x1, x2 = ctx.saved_tensors
        quantization_shape = ctx.quantization_shape
        grad_input1, grad_input2 = None, None

        if ctx.compress_type == "NF4":
            x1 = F.dequantize_nf4(x1, ctx.quant_state[0])
            x2 = F.dequantize_nf4(x2, ctx.quant_state[1])
        elif ctx.compress_type != "NONE" and ctx.compress_type != "PRUNE_ROW":
            quant_state1, quant_state2 = ctx.quant_state
            input_shape1, input_shape2 = ctx.input_shape
            x1 = per_block_dequantization(
                x1, input_shape1, quant_state1, quantization_shape
            )
            x2 = per_block_dequantization(
                x2, input_shape2, quant_state2, quantization_shape
            )

        if ctx.use_8bit:
            s1, s2 = ctx.scale
            sg = grad_output.abs().max()
            grad_output = torch.round(grad_output / sg * 127)
            x1 = torch.round(x1 / s1 * 127)
            x2 = torch.round(x2 / s2 * 127)
            grad_input1 = grad_output @ x2.transpose(-2, -1)
            grad_input2 = x1.transpose(-2, -1) @ grad_output
            grad_input1 = grad_input1 / 127 / 127 * s1 * sg
            grad_input2 = grad_input2 / 127 / 127 * sg * s2
        else:
            grad_input1 = grad_output @ x2.transpose(-2, -1)
            grad_input2 = x1.transpose(-2, -1) @ grad_output

        return (
            grad_input1,
            grad_input2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class EfficientMemoryGEMM(torch.nn.Module):
    def __init__(
        self,
        compress_type: str = "JPEG",
        compress_quality: int = 50,
        quantization_shape: int = 64,
        attn_first: bool = False,
        prune_ratio: float = 0.90,
        use_8bit: bool = False,
    ):
        super().__init__()
        self.compress_type = compress_type
        self.compress_quality = compress_quality
        self.jpeg_processor = JPEGProcessor(quality=compress_quality)
        self.dct_processor = DCTProcessor(quality=compress_quality)
        self.quantization_shape = quantization_shape
        self.attn_first = attn_first
        self.prune_ratio = prune_ratio
        self.use_8bit = use_8bit
        self.iteration = 0
        self.static_value1 = None
        self.static_value2 = None

    def forward(self, x1, x2):
        if self.extract_mode:
            torch.save(x1, f"output/{self.name}_1.pt")
            torch.save(x2, f"output/{self.name}_2.pt")

        result, static_value1, static_value2 = EfficientMemoryGEMMFunc.apply(
            x1,
            x2,
            self.compress_type,
            self.jpeg_processor,
            self.dct_processor,
            self.quantization_shape,
            self.attn_first,
            self.prune_ratio,
            self.use_8bit,
            self.iteration,
            self.static_value1,
            self.static_value2,
        )
        # ema
        self.static_value1 = (
            static_value1
            if self.static_value1 is None
            else (self.iteration * self.static_value1 + static_value1)
            / (self.iteration + 1)
        )
        self.static_value2 = (
            static_value2
            if self.static_value2 is None
            else (self.iteration * self.static_value2 + static_value2)
            / (self.iteration + 1)
        )
        self.iteration += 1

        return result
