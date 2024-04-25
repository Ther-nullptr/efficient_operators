import torch
import bitsandbytes.functional as F
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor

from gact.memory_efficient_function import (
    per_block_quantization,
    per_block_dequantization,
    dct_compression,
    jpeg_compression,
    naive_adjustment,
)


class EfficientMemoryLinearFunc(torch.autograd.Function):
    # only suitable for batched matmul: (BxMxK) @ (KxR) -> (BxKxR) or (BxKxR) @ (RxN) -> (BxKxN)
    # and LoRA do not have bias
    @staticmethod
    def forward(
        ctx,
        x,
        w,
        b,
        use_bias,
        compress_type,
        jpeg_processor,
        dct_processor,
        prune_ratio,
        iteration,
        static_value,
        gradient_mask,
    ):
        # print(x.shape, w.shape)
        if use_bias:
            ctx.needs_inputs_grad = [x.requires_grad, w.requires_grad, b.requires_grad]
        else:
            ctx.needs_inputs_grad = [x.requires_grad, w.requires_grad]
        ctx.compress_type = compress_type
        ctx.use_bias = use_bias
        if use_bias:
            output = (
                x @ w.transpose(0, 1) + b[None, ...]
            )  # TODO: what is the dimension of b?
        else:
            output = x @ w.transpose(0, 1)
        
        kth_val = None
        if compress_type == "NF4":
            # quantize the cached activation
            x, quant_state = F.quantize_nf4(x)
            ctx.quant_state = quant_state
        elif compress_type == "PRUNE_ROW":
            if iteration < 10:
                kth_val = torch.kthvalue(
                    x.abs().flatten(), int(x.numel() * prune_ratio)
                ).values
            else:
                kth_val = static_value
            mask = x.abs() > kth_val
            x = x * mask
        elif compress_type != "NONE":
            # shape preparation for DCT
            input_shape = x.shape
            ctx.input_shape = input_shape
            bit = 4 if compress_type == "INT4" else 8
            # quantize the cached activation
            x, quant_state = per_block_quantization(x, input_shape, 64, bit=bit)
            ctx.quant_state = quant_state

            if compress_type == "PRUNE":
                kth_val = torch.kthvalue(x.abs().flatten(), int(x.numel() * 0.1)).values
                x = torch.where(x.abs() < kth_val, torch.zeros_like(x), x)
                x = naive_adjustment(x, input_shape)

            # compress the cached activation
            if compress_type == "JPEG":
                x = jpeg_compression(x, input_shape, jpeg_processor)

            elif compress_type == "DCT":
                x = dct_compression(x, input_shape, dct_processor)

            elif compress_type == "NAIVE" or compress_type == "INT4":
                x = naive_adjustment(x, input_shape)

        # if the compress type is not JPEG or DCT, then the input will not be compressed(do nothing)
        ctx.save_for_backward(x, w)
        ctx.gradient_mask = gradient_mask
        
        if kth_val is None:
            kth_val = torch.tensor(0.0)
        ctx.mark_non_differentiable(kth_val)
        
        return output, kth_val

    @staticmethod
    def backward(ctx, grad_output, grad_kth_val):
        use_bias = ctx.use_bias
        x, w = ctx.saved_tensors

        if ctx.compress_type == "NF4":
            x = F.dequantize_nf4(x, ctx.quant_state)

        elif (
            ctx.needs_inputs_grad[1]
            and ctx.compress_type != "NONE"
            and ctx.compress_type != "PRUNE_ROW"
        ):
            quant_state = ctx.quant_state
            input_shape = ctx.input_shape
            # dequantize the cached activation
            x = per_block_dequantization(x, input_shape, quant_state)

        grad_input = grad_weight = grad_bias = None
        grad_input = grad_output @ w
        grad_weight = grad_output.transpose(-2, -1) @ x
        if ctx.gradient_mask:
            grad_mask = torch.cat([torch.ones_like(grad_weight)[...,:grad_weight.shape[-1]//2], torch.zeros_like(grad_weight)[...,:grad_weight.shape[-1]//2]], dim=-1)
            grad_weight *= grad_mask
        if use_bias:
            grad_bias = grad_output.sum(0)

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


class EfficientMemoryLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        compress_type: str = "JPEG",
        compress_quality: int = 50,
        prune_ratio: float = 0.9,
        gradient_mask: bool = False,
    ):
        super().__init__(in_features, out_features, bias)
        self.compress_type = compress_type
        self.compress_quality = compress_quality
        self.jpeg_processor = JPEGProcessor(quality=compress_quality)
        self.dct_processor = DCTProcessor(quality=compress_quality)
        self.prune_ratio = prune_ratio
        self.iteration = 0
        self.gradient_mask = gradient_mask
        self.static_value = None

    def forward(self, input: torch.Tensor):
        if self.extract_mode:
            torch.save(input, f"output/{self.name}.pt")

        result, static_value = EfficientMemoryLinearFunc.apply(
            input,
            self.weight,
            self.bias,
            self.bias != None,
            self.compress_type,
            self.jpeg_processor,
            self.dct_processor,
            self.prune_ratio,
            self.iteration,
            self.static_value,
            self.gradient_mask,
        )

        self.static_value = (
            static_value
            if self.static_value is None
            else (self.iteration * self.static_value + static_value)
            / (self.iteration + 1)
        )
        self.iteration += 1

        return result

class EfficientMaskedLinearFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, gradient_mask=False):
        output = x @ w.transpose(0, 1)
        ctx.save_for_backward(x, w)
        ctx.gradient_mask = gradient_mask
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad_x = grad_output @ w
        grad_w = grad_output.transpose(-2, -1) @ x
        # mask the grad_w to 0
        if ctx.gradient_mask:
            grad_mask = torch.cat([torch.ones_like(grad_w)[...,:grad_w.shape[-1]//2], torch.zeros_like(grad_w)[...,:grad_w.shape[-1]//2]], dim=-1)
            grad_w *= grad_mask
        return grad_x, grad_w, None
    
    
class EfficientMaskedLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=False, gradient_mask=False):
        super(EfficientMaskedLinear, self).__init__(in_features, out_features, bias)
        self.gradient_mask = gradient_mask
        
    def forward(self, x):
        return EfficientMaskedLinearFunc.apply(x, self.weight, self.gradient_mask)