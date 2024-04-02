import torch
import torch.nn.functional as F
import bitsandbytes.functional as BF
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor

from gact.memory_efficient_function import per_block_quantization, per_block_dequantization, dct_compression, jpeg_compression, naive_adjustment

class EfficientMemorySoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, compress_type, jpeg_processor, dct_processor, quantization_shape):
        y_return = F.softmax(x, dim=-1)
        y = y_return.clone()

        # save output instead of input
        if compress_type == 'NF4':
            y, quant_state = BF.quantize_nf4(y)
            ctx.quant_state = quant_state
        elif compress_type != 'NONE':
            input_shape = x.shape
            ctx.input_shape = input_shape
            y, quant_state = per_block_quantization(y, input_shape, quantization_shape)
            ctx.quant_state = quant_state

            if compress_type == 'PRUNE':
                y[y < -100] = -128
                y = naive_adjustment(y, input_shape, quantization_shape)
            if compress_type == 'JPEG':
                y = jpeg_compression(y, input_shape, jpeg_processor, quantization_shape)
            elif compress_type == 'DCT':
                y = dct_compression(y, input_shape, dct_processor, quantization_shape)
            elif compress_type == 'NAIVE':
                y = naive_adjustment(y, input_shape, quantization_shape)

        ctx.save_for_backward(y)
        ctx.compress_type = compress_type
        return y_return
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors

        if ctx.compress_type == 'NF4':
            y = BF.dequantize_nf4(y, ctx.quant_state)
        
        elif ctx.compress_type != 'NONE':
            quant_state = ctx.quant_state
            input_shape = ctx.input_shape
            # dequantize the cached activation
            y = per_block_dequantization(y, input_shape, quant_state)

        return (grad_output - (grad_output * y).sum(dim=-1, keepdims=True)) * y, None, None, None, None
    

class EfficientMemorySoftmax(torch.nn.Module):
    def __init__(self, compress_type: str = "JPEG", compress_quality: int = 50, quantization_shape: int = 64):
        super(EfficientMemorySoftmax, self).__init__()
        self.compress_type = compress_type
        self.compress_quality = compress_quality
        self.jpeg_processor = JPEGProcessor(quality=compress_quality)
        self.dct_processor = DCTProcessor(quality=compress_quality)
        self.quantization_shape = quantization_shape
    
    def forward(self, x):
        result = EfficientMemorySoftmaxFunc.apply(
            x,
            self.compress_type,
            self.jpeg_processor,
            self.dct_processor,
            self.quantization_shape
        )
        # notice softmax save the result of output, instead of input
        if self.extract_mode:
            torch.save(result, f"output/{self.name}.pt")

        return result
