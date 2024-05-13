import math
import torch
import torch.nn.functional as F
from .compress_function import (
    true_divide_outliner_suboutlinear_svd_compress,
    true_divide_outliner_suboutlinear_svd_decompress,
    get_statistics
)

class EfficientMemorySiLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        outliner_ratio,
        sub_outliner_ratio,
        sub_outliner_bit,
        sub_outlier_quantize_method,
        rank, 
        iteration,
        static_value,
    ):
        result = F.silu(x)
        
        # we just need to use the first batch to calculate the outliner
        if iteration < 10:
            outliner, L, R, scale = get_statistics(x, iteration, outliner_ratio, sub_outliner_ratio, sub_outliner_bit, sub_outlier_quantize_method)
        else:
            outliner = static_value[0]
            L = static_value[1]
            scale = static_value[2]
            R = static_value[3]
            
        x_outlier_compressed, x_sub_outliner_compressed, scale = true_divide_outliner_suboutlinear_svd_compress(x, outliner, scale, sub_outliner_bit, sub_outliner_ratio, L, R)
        
        ctx.mark_non_differentiable(outliner, L, R, scale)
        ctx.save_for_backward(x_outlier_compressed, x_sub_outliner_compressed, scale, L, R)
        ctx.sub_outliner_bit = sub_outliner_bit
        
        return result, outliner, L, R, scale

    @staticmethod
    def backward(ctx, grad_output, grad_outliner, grad_L, grad_R, grad_scale):
        x_outlier_compressed, x_sub_outliner_compressed, scale, L, R = ctx.saved_tensors
        x = true_divide_outliner_suboutlinear_svd_decompress(x_outlier_compressed, x_sub_outliner_compressed, ctx.sub_outliner_bit, scale, L=L, R=R)
        
        sigmoid = F.sigmoid(x)
        grad_input = sigmoid * (1 + x - x * sigmoid) * grad_output

        return grad_input, None, None, None, None, None, None, None


class EfficientMemorySiLU(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio: float = 0.01,
        sub_outliner_ratio: float = 0.2, #! initialize
        sub_outliner_bit: int = 8,
        sub_outlier_quantize_method: str = 'per-tensor',
        rank: int = 16,
    ):
        super(EfficientMemorySiLU, self).__init__()
        self.outliner_ratio = outliner_ratio
        self.sub_outliner_ratio = sub_outliner_ratio
        self.sub_outliner_bit = sub_outliner_bit
        self.sub_outlier_quantize_method = sub_outlier_quantize_method
        self.rank = rank
        self.iteration = 0
        self.static_value = [None, None, None, None]

    def forward(self, input):
        result, outliner, L, R, scale = EfficientMemorySiLUFunc.apply(
            input,
            self.outliner_ratio,
            self.sub_outliner_ratio,
            self.sub_outliner_bit,
            self.sub_outlier_quantize_method,
            self.rank,
            self.iteration,
            self.static_value,
        )

        if self.iteration <= 10:
            self.static_value[0] = (
                outliner
                if self.static_value[0] is None
                else (self.iteration * self.static_value[0] + outliner)
                / (self.iteration + 1)
            )
            self.static_value[1] = (
                L
                if self.static_value[1] is None
                else (self.iteration * self.static_value[1] + L) 
                / (self.iteration + 1)
            )
            self.static_value[2] = (
                scale
                if self.static_value[2] is None
                else (self.iteration * self.static_value[2] + scale) 
                / (self.iteration + 1)
            )
            self.static_value[3] = (
                R
                if self.static_value[3] is None
                else (self.iteration * self.static_value[3] + R) 
                / (self.iteration + 1)
            )
        self.iteration += 1

        return result
