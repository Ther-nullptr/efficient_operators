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
        rank, 
        iteration,
        static_value,
    ):
        result = F.silu(x)
        
        # we just need to use the first batch to calculate the outliner
        if iteration < 10:
            outliner, max_norm_column_list, scale = get_statistics(x, iteration, outliner_ratio, sub_outliner_ratio, sub_outliner_bit)
            # inorder to mark save_for_backward, we should convert the tensor
            max_norm_column_list = torch.tensor(max_norm_column_list)
        else:
            outliner = static_value[0]
            max_norm_column_list = static_value[1]
            scale = static_value[2]
            
        x_outlier_compressed, x_sub_outliner_compressed, scale = true_divide_outliner_suboutlinear_svd_compress(x, outliner, scale, sub_outliner_bit, sub_outliner_ratio)
        
        ctx.mark_non_differentiable(outliner, max_norm_column_list)
        ctx.save_for_backward(x_outlier_compressed, x_sub_outliner_compressed, scale)
        ctx.sub_outliner_bit = sub_outliner_bit
        
        return result, outliner, max_norm_column_list, scale

    @staticmethod
    def backward(ctx, grad_output, grad_outliner, grad_max_norm_column_list, grad_scale):
        (x_outlier_compressed, x_sub_outliner_compressed, scale) = ctx.saved_tensors
        x = true_divide_outliner_suboutlinear_svd_decompress(x_outlier_compressed, x_sub_outliner_compressed, ctx.sub_outliner_bit, scale)
        
        sigmoid = F.sigmoid(x)
        grad_input = sigmoid * (1 + x - x * sigmoid) * grad_output

        return grad_input, None, None, None, None, None, None


class EfficientMemorySiLU(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio: float = 0.01,
        sub_outliner_ratio: float = 0.2, #! initialize
        sub_outliner_bit: int = 8,
        rank: int = 16,
    ):
        super(EfficientMemorySiLU, self).__init__()
        self.outliner_ratio = outliner_ratio
        self.sub_outliner_ratio = sub_outliner_ratio
        self.sub_outliner_bit = sub_outliner_bit
        self.rank = rank
        self.iteration = 0
        self.static_value = [None, None, None]

    def forward(self, input):
        result, outliner, max_norm_column_list, scale = EfficientMemorySiLUFunc.apply(
            input,
            self.outliner_ratio,
            self.sub_outliner_ratio,
            self.sub_outliner_bit,
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
                max_norm_column_list
                if self.static_value[1] is None
                else self.static_value[1]
            )
            self.static_value[2] = (
                scale
                if self.static_value[2] is None
                else (self.iteration * self.static_value[2] + scale) 
                / (self.iteration + 1)
            )
        self.iteration += 1

        return result
