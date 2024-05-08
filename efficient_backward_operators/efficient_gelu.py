import math
import torch
import torch.nn.functional as F
from .compress_function import (
    true_divide_outliner_suboutlinear_svd_compress,
    true_divide_outliner_suboutlinear_svd_decompress,
    get_statistics
)

class EfficientMemoryGELUFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        R,
        R_inv,
        outliner_ratio,
        rank, 
        iteration,
        static_value,
    ):
        result = F.gelu(x)
        
        # we just need to use the first batch to calculate the outliner
        if iteration < 10:
            outliner = get_statistics(x, outliner_ratio, rank)
        else:
            outliner = static_value
        
        execute_svd = (iteration % 50) == 0
        if execute_svd:
            print(f"execute_svd at iteration {iteration}")
        x_outlier_compressed, L, R, R_inv = true_divide_outliner_suboutlinear_svd_compress(x, outliner, execute_svd, R, R_inv, rank)
        
        ctx.mark_non_differentiable(outliner)
        ctx.save_for_backward(x_outlier_compressed, L, R)
        
        return result, outliner, R, R_inv

    @staticmethod
    def backward(ctx, grad_output, grad_outliner, grad_R, grad_R_inv):
        (x_outlier_compressed, L, R) = ctx.saved_tensors
        x = true_divide_outliner_suboutlinear_svd_decompress(x_outlier_compressed, L, R)

        gamma = math.sqrt(2 / math.pi)
        kappa = 0.044715
        grad_input = None

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
        outliner_ratio: float = 0.01,
        sub_outliner_ratio: float = 0.2, #! initialize
        sub_outliner_bit: int = 8,
        sub_outlier_quantize_method: str = 'per-tensor',
        rank: int = 16,
    ):
        super(EfficientMemoryGELU, self).__init__()
        self.outliner_ratio = outliner_ratio
        self.sub_outliner_ratio = sub_outliner_ratio
        self.sub_outliner_bit = sub_outliner_bit
        self.sub_outlier_quantize_method = sub_outlier_quantize_method
        self.rank = rank
        self.iteration = 0
        self.static_value = None
        self.R = None
        self.R_inv = None

    def forward(self, input):
        result, outliner, R, R_inv = EfficientMemoryGELUFunc.apply(
            input,
            self.R,
            self.R_inv,
            self.outliner_ratio,
            self.sub_outliner_ratio,
            self.sub_outliner_bit,
            self.sub_outlier_quantize_method,
            self.rank,
            self.iteration,
            self.static_value,
        )
        
        self.R = R
        self.R_inv = R_inv

        if self.iteration <= 10:
            self.static_value = (
                outliner
                if self.static_value[0] is None
                else (self.iteration * self.static_value[0] + outliner)
                / (self.iteration + 1)
            )
        self.iteration += 1

        return result
