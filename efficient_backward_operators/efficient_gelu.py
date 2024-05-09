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
        Rinv,
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
        x_outlier_compressed, L, R, Rinv = true_divide_outliner_suboutlinear_svd_compress(x, outliner, execute_svd, R, Rinv, rank)
        
        ctx.mark_non_differentiable(outliner, R, Rinv)
        ctx.save_for_backward(x_outlier_compressed, L, R)
        
        return result, outliner, R, Rinv

    @staticmethod
    def backward(ctx, grad_output, grad_outliner, grad_R, grad_Rinv):
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
        rank: int = 16,
    ):
        super(EfficientMemoryGELU, self).__init__()
        self.outliner_ratio = outliner_ratio
        self.rank = rank
        self.iteration = 0
        self.static_value = None
        self.R = None
        self.Rinv = None

    def forward(self, input):
        result, outliner, R, Rinv = EfficientMemoryGELUFunc.apply(
            input,
            self.R,
            self.Rinv,
            self.outliner_ratio,
            self.rank,
            self.iteration,
            self.static_value,
        )
        
        self.R = R
        self.Rinv = Rinv

        if self.iteration <= 10:
            self.static_value = (
                outliner
                if self.static_value is None
                else (self.iteration * self.static_value + outliner)
                / (self.iteration + 1)
            )
        self.iteration += 1

        return result
