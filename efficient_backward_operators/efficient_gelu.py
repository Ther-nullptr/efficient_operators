import math
import torch
import torch.nn.functional as F
from compress_function import (
    fake_divide_outliner_suboutlinear_svd,
    get_statistics
)

class EfficientMemoryGELUFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        outliner_ratio,
        sub_outliner_ratio,
        rank, 
        iteration,
        static_value,
    ):
        result = F.gelu(x)
        
        # we just need to use the first batch to calculate the outliner
        if iteration < 10:
            outliner, max_norm_column_list, scale = get_statistics(x, iteration, outliner_ratio, sub_outliner_ratio)
        else:
            outliner = static_value[0]
            max_norm_column_list = static_value[1]            

        # inorder to mark save_for_backward, we should convert the tensor
        max_norm_column_list = torch.tensor(max_norm_column_list)
            
        x = fake_divide_outliner_suboutlinear_svd(x, outliner, max_norm_column_list, scale, rank)
        
        ctx.mark_non_differentiable(outliner, max_norm_column_list)
        ctx.save_for_backward(x)
        
        return result, outliner, max_norm_column_list

    @staticmethod
    def backward(ctx, grad_output, grad_outliner, grad_max_norm_column_list):
        (x,) = ctx.saved_tensors

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

        return grad_input, None, None, None, None, None


class EfficientMemoryGELU(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio: float = 0.01,
        sub_outliner_ratio: float = 0.1, #! initialize
        rank: int = 16,
    ):
        super(EfficientMemoryGELU, self).__init__()
        self.outliner_ratio = outliner_ratio
        self.sub_outliner_ratio = sub_outliner_ratio
        self.rank = rank
        self.iteration = 0
        self.static_value = [None, None]

    def forward(self, input):
        result, outliner, max_norm_column_list = EfficientMemoryGELUFunc.apply(
            input,
            self.outliner_ratio,
            self.sub_outliner_ratio,
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
        self.iteration += 1

        return result
