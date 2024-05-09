import torch
from .compress_function import (
    true_divide_outliner_suboutlinear_svd_compress,
    true_divide_outliner_suboutlinear_svd_decompress,
    get_statistics
)

class EfficientMemoryHadamardFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x1,
        R1,
        Rinv1,
        x2,
        R2,
        Rinv2,
        outliner_ratio_1,
        outliner_ratio_2,
        rank,
        iteration,
        static_value_1,
        static_value_2,
    ):
        result = x1 * x2
        
        # we just need to use the first batch to calculate the outliner
        # for the value 1
        if iteration < 10:
            outliner_1 = get_statistics(x1, outliner_ratio_1, rank)
            outliner_2 = get_statistics(x2, outliner_ratio_2, rank)
        else:
            outliner_1 = static_value_1
            outliner_2 = static_value_2
            
        execute_svd = (iteration % 50) == 0
        
        x1_outlier_compressed, L1, R1, Rinv1 = true_divide_outliner_suboutlinear_svd_compress(x1, outliner_1, execute_svd, R1, Rinv1, rank)
        x2_outlier_compressed, L2, R2, Rinv2 = true_divide_outliner_suboutlinear_svd_compress(x2, outliner_2, execute_svd, R2, Rinv2, rank)
        
        ctx.mark_non_differentiable(outliner_1, outliner_2, R1, Rinv1, R2, Rinv2)
        ctx.save_for_backward(x1_outlier_compressed, L1, R1, x2_outlier_compressed, L2, R2)

        return result, outliner_1, R1, Rinv1, outliner_2, R2, Rinv2
            
    def backward(ctx, grad_output, grad_outliner_1, grad_R1, grad_Rinv1, grad_outliner_2, grad_R2, grad_Rinv2):
        x1_outlier_compressed, L1, R1, x2_outlier_compressed, L2, R2 = ctx.saved_tensors
        grad_input1, grad_input2 = None, None
        x1 = true_divide_outliner_suboutlinear_svd_decompress(x1_outlier_compressed, L1, R1)
        x2 = true_divide_outliner_suboutlinear_svd_decompress(x2_outlier_compressed, L2, R2)

        grad_input1 = grad_output * x2
        grad_input2 = x1 * grad_output

        return (
            grad_input1,
            None,
            None,
            grad_input2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class EfficientMemoryHadamard(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio_1: float = 0.01,
        outliner_ratio_2: float = 0.01,
        rank: int = 16,
    ):
        super(EfficientMemoryHadamard, self).__init__()
        self.outliner_ratio_1 = outliner_ratio_1
        self.outliner_ratio_2 = outliner_ratio_2
        self.rank = rank
        self.iteration = 0
        self.static_value_1 = None
        self.static_value_2 = None
        self.R1 = None
        self.Rinv1 = None
        self.R2 = None
        self.Rinv2 = None

    def forward(self, x1, x2):
        result, outliner_1, R1, Rinv1, outliner_2, R2, Rinv2 = EfficientMemoryHadamardFunc.apply(
            x1,
            self.R1,
            self.Rinv1,
            x2,
            self.R2,
            self.Rinv2,
            self.outliner_ratio_1,
            self.outliner_ratio_2,
            self.rank,
            self.iteration,
            self.static_value_1,
            self.static_value_2,
        )
        
        self.R1 = R1
        self.Rinv1 = Rinv1
        self.R2 = R2
        self.Rinv2 = Rinv2

        if self.iteration <= 10:
            self.static_value_1 = (
                outliner_1
                if self.static_value_1 is None
                else (self.iteration * self.static_value_1 + outliner_1)
                / (self.iteration + 1)
            )
            self.static_value_2 = (
                outliner_2
                if self.static_value_2 is None
                else (self.iteration * self.static_value_2 + outliner_2)
                / (self.iteration + 1)
            )
            
        self.iteration += 1

        return result

