import torch
import bitsandbytes.functional as F
from .compress_function import (
    true_divide_outliner_suboutlinear_svd_compress,
    true_divide_outliner_suboutlinear_svd_decompress,
    get_statistics
)


class EfficientMemoryLinearFunc(torch.autograd.Function):
    # only suitable for batched matmul: (BxMxK) @ (KxR) -> (BxKxR) or (BxKxR) @ (RxN) -> (BxKxN)
    # and LoRA do not have bias
    @staticmethod
    def forward(
        ctx,
        x,
        R,
        Rinv,
        w,
        b,
        use_bias,
        outliner_ratio,
        rank,
        iteration,
        static_value,
    ):
        ctx.use_bias = use_bias
        if use_bias:
            output = (
                x @ w.transpose(0, 1) + b[None, ...]
            )  # TODO: what is the dimension of b?
        else:
            output = x @ w.transpose(0, 1)
        
        # we just need to use the first batch to calculate the outliner
        if iteration < 10:
            outliner = get_statistics(x, outliner_ratio, rank)
        else:
            outliner = static_value
        
        execute_svd = (iteration % 50) == 0
        x_outlier_compressed, L, R, Rinv = true_divide_outliner_suboutlinear_svd_compress(x, outliner, execute_svd, R, Rinv, rank)
        
        ctx.mark_non_differentiable(outliner, R, Rinv)
        ctx.save_for_backward(x_outlier_compressed, L, R, w)
        
        return output, outliner, R, Rinv

    @staticmethod
    def backward(ctx, grad_output, grad_outliner, grad_R, grad_Rinv):
        use_bias = ctx.use_bias
        x_outlier_compressed, L, R, w = ctx.saved_tensors
        x = true_divide_outliner_suboutlinear_svd_decompress(x_outlier_compressed, L, R)

        grad_input = grad_weight = grad_bias = None
        grad_input = grad_output @ w
        grad_weight = grad_output.transpose(-2, -1) @ x
        if use_bias:
            grad_bias = grad_output.sum(0)

        return (
            grad_input,
            None,
            None,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


class EfficientMemoryLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        outliner_ratio: float = 0.01,
        rank: int = 16,
    ):
        super(EfficientMemoryLinear, self).__init__(in_features, out_features, bias)
        self.outliner_ratio = outliner_ratio
        self.rank = rank
        self.iteration = 0
        self.static_value = None
        self.R = None
        self.Rinv = None

    def forward(self, input: torch.Tensor):
        result, outliner, R, Rinv = EfficientMemoryLinearFunc.apply(
            input,
            self.R,
            self.Rinv,
            self.weight,
            self.bias,
            self.bias != None,
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