import torch
import bitsandbytes.functional as F
from .compress_function import (
    fake_divide_outliner_suboutlinear_svd,
    get_statistics
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
        outliner_ratio,
        sub_outliner_ratio,
        sub_outliner_bit,
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
            outliner, max_norm_column_list, scale = get_statistics(x, iteration, outliner_ratio, sub_outliner_ratio, sub_outliner_bit)
            # inorder to mark save_for_backward, we should convert the tensor
            max_norm_column_list = torch.tensor(max_norm_column_list)
        else:
            outliner = static_value[0]
            max_norm_column_list = static_value[1]
            scale = static_value[2]
            
        x = fake_divide_outliner_suboutlinear_svd(x, outliner, max_norm_column_list, scale, rank, sub_outliner_bit, sub_outliner_ratio)
        
        ctx.mark_non_differentiable(outliner, max_norm_column_list)
        ctx.save_for_backward(x, w)
        
        return output, outliner, max_norm_column_list, scale

    @staticmethod
    def backward(ctx, grad_output, grad_outliner, grad_max_norm_column_list, grad_scale):
        use_bias = ctx.use_bias
        x, w = ctx.saved_tensors

        grad_input = grad_weight = grad_bias = None
        grad_input = grad_output @ w
        grad_weight = grad_output.transpose(-2, -1) @ x
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
            None
        )


class EfficientMemoryLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        outliner_ratio: float = 0.01,
        sub_outliner_ratio: float = 0.2, #! initialize
        sub_outliner_bit: int = 8,
        rank: int = 16,
    ):
        super(EfficientMemoryLinear, self).__init__(in_features, out_features, bias)
        self.outliner_ratio = outliner_ratio
        self.sub_outliner_ratio = sub_outliner_ratio
        self.sub_outliner_bit = sub_outliner_bit
        self.rank = rank
        self.iteration = 0
        self.static_value = [None, None, None]

    def forward(self, input: torch.Tensor):
        result, outliner, max_norm_column_list, scale = EfficientMemoryLinearFunc.apply(
            input,
            self.weight,
            self.bias,
            self.bias != None,
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