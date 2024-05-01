import torch
import torch.nn.functional as F
from .compress_function import (
    prune_softmax,
    get_statistics_softmax
)

class EfficientMemorySoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        outliner_ratio,
        iteration,
        static_value,
    ):
        y_return = F.softmax(x, dim=-1)
        y = y_return.clone()
        
        if iteration < 10:
            outliner = get_statistics_softmax(y, iteration, outliner_ratio)
        else:
            outliner = static_value

        y = prune_softmax(y, outliner)
        
        ctx.mark_non_differentiable(outliner)
        ctx.save_for_backward(y)
        
        return y_return, outliner

    @staticmethod
    def backward(ctx, grad_output, grad_outliner):
        (y,) = ctx.saved_tensors

        return (
            (grad_output - (grad_output * y).sum(dim=-1, keepdims=True)) * y,
            None,
            None,
            None,
        )


class EfficientMemorySoftmax(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio: float = 0.01,
    ):
        super(EfficientMemorySoftmax, self).__init__()
        self.outliner_ratio = outliner_ratio
        self.iteration = 0
        self.static_value = None

    def forward(self, x):
        result, outliner = EfficientMemorySoftmaxFunc.apply(
            x,
            self.outliner_ratio,
            self.iteration,
            self.static_value,
        )
        
        if self.iteration <= 10:
            self.static_value = (
                outliner
                if self.static_value is None
                else (self.iteration * self.static_value + outliner)
                / (self.iteration + 1)
            )
        self.iteration += 1

        return result
