import torch
from .compress_function import (
    fake_divide_outliner_suboutlinear_svd,
    prune_softmax,
    get_statistics,
    get_statistics_softmax
)

class EfficientMemoryGEMMFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x1,
        x2,
        outliner_ratio_1,
        sub_outliner_ratio_1,
        sub_outliner_bit_1,
        outliner_ratio_2,
        sub_outliner_ratio_2,
        sub_outliner_bit_2,
        rank,
        iteration,
        static_value_1,
        static_value_2,
    ):
        result = x1 @ x2
        
        # we just need to use the first batch to calculate the outliner
        # for the value 1
        if iteration < 10:
            outliner_1, max_norm_column_list_1, scale_1 = get_statistics(x1, iteration, outliner_ratio_1, sub_outliner_ratio_1, sub_outliner_bit_1)
            outliner_2, max_norm_column_list_2, scale_2 = get_statistics(x2.mT, iteration, outliner_ratio_2, sub_outliner_ratio_2, sub_outliner_bit_2)
            max_norm_column_list_1 = torch.tensor(max_norm_column_list_1)
            max_norm_column_list_2 = torch.tensor(max_norm_column_list_2)
        else:
            outliner_1 = static_value_1[0]
            max_norm_column_list_1 = static_value_1[1]
            scale_1 = static_value_1[2]
            outliner_2 = static_value_2[0]
            max_norm_column_list_2 = static_value_2[1]
            scale_2 = static_value_2[2]
        
        x1 = fake_divide_outliner_suboutlinear_svd(x1, outliner_1, max_norm_column_list_1, scale_1, rank, sub_outliner_bit_1, sub_outliner_ratio_1)
        x2 = fake_divide_outliner_suboutlinear_svd(x2.mT, outliner_2, max_norm_column_list_2, scale_2, rank, sub_outliner_bit_2, sub_outliner_ratio_2).mT
        
        ctx.mark_non_differentiable(outliner_1, max_norm_column_list_1, outliner_2, max_norm_column_list_2)
        
        ctx.save_for_backward(x1, x2)
        return result, outliner_1, max_norm_column_list_1, scale_1, outliner_2, max_norm_column_list_2, scale_2
            
    def backward(ctx, grad_output, grad_outliner_1, grad_max_norm_column_list_1, grad_scale_1, grad_outliner_2, grad_max_norm_column_list_2, grad_scale_2):
        x1, x2 = ctx.saved_tensors
        grad_input1, grad_input2 = None, None

        grad_input1 = grad_output @ x2.transpose(-2, -1)
        grad_input2 = x1.transpose(-2, -1) @ grad_output

        return (
            grad_input1,
            grad_input2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


class EfficientMemoryGEMM(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio_1: float = 0.01,
        sub_outliner_ratio_1: float = 0.2,
        sub_outliner_bit_1: int = 8,
        outliner_ratio_2: float = 0.01,
        sub_outliner_ratio_2: float = 0.2,
        sub_outliner_bit_2: int = 8,
        rank: int = 16,
    ):
        super(EfficientMemoryGEMM, self).__init__()
        self.outliner_ratio_1 = outliner_ratio_1
        self.sub_outliner_ratio_1 = sub_outliner_ratio_1
        self.sub_outliner_bit_1 = sub_outliner_bit_1
        self.outliner_ratio_2 = outliner_ratio_2
        self.sub_outliner_ratio_2 = sub_outliner_ratio_2
        self.sub_outliner_bit_2 = sub_outliner_bit_2
        self.rank = rank
        self.iteration = 0
        self.static_value_1 = [None, None, None]
        self.static_value_2 = [None, None, None]

    def forward(self, x1, x2):
        result, outliner_1, max_norm_column_list_1, scale_1, outliner_2, max_norm_column_list_2, scale_2 = EfficientMemoryGEMMFunc.apply(
            x1,
            x2,
            self.outliner_ratio_1,
            self.sub_outliner_ratio_1,
            self.sub_outliner_bit_1,
            self.outliner_ratio_2,
            self.sub_outliner_ratio_2,
            self.sub_outliner_bit_2,
            self.rank,
            self.iteration,
            self.static_value_1,
            self.static_value_2,
        )
        if self.iteration <= 10:
            self.static_value_1[0] = (
                outliner_1
                if self.static_value_1[0] is None
                else (self.iteration * self.static_value_1[0] + outliner_1)
                / (self.iteration + 1)
            )
            self.static_value_1[1] = (
                max_norm_column_list_1
                if self.static_value_1[1] is None
                else self.static_value_1[1]
            )
            self.static_value_1[2] = (
                scale_1
                if self.static_value_1[2] is None
                else (self.iteration * self.static_value_1[2] + scale_1)
                / (self.iteration + 1)
            )
            self.static_value_2[0] = (
                outliner_2
                if self.static_value_2[0] is None
                else (self.iteration * self.static_value_2[0] + outliner_2)
                / (self.iteration + 1)
            )
            self.static_value_2[1] = (
                max_norm_column_list_2
                if self.static_value_2[1] is None
                else self.static_value_2[1]
            )
            self.static_value_2[2] = (
                scale_2
                if self.static_value_2[2] is None
                else (self.iteration * self.static_value_2[2] + scale_2)
                / (self.iteration + 1)
            )
            
        self.iteration += 1

        return result


class EfficientMemoryGEMMWithSoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x1,
        x2,
        outliner_ratio_1,
        outliner_ratio_2,
        sub_outliner_ratio_2,
        sub_outliner_bit_2,
        rank,
        iteration,
        static_value_1,
        static_value_2,
    ):
        result = x1 @ x2
        
        # we just need to use the first batch to calculate the outliner
        # for the value 1
        if iteration < 10:
            outliner_1 = get_statistics_softmax(x1, iteration, outliner_ratio_1)
            outliner_2, max_norm_column_list_2, scale_2 = get_statistics(x2, iteration, outliner_ratio_2, sub_outliner_ratio_2, sub_outliner_bit_2)
            max_norm_column_list_2 = torch.tensor(max_norm_column_list_2)
        else:
            outliner_1 = static_value_1
            outliner_2 = static_value_2[0]
            max_norm_column_list_2 = static_value_2[1]
            scale_2 = static_value_2[2]
        
        x1 = prune_softmax(x1, outliner_1)
        x2 = fake_divide_outliner_suboutlinear_svd(x2, outliner_2, max_norm_column_list_2, scale_2, rank, sub_outliner_bit_2)
        
        ctx.mark_non_differentiable(outliner_1, outliner_2, max_norm_column_list_2, scale_2)
        
        ctx.save_for_backward(x1, x2)
        return result, outliner_1, outliner_2, max_norm_column_list_2, scale_2
            
    def backward(ctx, grad_output, grad_outliner_1, grad_outliner_2, grad_max_norm_column_list_2, grad_scale_2):
        x1, x2 = ctx.saved_tensors
        grad_input1, grad_input2 = None, None

        grad_input1 = grad_output @ x2.transpose(-2, -1)
        grad_input2 = x1.transpose(-2, -1) @ grad_output

        return (
            grad_input1,
            grad_input2,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


class EfficientMemoryGEMMWithSoftmax(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio_1: float = 0.01,
        outliner_ratio_2: float = 0.01,
        sub_outliner_ratio_2: float = 0.2,
        sub_outliner_bit_2: int = 8,
        rank: int = 16,
    ):
        super(EfficientMemoryGEMMWithSoftmax, self).__init__()
        self.outliner_ratio_1 = outliner_ratio_1
        self.outliner_ratio_2 = outliner_ratio_2
        self.sub_outliner_ratio_2 = sub_outliner_ratio_2
        self.sub_outliner_bit_2 = sub_outliner_bit_2
        self.rank = rank
        self.iteration = 0
        self.static_value_1 = None
        self.static_value_2 = [None, None, None]

    def forward(self, x1, x2):
        result, outliner_1, outliner_2, max_norm_column_list_2, scale_2 = EfficientMemoryGEMMWithSoftmaxFunc.apply(
            x1,
            x2,
            self.outliner_ratio_1,
            self.outliner_ratio_2,
            self.sub_outliner_ratio_2,
            self.sub_outliner_bit_2,
            self.rank,
            self.iteration,
            self.static_value_1,
            self.static_value_2,
        )
        if self.iteration <= 10:
            self.static_value_1 = (
                outliner_1
                if self.static_value_1 is None
                else (self.iteration * self.static_value_1 + outliner_1)
                / (self.iteration + 1)
            )
            self.static_value_2[0] = (
                outliner_2
                if self.static_value_2[0] is None
                else (self.iteration * self.static_value_2[0] + outliner_2)
                / (self.iteration + 1)
            )
            self.static_value_2[1] = (
                max_norm_column_list_2
                if self.static_value_2[1] is None
                else self.static_value_2[1]
            )
            self.static_value_2[2] = (
                scale_2
                if self.static_value_2[2] is None
                else (self.iteration * self.static_value_2[2] + scale_2)
                / (self.iteration + 1)
            )
            
        self.iteration += 1

        return result
