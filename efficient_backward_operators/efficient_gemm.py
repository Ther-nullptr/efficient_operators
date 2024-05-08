import torch
from .compress_function import (
    true_divide_outliner_suboutlinear_svd_compress,
    true_divide_outliner_suboutlinear_svd_decompress,
    true_compress_softmax,
    true_decompress_softmax,
    prune_softmax,
    get_statistics,
    get_statistics_softmax
)

class EfficientMemoryGEMMFunc(torch.autograd.Function):
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
        result = x1 @ x2
        num_heads = x1.shape[1]
        
        # we just need to use the first batch to calculate the outliner
        # for the value 1
        if iteration < 10:
            outliner_1, max_norm_column_list_1, scale_1 = get_statistics(x1, outliner_ratio_1, rank)
            outliner_2, max_norm_column_list_2, scale_2 = get_statistics(x2.mT, outliner_ratio_2, rank)
        else:
            outliner_1 = static_value_1
            outliner_2 = static_value_2
        
        execute_svd = (iteration % 50) == 0
        if execute_svd:
            print(f"execute_svd at iteration {iteration}")
        
        x1_outlier_compressed, L1, R1, Rinv1 = true_divide_outliner_suboutlinear_svd_compress(x1, outliner_1, scale_1, sub_outliner_bit_1, sub_outliner_ratio_1)
        x2_outlier_compressed, L2, R2, Rinv2 = true_divide_outliner_suboutlinear_svd_compress(x2.mT, outliner_2, scale_2, sub_outliner_bit_2, sub_outliner_ratio_2)
        ctx.num_heads = num_heads
        ctx.mark_non_differentiable(outliner_1, max_norm_column_list_1, outliner_2, max_norm_column_list_2)
        ctx.save_for_backward(x1_outlier_compressed, x1_sub_outliner_compressed, scale1, x2_outlier_compressed, x2_sub_outliner_compressed, scale2)

        return result, outliner_1, max_norm_column_list_1, scale_1, outliner_2, max_norm_column_list_2, scale_2
            
    def backward(ctx, grad_output, grad_outliner_1, grad_max_norm_column_list_1, grad_scale_1, grad_outliner_2, grad_max_norm_column_list_2, grad_scale_2):
        x1_outlier_compressed, x1_sub_outliner_compressed, scale1, x2_outlier_compressed, x2_sub_outliner_compressed, scale2 = ctx.saved_tensors
        grad_input1, grad_input2 = None, None
        
        x1 = true_divide_outliner_suboutlinear_svd_decompress(x1_outlier_compressed, x1_sub_outliner_compressed, ctx.sub_outliner_bit_1, scale1, True, ctx.num_heads)
        x2 = true_divide_outliner_suboutlinear_svd_decompress(x2_outlier_compressed, x2_sub_outliner_compressed, ctx.sub_outliner_bit_2, scale2, True, ctx.num_heads).mT

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
        sub_outlier_quantize_method_1: str = 'per-tensor',
        outliner_ratio_2: float = 0.01,
        sub_outliner_ratio_2: float = 0.2,
        sub_outliner_bit_2: int = 8,
        sub_outlier_quantize_method_2: str = 'per-tensor',
        rank: int = 16,
    ):
        super(EfficientMemoryGEMM, self).__init__()
        self.outliner_ratio_1 = outliner_ratio_1
        self.sub_outliner_ratio_1 = sub_outliner_ratio_1
        self.sub_outliner_bit_1 = sub_outliner_bit_1
        self.sub_outlier_quantize_method_1 = sub_outlier_quantize_method_1
        self.outliner_ratio_2 = outliner_ratio_2
        self.sub_outliner_ratio_2 = sub_outliner_ratio_2
        self.sub_outliner_bit_2 = sub_outliner_bit_2
        self.sub_outlier_quantize_method_2 = sub_outlier_quantize_method_2
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
            self.sub_outlier_quantize_method_1,
            self.outliner_ratio_2,
            self.sub_outliner_ratio_2,
            self.sub_outliner_bit_2,
            self.sub_outlier_quantize_method_2,
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
        sub_outlier_quantize_method_2,
        rank,
        iteration,
        static_value_1,
        static_value_2,
    ):
        result = x1 @ x2
        num_heads = x1.shape[1]
        
        # we just need to use the first batch to calculate the outliner
        # for the value 1
        if iteration < 10:
            outliner_1 = get_statistics_softmax(x1, iteration, outliner_ratio_1)
            outliner_2, max_norm_column_list_2, scale_2 = get_statistics(x2, iteration, outliner_ratio_2, sub_outliner_ratio_2, sub_outliner_bit_2, sub_outlier_quantize_method_2)
            max_norm_column_list_2 = torch.tensor(max_norm_column_list_2)
        else:
            outliner_1 = static_value_1
            outliner_2 = static_value_2[0]
            max_norm_column_list_2 = static_value_2[1]
            scale_2 = static_value_2[2]
        
        x1_sparse = true_compress_softmax(x1, outliner_1)
        x2_outlier_compressed, x2_sub_outliner_compressed, scale_2 = true_divide_outliner_suboutlinear_svd_compress(x2, outliner_2, scale_2, sub_outliner_bit_2, sub_outliner_ratio_2)
        
        ctx.mark_non_differentiable(outliner_1, outliner_2, max_norm_column_list_2, scale_2)
        ctx.save_for_backward(x1_sparse, x2_outlier_compressed, x2_sub_outliner_compressed, scale_2)
        ctx.sub_outliner_bit_2 = sub_outliner_bit_2
        ctx.num_heads = num_heads
        return result, outliner_1, outliner_2, max_norm_column_list_2, scale_2
            
    def backward(ctx, grad_output, grad_outliner_1, grad_outliner_2, grad_max_norm_column_list_2, grad_scale_2):
        x1_sparse, x2_outlier_compressed, x2_sub_outliner_compressed, scale_2 = ctx.saved_tensors
        grad_input1, grad_input2 = None, None
        
        x1 = true_decompress_softmax(x1_sparse)
        x2 = true_divide_outliner_suboutlinear_svd_decompress(x2_outlier_compressed, x2_sub_outliner_compressed, ctx.sub_outliner_bit_2, scale_2, True, ctx.num_heads)
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
            None
        )


class EfficientMemoryGEMMWithSoftmax(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio_1: float = 0.01,
        outliner_ratio_2: float = 0.01,
        sub_outliner_ratio_2: float = 0.2,
        sub_outliner_bit_2: int = 8,
        sub_outlier_quantize_method_2: str = 'per-tensor',
        rank: int = 16,
    ):
        super(EfficientMemoryGEMMWithSoftmax, self).__init__()
        self.outliner_ratio_1 = outliner_ratio_1
        self.outliner_ratio_2 = outliner_ratio_2
        self.sub_outliner_ratio_2 = sub_outliner_ratio_2
        self.sub_outliner_bit_2 = sub_outliner_bit_2
        self.sub_outlier_quantize_method_2 = sub_outlier_quantize_method_2
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
            self.sub_outlier_quantize_method_2,
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
