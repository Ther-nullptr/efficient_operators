import torch
from compress_function import (
    fake_divide_outliner_suboutlinear_svd,
    get_statistics
)

num_heads, head_dim = 32, 128
def _shape(tensor: torch.Tensor, seq_len: int, bsz: int):
    # (bsz, seq_len, hidden_dim) -> (bsz, num_heads, seq_len, head_dim) -> (bsz * num_heads, seq_len, head_dim)
    tensor = tensor.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    tensor = tensor.reshape(-1, *tensor.shape[2:]).contiguous()
    return tensor

class EfficientMemoryGEMMFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x1,
        x2,
        outliner_ratio_1,
        sub_outliner_ratio_1,
        outliner_ratio_2,
        sub_outliner_ratio_2,
        rank, 
        iteration,
        static_value_1,
        static_value_2,
    ):
        result = x1 @ x2
        
        # we just need to use the first batch to calculate the outliner
        # for the value 1
        if iteration < 10:
            outliner_1, max_norm_column_list_1, scale_1 = get_statistics(x1, iteration, outliner_ratio_1, sub_outliner_ratio_1)
            outliner_2, max_norm_column_list_2, scale_2 = get_statistics(x2, iteration, outliner_ratio_2, sub_outliner_ratio_2)
        else:
            outliner_1 = static_value_1[0]
            max_norm_column_list_1 = static_value_1[1]
            outliner_2 = static_value_2[0]
            max_norm_column_list_2 = static_value_2[1]
            
        max_norm_column_list_1 = torch.tensor(max_norm_column_list_1)
        max_norm_column_list_2 = torch.tensor(max_norm_column_list_2)
        
        x1 = fake_divide_outliner_suboutlinear_svd(x1, outliner_1, max_norm_column_list_1, scale_1, rank)
        x2 = fake_divide_outliner_suboutlinear_svd(x2, outliner_2, max_norm_column_list_2, scale_2, rank)
        
        ctx.mark_non_differentiable(outliner_1, max_norm_column_list_1, outliner_2, max_norm_column_list_2)
        
        ctx.save_for_backward(x1, x2)
        return result, outliner_1, max_norm_column_list_1, outliner_2, max_norm_column_list_2
            
    def backward(ctx, grad_output, grad_outliner_1, grad_max_norm_column_list_1, grad_outliner_2, grad_max_norm_column_list_2):
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
        )


class EfficientMemoryGEMM(torch.nn.Module):
    def __init__(
        self,
        outliner_ratio_1: float = 0.01,
        sub_outliner_ratio_1: float = 0.1,
        outliner_ratio_2: float = 0.01,
        sub_outliner_ratio_2: float = 0.1,
        rank: int = 16,
    ):
        super(EfficientMemoryGEMM).__init__()
        self.outliner_ratio_1 = outliner_ratio_1
        self.sub_outliner_ratio_1 = sub_outliner_ratio_1
        self.outliner_ratio_2 = outliner_ratio_2
        self.sub_outliner_ratio_2 = sub_outliner_ratio_2
        self.rank = rank
        self.iteration = 0
        self.static_value_1 = [None, None]
        self.static_value_2 = [None, None]

    def forward(self, x1, x2):
        result, outliner_1, max_norm_column_list_1, outliner_2, max_norm_column_list_2 = EfficientMemoryGEMMFunc.apply(
            x1,
            x2,
            self.outliner_ratio_1,
            self.sub_outliner_ratio_1,
            self.outliner_ratio_2,
            self.sub_outliner_ratio_2,
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
                else (self.iteration * self.static_value_1[1] + max_norm_column_list_1)
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
                else (self.iteration * self.static_value_2[1] + max_norm_column_list_2)
                / (self.iteration + 1)
            )
        self.iteration += 1

        return result

