import torch


def fake_svd_lowrank_simple_tensor(input: torch.Tensor, q: int, niter: int = 2):
    batch, seq_len, model_dim = input.shape
    input = input.reshape(batch * seq_len, model_dim)
    U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
    V = V.transpose(-1, -2)
    S = torch.diag_embed(S)
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, model_dim)
    return output


def fake_svd_lowrank_head(input: torch.Tensor, q: int, niter: int = 2):
    batch, num_head, seq_len, sep_dim = input.shape
    input = input.permute(0, 2, 1, 3).reshape(batch * seq_len, num_head * sep_dim)
    U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
    V = V.transpose(-1, -2)
    S = torch.diag_embed(S)
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output


def svd_lowrank_simple_tensor_compress(input: torch.Tensor, q: int, niter: int = 2):
    batch, seq_len, model_dim = input.shape
    input = input.reshape(batch * seq_len, model_dim)
    U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
    V = V.transpose(-1, -2)
    return U, S, V


def svd_lowrank_simple_tensor_decompress(U: torch.tensor, S: torch.tensor, V: torch.tensor, input_shape: torch.Size):
    batch, seq_len, model_dim = input_shape
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, model_dim)
    return output


def svd_lowrank_head_compress(input: torch.Tensor, q: int, niter: int = 2):
    batch, num_head, seq_len, sep_dim = input.shape
    input = input.permute(0, 2, 1, 3).reshape(batch * seq_len, num_head * sep_dim)
    U, S, V = torch.svd_lowrank(input, q=q, niter=niter)
    V = V.transpose(-1, -2)
    return U, S, V


def svd_lowrank_head_decompress(U: torch.tensor, S: torch.tensor, V: torch.tensor, input_shape: torch.Size):
    batch, seq_len, num_head, sep_dim = input_shape
    output = torch.matmul(U[..., :, :], S[..., :, :])
    output = torch.matmul(output[..., :, :], V[..., :, :])
    output = output.reshape(batch, seq_len, num_head, sep_dim).permute(0, 2, 1, 3)
    return output


def fake_divide_outliner_suboutlinear_svd(x: torch.Tensor, outliner: float, max_norm_column_list: float, scale: float, rank: int):
    # step 1: prune the outliner
    x_outliner = x * (x.abs() > outliner)
    x = x - x_outliner
    
    # step 2: prune the sub_outliner
    x_sub_outliner = torch.zeros_like(x).to(x.device).to(x.dtype)
    x_sub_outliner = x[:, :, max_norm_column_list]
    x = x - x_sub_outliner
    x_sub_outliner = torch.round(x_sub_outliner / scale) * scale
    
    # step 3: apply SVD
    x = fake_svd_lowrank_head(x, rank)
    
    return x + x_outliner + x_sub_outliner
    

def get_statistics(x: torch.Tensor, iteration: int, outliner_ratio: float, sub_outliner_ratio: float):
    outliner = torch.kthvalue(x[0].flatten(), int(x[0].numel() * (1 - outliner_ratio))).values
    print(f'iter {iteration} | outliner: {outliner}')
    mean_norm = list(torch.mean(torch.abs(x[0]), dim=-1).cpu().detach().numpy())
    max_norm_column_list = sorted(enumerate(mean_norm), key=lambda x: x[1], reverse=True)
    max_norm_column_list = max_norm_column_list[:int(len(max_norm_column_list) * sub_outliner_ratio)]
    print(f'iter {iteration} | max_norm_column_list: {max_norm_column_list}')
    # calculate the quantized version(8 bit first)
    x_outliner = x[0] * (x[0].abs() > outliner)
    x = x - x_outliner
    x_sub_outliner = x[0][:, max_norm_column_list]
    # TODO: set the scale factor to per channel or per tensor?
    scale = (x_sub_outliner.max() - x_sub_outliner.min()) / 255
    
    return outliner, max_norm_column_list, scale