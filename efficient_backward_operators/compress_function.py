import torch

def hidden_to_head_shape(x: torch.Tensor, num_heads: int):
    bsz, seq_len, hidden_dim = x.shape
    head_dim = hidden_dim // num_heads
    return x.reshape(bsz, seq_len, num_heads, head_dim).transpose(1, 2)


def head_to_hidden_shape(x: torch.Tensor):
    bsz, num_heads, seq_len, head_dim = x.shape
    return x.transpose(1, 2).reshape(bsz, seq_len, -1)


def fake_svd_lowrank_simple_tensor(input: torch.Tensor, q: int, niter: int = 1):
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


def fake_divide_outliner_suboutlinear_svd(x: torch.Tensor, outliner: float, max_norm_column_list: float, scale: float, rank: int, sub_outliner_bit: int = 8, sub_outliner_ratio: float = 1.):
    is_head = len(x.shape) == 4
    if is_head:
        num_heads = x.shape[1]
        x = head_to_hidden_shape(x)
    
    # step 1: prune the outliner
    mask_1 = (x.abs() > outliner)
    x_outliner = x * mask_1
    x = x - x_outliner
    
    # step 2: prune the suboutliner
    if sub_outliner_ratio == 0.:
        x_sub_outliner = 0.
    else:
        if sub_outliner_ratio < 1.:
            mask_2 = torch.zeros_like(x).to(x.device).to(x.dtype)
            mask_2[:, :, max_norm_column_list] = 1
            mask_2 = mask_2.bool()

            x_sub_outliner = x * mask_2
            x = x - x_sub_outliner
        else:
            x_sub_outliner = x
        x_sub_outliner = torch.clamp(torch.round(x_sub_outliner / (scale + 1e-10)), min=-(2 ** (sub_outliner_bit - 1)), max=2 ** (sub_outliner_bit - 1) - 1) * scale
    
    # step 3: apply SVD
    if rank > 0:
        x = fake_svd_lowrank_simple_tensor(x, rank)
        x = x + x_outliner + x_sub_outliner
    else:
        x = x_outliner + x_sub_outliner
    
    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    return x


def true_divide_outliner_suboutlinear_svd_compress(x: torch.Tensor, outliner: float, execute_svd = False, R = None, R_inv = None, svd_rank = 16):
    is_head = len(x.shape) == 4
    if is_head:
        x = head_to_hidden_shape(x)
    
    # step 1: exectue SVD or pruning
    # L \approx X @ R^T, X_svd \approx L @ R = X @ R^T @ R
    if execute_svd: 
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        L = U[..., 0:svd_rank]
        R = torch.diag(S)[..., 0:svd_rank, :] @ Vh
        R_inv = torch.pinverse(R)
        x_svd = L @ R
    else: # use pervious base
        L = x @ R_inv
        x_svd = L @ R
        
    x_res = x - x_svd # the residual is usually more sparse...
    mask_1 = (x_res.abs() > outliner)
    x_outlier = x_res * mask_1
    
    # TODO: sparsify the x_outlier
    x_outlier_compressed = x_outlier    
    
    return x_outlier_compressed, L, R, R_inv


def true_divide_outliner_suboutlinear_svd_decompress(x_outlier_compressed, L, R, is_head = False, num_heads = 1):
    # TODO: unsparsify the x_outlier
    x_outlier = x_outlier_compressed
    x = x_outlier + L @ R

    if is_head:
        x = hidden_to_head_shape(x, num_heads=num_heads)
    
    return x


def prune_softmax(x: torch.Tensor, outliner: float):
    mask = (x > outliner)
    x_outliner = x * mask
    return x_outliner


def true_compress_softmax(x: torch.Tensor, outliner: float):
    mask = (x > outliner)
    x_outliner = x * mask
    x_outliner_sparse = x_outliner.to_sparse()
    return x_outliner_sparse


def true_decompress_softmax(x_sparse: torch.Tensor):
    return x_sparse.to_dense()
  

def get_statistics(x: torch.Tensor, outliner_ratio: float, svd_rank: int):
    if len(x.shape) == 4:
        batch, num_head, seq_len, sep_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)

    # compute the SVD
    x = x[0]
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)
    L = U[..., 0:svd_rank]
    R = torch.diag(S)[..., 0:svd_rank, :] @ Vh
    x_svd = L @ R
    
    x_residual = x - x_svd
    outliner = torch.kthvalue(x_residual.flatten(), int(x_residual.numel() * (1 - outliner_ratio))).values
    
    return outliner


def get_statistics_softmax(x: torch.Tensor, iteration: int, outliner_ratio: float):
    outliner = torch.kthvalue(x[0].flatten(), int(x[0].numel() * (1 - outliner_ratio))).values
    # print(f'iter {iteration} | outliner: {outliner}')
    return outliner
    