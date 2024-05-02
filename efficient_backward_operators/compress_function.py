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


def true_divide_outliner_suboutlinear_svd_compress(x: torch.Tensor, outliner: float, scale: float, sub_outliner_bit: int = 8, sub_outliner_ratio: float = 1.):
    is_head = len(x.shape) == 4
    if is_head:
        num_heads = x.shape[1]
        x = head_to_hidden_shape(x)
    
    # step 1: prune the outliner
    mask_1 = (x.abs() > outliner)
    x_outliner = x * mask_1
    x = x - x_outliner
    # compress the x_outlier
    x_outlier_compressed = x_outlier_compressed.to_sparse() # coo
    
    # step 2: prune the suboutliner
    if sub_outliner_ratio == 0.:
        x_sub_outliner = 0.
    else:
        x_sub_outliner = x
        assert (sub_outliner_bit % 2 == 0) or (sub_outliner_bit == 1), "Only support 2,4,8 bit quantization"
        x_sub_outliner = torch.clamp(torch.round(x_sub_outliner / (scale + 1e-10)), min=-(2 ** (sub_outliner_bit - 1)), max=2 ** (sub_outliner_bit - 1) - 1)
        # now the x_sub_outlier is int type, then we can use bit squeeze method
        # since the shape is [bs, seq_len, hidden_dim], and hidden_dim is usually divisble by 8, so use hidden_dim dim to squeeze
        hidden_dim = x_sub_outliner.shape[-1]
        
        if sub_outliner_bit == 8:
            x_sub_outliner_compressed = x_sub_outliner
        elif sub_outliner_bit == 4:
            # shift to positive
            x_sub_outliner = x_sub_outliner.to(torch.uint8) + 8
            x_sub_outliner_compressed = x_sub_outliner[..., 0:hidden_dim // 2] + torch.bitwise_left_shift(x_sub_outliner[..., hidden_dim // 2:], 4)
        elif sub_outliner_bit == 2:
            x_sub_outliner = x_sub_outliner.to(torch.uint8) + 2
            x_sub_outliner_compressed = x_sub_outliner[..., 0:hidden_dim // 4] + torch.bitwise_left_shift(x_sub_outliner[..., hidden_dim // 4:hidden_dim // 2] + torch.bitwise_left_shift(x_sub_outliner[..., hidden_dim // 2:hidden_dim // 4 * 3] + torch.bitwise_left_shift(x_sub_outliner[..., hidden_dim // 4 * 3:hidden_dim], 2), 2), 2)
    
    return x_outlier_compressed, x_sub_outliner_compressed, scale


def true_divide_outliner_suboutlinear_svd_decompress(x_outlier_compressed, x_sub_outliner_compressed, sub_outliner_bit, scale):
    # step 1: decompress the outliers
    x_outlier = x_outlier_compressed.to_dense()
   
    # step 2: decompress the sub_outliners
    if sub_outliner_bit == 8:
        # just return to the original value
        x_sub_outliner = x_sub_outliner_compressed * scale
    elif sub_outliner_bit == 4:
        x_sub_outliner_1st = x_sub_outliner_compressed % (2 ** 4)
        x_sub_outliner_2nd = (x_sub_outliner_compressed - x_sub_outliner_1st) // (2 ** 4)
        x_sub_outliner = torch.cat((x_sub_outliner_1st, x_sub_outliner_2nd), dim=-1)
        x_sub_outliner = ((x_sub_outliner - 8).to(x_outlier.dtype)) * scale
    elif sub_outliner_bit == 2:
        x_sub_outliner_1st = x_sub_outliner_compressed % (2 ** 2)
        x_sub_outliner_compressed = (x_sub_outliner_compressed - x_sub_outliner_1st) // (2 ** 2)
        x_sub_outliner_2nd = x_sub_outliner_compressed % (2 ** 2)
        x_sub_outliner_compressed = (x_sub_outliner_compressed - x_sub_outliner_2nd) // (2 ** 2)
        x_sub_outliner_3rd = x_sub_outliner_compressed % (2 ** 2)
        x_sub_outliner_4th = (x_sub_outliner_compressed - x_sub_outliner_3rd) // (2 ** 2)
        x_sub_outliner = torch.cat((x_sub_outliner_1st, x_sub_outliner_2nd, x_sub_outliner_3rd, x_sub_outliner_4th), dim=-1)
        x_sub_outliner = ((x_sub_outliner - 2).to(x_outlier.dtype)) * scale
        
    return x_outlier + x_sub_outliner

def prune_softmax(x: torch.Tensor, outliner: float):
    mask = (x > outliner)
    x_outliner = x * mask
    return x_outliner


def get_statistics(x: torch.Tensor, iteration: int, outliner_ratio: float, sub_outliner_ratio: float, sub_outliner_bit: int = 8):
    outliner = torch.kthvalue(x[0].flatten(), int(x[0].numel() * (1 - outliner_ratio))).values
    print(f'iter {iteration} | outliner: {outliner}')
    
    if len(x.shape) == 4:
        batch, num_head, seq_len, sep_dim = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch, seq_len, num_head * sep_dim)

    mean_norm = list(torch.mean(torch.abs(x[0]), dim=-2).cpu().detach().numpy())
    max_norm_column_list = sorted(enumerate(mean_norm), key=lambda x: x[1], reverse=True)
    max_norm_column_list = [item[0] for item in max_norm_column_list]
    max_norm_column_list = sorted(max_norm_column_list[:int(len(max_norm_column_list) * sub_outliner_ratio)])

    x_outliner = x[0] * (x[0].abs() > outliner)
    x = x - x_outliner
    if sub_outliner_ratio > 0:
        x_sub_outliner = x[0][:, max_norm_column_list]
        # TODO: set the scale factor to per channel or per tensor?
        scale = (x_sub_outliner.max() - x_sub_outliner.min()) / (2 ** sub_outliner_bit)
    else:
        scale = 0
    return outliner, max_norm_column_list, scale


def get_statistics_softmax(x: torch.Tensor, iteration: int, outliner_ratio: float):
    outliner = torch.kthvalue(x[0].flatten(), int(x[0].numel() * (1 - outliner_ratio))).values
    print(f'iter {iteration} | outliner: {outliner}')
    return outliner
    