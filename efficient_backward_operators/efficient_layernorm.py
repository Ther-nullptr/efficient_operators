import torch

import triton
import triton.language as tl
from .compress_function import (
    true_divide_outliner_suboutlinear_svd_compress,
    true_divide_outliner_suboutlinear_svd_decompress,
    get_statistics
)

HAS_APEX = False


@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


# %%
# Backward pass
# -------------
#
# The backward pass for the layer normalization operator is a bit more involved than the forward pass.
# Let :math:`\hat{x}` be the normalized inputs :math:`\frac{ x - \text{E}[x] }{ \sqrt{\text{Var}(x) + \epsilon} }` before the linear transformation,
# the Vector-Jacobian Products (VJP) :math:`\nabla_{x}` of :math:`x` are given by:
#
# .. math::
#    \nabla_{x} = \frac{1}{\sigma}\Big( \nabla_{y} \odot w - \underbrace{ \big( \frac{1}{N} \hat{x} \cdot (\nabla_{y} \odot w) \big) }_{c_1} \odot \hat{x} - \underbrace{ \frac{1}{N} \nabla_{y} \cdot w }_{c_2} \Big)
#
# where :math:`\odot` denotes the element-wise multiplication, :math:`\cdot` denotes the dot product, and :math:`\sigma` is the standard deviation.
# :math:`c_1` and :math:`c_2` are intermediate constants that improve the readability of the following implementation.
#
# For the weights :math:`w` and biases :math:`b`, the VJPs :math:`\nabla_{w}` and :math:`\nabla_{b}` are more straightforward:
#
# .. math::
#    \nabla_{w} = \nabla_{y} \odot \hat{x} \quad \text{and} \quad \nabla_{b} = \nabla_{y}
#
# Since the same weights :math:`w` and biases :math:`b` are used for all rows in the same batch, their gradients need to sum up.
# To perform this step efficiently, we use a parallel reduction strategy: each kernel instance accumulates
# partial :math:`\nabla_{w}` and :math:`\nabla_{b}` across certain rows into one of :math:`\text{GROUP_SIZE_M}` independent buffers.
# These buffers stay in the L2 cache and then are further reduced by another function to compute the actual :math:`\nabla_{w}` and :math:`\nabla_{b}`.
#
# Let the number of input rows :math:`M = 4` and :math:`\text{GROUP_SIZE_M} = 2`,
# here's a diagram of the parallel reduction strategy for :math:`\nabla_{w}` (:math:`\nabla_{b}` is omitted for brevity):
#
#   .. image:: parallel_reduction.png
#
# In Stage 1, the rows of X that have the same color share the same buffer and thus a lock is used to ensure that only one kernel instance writes to the buffer at a time.
# In Stage 2, the buffers are further reduced to compute the final :math:`\nabla_{w}` and :math:`\nabla_{b}`.
# In the following implementation, Stage 1 is implemented by the function :code:`_layer_norm_bwd_dx_fused` and Stage 2 is implemented by the function :code:`_layer_norm_bwd_dwdb`.


@triton.jit
def _layer_norm_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    Lock,  # pointer to the lock
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
    dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32)
    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    c2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _layer_norm_bwd_dwdb(
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    FINAL_DW,  # pointer to the weights gradient
    FINAL_DB,  # pointer to the biases gradient
    M,  # GROUP_SIZE_M
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
        db += tl.load(DB + offs, mask=mask, other=0.0)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


# %%
# Benchmark
# ---------
#
# We can now compare the performance of our kernel against that of PyTorch.
# Here we focus on inputs that have Less than 64KB per feature.
# Specifically, one can set :code:`'mode': 'backward'` to benchmark the backward pass.


class EfficientMemoryLayerNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        normalized_shape,
        outliner_ratio,
        sub_outliner_ratio,
        sub_outliner_bit,
        rank, 
        weight,
        bias,
        eps,
        iteration,
        static_value,
    ):
        # allocate output
        x = x.contiguous()
        y = torch.empty_like(x)
        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
        # enqueue kernel
        _layer_norm_fwd_fused[(M,)](  #
            x_arg,
            y,
            weight,
            bias,
            mean,
            rstd,  #
            x_arg.stride(0),
            N,
            eps,  #
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        # we just need to use the first batch to calculate the outliner
        if iteration < 10:
            outliner, max_norm_column_list, scale = get_statistics(x, iteration, outliner_ratio, sub_outliner_ratio, sub_outliner_bit)
            # inorder to mark save_for_backward, we should convert the tensor
            max_norm_column_list = torch.tensor(max_norm_column_list)
        else:
            outliner = static_value[0]
            max_norm_column_list = static_value[1]
            scale = static_value[2]         
            
        x_outlier_compressed, x_sub_outliner_compressed, scale = true_divide_outliner_suboutlinear_svd_compress(x, outliner, scale, sub_outliner_bit, sub_outliner_ratio)
        
        ctx.mark_non_differentiable(outliner, max_norm_column_list)
        ctx.save_for_backward(x_outlier_compressed, x_sub_outliner_compressed, scale, weight, bias, mean, rstd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        ctx.sub_outliner_bit = sub_outliner_bit
        
        y = y.contiguous()
        return y, outliner, max_norm_column_list, scale

    @staticmethod
    def backward(ctx, dy, grad_outliner, grad_max_norm_column_list, grad_scale):
        x_outlier_compressed, x_sub_outliner_compressed, scale, w, b, m, v = ctx.saved_tensors
        x = true_divide_outliner_suboutlinear_svd_decompress(x_outlier_compressed, x_sub_outliner_compressed, ctx.sub_outliner_bit, scale)
        dx, dw, db = None, None, None

        # heuristics for amount of parallel reduction stream for DW/DB
        N = w.shape[0]
        GROUP_SIZE_M = 64
        if N <= 8192:
            GROUP_SIZE_M = 96
        if N <= 4096:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256
        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
        _dw = torch.empty(
            (GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device
        )
        _db = torch.empty(
            (GROUP_SIZE_M, w.shape[0]), dtype=x.dtype, device=w.device
        )
        dw = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        db = torch.empty((w.shape[0],), dtype=w.dtype, device=w.device)
        dx = torch.empty_like(dy)
        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M,)](  #
            dx,
            dy,
            _dw,
            _db,
            x,
            w,
            b,
            m,
            v,
            locks,  #
            x_arg.stride(0),
            N,
            ctx.eps,  #
            BLOCK_SIZE_N=ctx.BLOCK_SIZE,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            num_warps=ctx.num_warps,
        )
        grid = lambda meta: [triton.cdiv(N, meta["BLOCK_SIZE_N"])]
        # accumulate partial sums in separate kernel
        _layer_norm_bwd_dwdb[grid](
            _dw,
            _db,
            dw,
            db,
            min(GROUP_SIZE_M, M),
            N,  #
            BLOCK_SIZE_M=32,  #
            BLOCK_SIZE_N=128,
        )

        return dx, None, None, None, None, None, None, None, None, None, None, None


class EfficientMemoryLayerNorm(torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        bias=True,
        outliner_ratio: float = 0.01,
        sub_outliner_ratio: float = 0.2, #! initialize
        sub_outliner_bit: int = 8,
        rank: int = 16,
    ):
        super(EfficientMemoryLayerNorm, self).__init__(
            normalized_shape, eps, elementwise_affine, bias
        )
        self.outliner_ratio = outliner_ratio
        self.sub_outliner_ratio = sub_outliner_ratio
        self.sub_outliner_bit = sub_outliner_bit
        self.rank = rank
        self.iteration = 0
        self.static_value = [None, None, None]

    def forward(self, x):
        result, outliner, max_norm_column_list, scale = EfficientMemoryLayerNormFunc.apply(
            x,
            self.normalized_shape,
            self.outliner_ratio,
            self.sub_outliner_ratio,
            self.sub_outliner_bit,
            self.rank, 
            self.weight,
            self.bias,
            self.eps,
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
