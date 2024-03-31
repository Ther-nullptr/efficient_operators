import math
import torch
import torch.nn.functional as F
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor
from gact.memory_efficient_function import per_block_quantization, per_block_dequantization, dct_compression, jpeg_compression, naive_adjustment

class EfficientMemoryGELUFunc(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, compress_type, jpeg_processor, dct_processor, quantization_shape):
    result = F.gelu(x)
    ctx.needs_inputs_grad = x.requires_grad
    ctx.compress_type = compress_type
    ctx.quantization_shape = quantization_shape

    if compress_type == 'NF4':
        # quantize the cached activation
        x, quant_state = F.quantize_nf4(x)
        ctx.quant_state = quant_state
    elif compress_type != 'NONE':
      input_shape = x.shape
      ctx.input_shape = input_shape

      x, quant_state = per_block_quantization(x, input_shape, quantization_shape)
      ctx.quant_state = quant_state

      if compress_type == 'JPEG':
          x = jpeg_compression(x, input_shape, jpeg_processor, quantization_shape)

      elif compress_type == 'DCT':
          x = dct_compression(x, input_shape, dct_processor, quantization_shape)

      elif compress_type == 'NAIVE':
          x = naive_adjustment(x, input_shape, quantization_shape)

    ctx.save_for_backward(x)
    return result

  @staticmethod
  def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    quantization_shape = ctx.quantization_shape

    gamma = math.sqrt(2 / math.pi)
    kappa = 0.044715
    grad_input = None

    if ctx.needs_inputs_grad:
      if ctx.compress_type == 'NF4':
        x = F.dequantize_nf4(x, ctx.quant_state)
      elif ctx.compress_type != 'NONE':
        quant_state = ctx.quant_state
        input_shape = ctx.input_shape
        x = per_block_dequantization(x, input_shape, quant_state, quantization_shape)
      y = gamma * (x + kappa * x ** 3)
      tanh_y = F.tanh(y)
      grad_input = 0.5 * ( (1 + tanh_y) + x * ( (1 - tanh_y ** 2) * gamma * (1 + 3 * kappa * x ** 2) ) ) * grad_output

    return grad_input, None, None, None, None
  

class EfficientMemoryGELU(torch.nn.Module):
  def __init__(self, compress_type: str = "JPEG", compress_quality: int = 50, quantization_shape: int = 64):
    super(EfficientMemoryGELU, self).__init__()
    self.compress_type = compress_type
    self.compress_quality = compress_quality
    self.jpeg_processor = JPEGProcessor(quality=compress_quality)
    self.dct_processor = DCTProcessor(quality=compress_quality, interpolation=quantization_shape / 64)
    self.quantization_shape = quantization_shape

  def forward(self, input):
    return EfficientMemoryGELUFunc.apply(
      input,
      self.compress_type,
      self.jpeg_processor,
      self.dct_processor,
      self.quantization_shape
    )