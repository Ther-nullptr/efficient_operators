import math
import torch
import bitsandbytes.functional as F
from gact.dct_processor import DCTProcessor
from gact.jpeg_processor import JPEGProcessor
from gact.memory_efficient_function import per_block_quantization, per_block_dequantization, dct_compression, jpeg_compression, naive_adjustment

class EfficientMemoryHadamardFunc(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x1, x2, compress_type, jpeg_processor, dct_processor, quantization_shape):
    result = x1 * x2
    ctx.needs_inputs_grad = [x1.requires_grad, x2.requires_grad]
    ctx.compress_type = compress_type
    ctx.quantization_shape = quantization_shape

    if compress_type == 'NF4':
      # quantize the cached activation
      x1, quant_state_1 = F.quantize_nf4(x1)
      x2, quant_state_2 = F.quantize_nf4(x2)
      ctx.quant_state = quant_state_1, quant_state_2
    elif compress_type != 'NONE':
      # shape preparation for DCT
      input_shape = [x1.shape, x2.shape]
      ctx.input_shape = input_shape
      # quantize the cached activation
      x1, quant_state1 = per_block_quantization(x1.contiguous(), input_shape[0])
      x2, quant_state2 = per_block_quantization(x2.contiguous(), input_shape[1])
      ctx.quant_state = [quant_state1, quant_state2]

      # compress the cached activation
      if compress_type == 'JPEG':
        x1 = jpeg_compression(x1, input_shape[0], jpeg_processor)
        x2 = jpeg_compression(x2, input_shape[1], jpeg_processor)

      elif compress_type == 'DCT':
        x1 = dct_compression(x1, input_shape[0], dct_processor)
        x2 = dct_compression(x2, input_shape[1], dct_processor)

      elif compress_type == 'NAIVE':
        x1 = naive_adjustment(x1, input_shape[0])
        x2 = naive_adjustment(x2, input_shape[1])

    ctx.save_for_backward(x1, x2)
    return result
  

  def backward(ctx, grad_output):
    x1, x2 = ctx.saved_tensors
    quantization_shape = ctx.quantization_shape
    grad_input1, grad_input2 = None, None

    if ctx.needs_inputs_grad[0] or ctx.needs_inputs_grad[1]:
      if ctx.compress_type == 'NF4':
        x1 = F.dequantize_nf4(x1, ctx.quant_state[0])
        x2 = F.dequantize_nf4(x2, ctx.quant_state[1])
      elif ctx.compress_type != 'NONE':
        quant_state1, quant_state2 = ctx.quant_state
        input_shape1, input_shape2 = ctx.input_shape
        x1 = per_block_dequantization(x1, input_shape1, quant_state1, quantization_shape)
        x2 = per_block_dequantization(x2, input_shape2, quant_state2, quantization_shape)
      
      if ctx.needs_inputs_grad[0]:
        grad_input1 = grad_output * x2
      if ctx.needs_inputs_grad[1]:
        grad_input2 = grad_output * x1

    return grad_input1, grad_input2, None, None, None, None
  

class EfficientMemoryHadamard(torch.nn.Module):
  def __init__(self, compress_type: str = "JPEG", compress_quality: int = 50, quantization_shape: int = 64):
    super().__init__()
    self.compress_type = compress_type
    self.compress_quality = compress_quality
    self.jpeg_processor = JPEGProcessor(quality=compress_quality)
    self.dct_processor = DCTProcessor(quality=compress_quality)
    self.quantization_shape = quantization_shape

  def forward(self, x1, x2):
    if self.extract_mode:
        torch.save(x1, f"output/{self.name}_1.pt")
        torch.save(x2, f"output/{self.name}_2.pt")

    return EfficientMemoryHadamardFunc.apply(
      x1, 
      x2,
      self.compress_type,
      self.jpeg_processor,
      self.dct_processor,
      self.quantization_shape
    )