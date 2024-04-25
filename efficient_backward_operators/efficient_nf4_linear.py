import torch
import bitsandbytes as bnb
import bitsandbytes.functional as F

#! notice this layer doesn't update w & b
class EfficientMemoryLinearnf4Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_quant, w_state, b, use_bias, use_8bit):
        # dequantize the weight
        w = F.dequantize_nf4(w_quant, w_state)
        if use_8bit:
            # quantize the w to 8bit
            s_w = w.abs().max()
            w = torch.round(w / s_w * 127)
            s_x = x.abs().max()
            x = torch.round(x / s_x * 127)
            output = x @ w.t() / 127 / 127 * s_x * s_w # dequantize the output
            output = output + b if use_bias else output
            ctx.scale = (s_x, s_w)
        else:
            output = x @ w.t()
            output = output + b if use_bias else output
            
        ctx.save_for_backward(x, w_quant)
        ctx.use_8bit = use_8bit
        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        x, w_quant = ctx.saved_tensors
        w = F.dequantize_nf4(w_quant, ctx.w_state)
        if ctx.use_8bit:
            s_g = grad_output.abs().max()
            grad_output = torch.round(grad_output / s_g * 127)
            _, s_w = ctx.scale
            w = torch.round(w / s_w * 127)
            grad_x = grad_output @ w
        else:
            grad_x = grad_output @ w

        return grad_x, None, None, None, None, None
        
        
class EfficientMemoryLinearnf4(bnb.nn.LinearNF4):
    def __init__(self, in_features, out_features, use_bias=True, use_8bit=False):
        super(EfficientMemoryLinearnf4, self).__init__(in_features, out_features, bias = use_bias, compute_dtype=torch.bfloat16)
        self.use_bias = use_bias
        self.use_8bit = use_8bit
    
    def forward(self, x):
        return EfficientMemoryLinearnf4Func.apply(x, self.weight.data, self.weight.quant_state, self.bias, self.use_bias, self.use_8bit)