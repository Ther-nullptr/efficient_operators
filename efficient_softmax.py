import torch

class EfficientMemorySoftmaxFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = torch.softmax(x, dim=-1)
        ctx.save_for_backward(y)
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        return (grad_output - (grad_output * y).sum(dim=-1, keepdims=True)) * y
    

class EfficientMemorySoftmax(torch.nn.Module):
    def __init__(self):
        super(EfficientMemorySoftmax, self).__init__()
    
    def forward(self, x):
        return EfficientMemorySoftmaxFunc.apply(x)
