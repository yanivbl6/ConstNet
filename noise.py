import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class noiseF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, stdin, stdback ):
        ctx.save_for_backward(stdback)
        return input + stdin
      
    @staticmethod
    def backward(ctx, grad_output):
        stdback, = ctx.saved_tensors
        return (grad_output + stdback), None, None, None, None

class Noise(nn.Module):
    __constants__ = ['stdin','stdback']
    def __init__(self,shape, sigma_forward = 0.0, sigma_backward =0.0): 
        super(Noise, self).__init__()
        
        self.sigma1 = sigma_forward
        self.sigma2 = sigma_backward

        self.noise1 = Variable(torch.zeros(shape,dtype=torch.float32).cuda())
        self.noise2 = Variable(torch.zeros(shape,dtype=torch.float32).cuda())
    def forward(self,x):
        
        if self.sigma1 > 0.0 or self.sigma2 > 0.0:
            if self.sigma1 > 0.0:
                self.noise1.data.normal_(0, std=self.sigma1)
            if self.sigma2 > 0.0:
                self.noise2.data.normal_(0, std=self.sigma2)
            
            return noiseF.apply(x,self.noise1,self.noise2)
        else:
            return x

