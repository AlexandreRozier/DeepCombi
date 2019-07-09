import torch
import numpy as np


class ExSequential(torch.nn.Sequential):
    layers = None
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.layers = list(self)

    def relprop(self,A, R):
        # Populate activations through forward pass
        for i,l in enumerate(self.layers):
            if i == 0:
                l.input = A
            else:
                l.input = self.layers[i-1].forward(self.layers[i-1].input)
        
        # RELPROP BABY
        self.layers[0].weight = self.layers[0].weight**2
        self.layers[0].input = torch.ones_like(A)
        
        for l in reversed(self.layers):                
            R = l.relprop(R)

        return R    

class ExReLU(torch.nn.ReLU):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    
    def relprop(self, R):
        return R



class ExDropout(torch.nn.Dropout):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    
    def relprop(self, R):
        return R

class ExConv1d(torch.nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    
    # LRP-plus prop rule
    def relprop(self, R):
        zeros = torch.zeros_like(self.weight)
        self.weight = torch.max(zeros, self.weight)
        self.bias = torch.zeros_like(self.bias)
        A = torch.tensor(self.input, require_grad=True)
        Z = self.forward(A) + 1e-9
        (Z*(R/Z).data).sum().backward()
        R = A*A.grad
        return R




class ExLinear(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
    
    def relprop(self, R):
        prev_activ = self.input
        zeros = torch.zeros_like(self.weight)
        W_plus = torch.max(zeros , self.weight) # w+ 
        Z = torch.mm(prev_activ, torch.t(W_plus))+1e-9  # (batch, n_in ) * ( batch, n_in, n_out ) 
        S = torch.div(R,Z) # elt wise, yields (batch, out)
        C = torch.mm(S, W_plus)  #(batch, out) * (batch, out, in) = (batch, in)
        R = torch.mul(C, prev_activ)
        return R

