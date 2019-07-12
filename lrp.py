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



### PYTORCH MODELS
def create_montaez_pytorch_model(params):
    
    return ExSequential(
        ExLinear(n_total_snps * 3, 10, bias=False),
        ExReLU(),
        ExDropout(p=params['dropout_rate']),
        ExLinear(10, 10, bias=False),
        ExReLU(),
        ExDropout(p=params['dropout_rate']),
        ExLinear(10, 2, bias=False),
        ExReLU(),
    )
def create_dummy_conv_pytorch_model():
    return ExSequential(
        ExConv1d(1 , 10, 100, bias=False),
        ExReLU(),
        
        ExLinear(10, 2, bias=False),
        ExReLU(),
    )

def create_dummy_pytorch_linear():
    return ExSequential(
        ExLinear(n_total_snps * 3, 2, bias=False),
        ExReLU(),
    )


class ExConvNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self.conv1 = ExConv1d(1, 10, kernel_size=100, bias=False) #(10, (n_total_snps*3 - 99))
        self.lin1 = ExLinear(((n_total_snps*3 - 99)) * self.conv1.out_channels, 2, bias=False)


    def forward(self, x):
        x =self.conv1(x)
        x = F.relu(x)
  
        x = x.view(-1, ((n_total_snps*3 - 99)) * 10)
        x = self.lin1(x)
        x = F.relu(x)
        return x

    def relprop(self,A, R):
        # Populate inputs
        self.conv1.input = A
        self.lin1.input = self.conv1(self.conv1.input)

        # Z rule
        self.conv1.weight.data = self.conv1.weight.data.pow(2)
        #self.conv1.bias.data = torch.ones_like(self.conv1.bias.data)
        self.conv1.input.data = torch.ones_like(self.conv1.input.data)

        # LRP !
        R = self.lin1.relprop(R)
        R = R.view(-1, self.conv1.out_channels, n_total_snps*3 - 99) 
        R = self.conv1.relprop(R)
        return R

    
        