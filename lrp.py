import torch
import numpy as np

import torch.nn as nn
import torchvision
import torchvision.models as models
import vggutils

def DTD(x,cl,gamma=None,epsilon=None,net='vgg16'):

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
    std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1)

    lbound = (0-mean) / std
    x      = (x-mean) / std
    hbound = (1-mean) / std

    model = {'vgg16':models.vgg16,'vgg16_bn1':models.vgg16_bn}[net](pretrained=True)

    model.eval()

    layers = list(model._modules['features']) + vggutils.toconv(list(model._modules['classifier']))

    X     = [x.data]+[None for l in layers]


    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------
    for i,layer in enumerate(layers): X[i+1] = layer.forward(X[i]).data
    y = X[-1]

    t = torch.FloatTensor((1.0*(np.arange(1000)==cl).reshape([1,1000,1,1])))
    r = (y*t).data

    # ---------------------------------------------------------
    # Backward pass
    # ---------------------------------------------------------
    for i,layer in list(enumerate(layers))[::-1]:

        print(i)

        x = torch.tensor(X[i],requires_grad=True)

        if isinstance(layer,nn.MaxPool2d):
            layer = nn.AvgPool2d(2)

        if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.AvgPool2d):
            
            # Various prop rules according to the nature of the model
            if i>0:
                # Gamma rule
                z = vggutils.newlayer(layer,lambda p: p + gamma*p.clamp(min=0)).forward(x)
                ea = ((z.clamp(min=0)**2).mean()**.5).data
                z = z + epsilon*ea + 1e-9
                (z*(r/z).data).sum().backward()
                r = (x*x.grad).data
                r = r / r.std()

            else:
                l = torch.tensor(x.data*0+lbound,requires_grad=True)
                h = torch.tensor(x.data*0+hbound,requires_grad=True)
                z = layer.forward(x)-vggutils.newlayer(layer,lambda p: p.clamp(min=0)).forward(l)-vggutils.newlayer(layer,lambda p: p.clamp(max=0)).forward(h)
                (z*(r/(z+1e-6)).data).sum().backward()
                r = (x*x.grad+l*l.grad+h*h.grad).data

    return r,y













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

    
