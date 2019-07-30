import copy
import numpy
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# NewLayer
# -----------------------------------------------------------------------------
#
# new layer where parameters have passed element-wise through the function g.
# -----------------------------------------------------------------------------

def newlayer(layer,g):

    layer = copy.deepcopy(layer)

    try: layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError: pass

    try: layer.bias   = nn.Parameter(g(layer.bias))
    except AttributeError: pass

    return layer

# -----------------------------------------------------------------------------
# ToConv
# -----------------------------------------------------------------------------
#
# convert VGG dense layers to convolutional layers
# -----------------------------------------------------------------------------

def toconv(dlayers):

    clayers = []

    for i,dlayer in enumerate(dlayers):

        if isinstance(dlayer,nn.Linear):

            clayer = None

            if i == 0:
                
                m,n = 512,dlayer.weight.shape[0]
                clayer = nn.Conv2d(m,n,7)

                clayer.weight = nn.Parameter(dlayer.weight.reshape(n,m,7,7))

            else:
                m,n = dlayer.weight.shape[1],dlayer.weight.shape[0]
                clayer = nn.Conv2d(m,n,1)
                clayer.weight = nn.Parameter(dlayer.weight.reshape(n,m,1,1))

            clayer.bias = nn.Parameter(dlayer.bias)

            clayers += [clayer]

        else:

            clayers += [dlayer]

    return clayers

# -----------------------------------------------------------------------------
# RemoveBN
# -----------------------------------------------------------------------------
#
# fuse batch norm into the lower linear layer
# -----------------------------------------------------------------------------
def removebn(layers):

    for i in numpy.arange(len(layers))[::-1]:

        if isinstance(layers[i],nn.BatchNorm2d):

            v = layers[i].running_var; s = ((v+layers[i].eps)**.5)
            w = layers[i].weight
            b = layers[i].bias
            m = layers[i].running_mean

            layers[i-1].weight = nn.Parameter(layers[i-1].weight*(w/s).reshape(-1,1,1,1))
            layers[i-1].bias = nn.Parameter((layers[i-1].bias - m)*(w/s) + b)
            layers[i] = None

    return [layer for layer in layers if layer is not None]

