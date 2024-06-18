
import torch
import numpy as np 

from . import utils

from torch import nn


class CustomModule(nn.Module):
    """A simple two layer type I MLP structure.
    """
    def __init__(self, w1_weight=None, w2_bias=None, w2_weight=None, act='gelu'):
        super().__init__()

        self.linear1 = nn.Linear(w1_weight.shape[1], w1_weight.shape[0])
        self.linear2 = nn.Linear(w1_weight.shape[0], w1_weight.shape[1])
        self.act = utils.load_activation(act)

        self.linear1.weight = nn.Parameter(w1_weight.float())
        self.linear1.bias = nn.Parameter(w2_bias.float())
        self.linear2.weight = nn.Parameter(w2_weight.T.float())
        self.linear2.bias = nn.Parameter(torch.zeros_like(self.linear2.bias))

    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class CustomNormModule(nn.Module):
    """A simple two layer type I MLP structure.
    """
    def __init__(self, 
            w1_weight=None, 
            w1_bias = None,
            w2_weight=None, 
            centroid=None, 
            norm_weight=None, 
            norm_bias=None,
            add_norm = True, 
            return_w1 = False,
            act='relu'
        ):
        super().__init__()

        self.linear1 = nn.Linear(w1_weight.shape[1], w1_weight.shape[0])
        self.linear2 = nn.Linear(w1_weight.shape[0], w1_weight.shape[1])
        self.act = utils.load_activation(act)

        self.centroid = centroid
        self.norm_weight = norm_weight
        self.norm_bias = norm_bias
        if self.norm_bias is None: self.norm_bias = 0
        self.add_norm = add_norm

        self.return_w1 = return_w1

        self.linear1.weight = nn.Parameter(w1_weight)
        if w1_bias is not None: self.linear1.bias = nn.Parameter(w1_bias)
        self.linear2.weight = nn.Parameter(w2_weight.T)
        self.linear2.bias = nn.Parameter(torch.zeros_like(self.linear2.bias).to(w1_weight.dtype).cuda())

    def forward(self, x):

        # normalisation (part I)
        x = (x - self.norm_bias) / self.norm_weight / np.sqrt(self.centroid.shape[0])

        x = x - self.centroid

        if self.add_norm:
            x = x / torch.norm(x, dim=-1)[:,:,None]

        w1_output = self.act(self.linear1(x))

        if self.return_w1:
            return w1_output

        w2_output = self.linear2(w1_output)
        return w2_output


class ModifiedMLP(nn.Module):
    """Modifed MLP structure
    """
    def __init__(self, original_mlp, custom_module):
        super(ModifiedMLP, self).__init__()
        self.original_mlp = original_mlp
        self.custom_module = custom_module
    
    def forward(self, x):
        # Get the output from the original MLP
        o = self.original_mlp(x)
        # Pass the output through the CustomModule
        return o + self.custom_module(x)


class ModifieMambadMLP(nn.Module):
    """Modifed MLP structure
    """
    def __init__(self, original_mlp, custom_module):
        super(ModifieMambadMLP, self).__init__()
        self.original_mlp = original_mlp
        self.custom_module = custom_module
    
    def forward(self, x, cache_params=None):
        # Get the output from the original MLP
        o = self.original_mlp(x, cache_params=cache_params)
        # Pass the output through the CustomModule
        return o + self.custom_module(x)


