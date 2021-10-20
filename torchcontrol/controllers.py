import torch
from torch import nn
from warnings import warn
tanh = nn.Tanh() 

class BoxConstrainedController(nn.Module):
    """Simple controller  based on a Neural Network with
    bounded control inputs

    Args:
        in_dim: input dimension
        out_dim: output dimension
        hid_dim: hidden dimension
        zero_init: initialize last layer to zeros
    """
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 h_dim=64, 
                 num_layers=2, 
                 zero_init=True,
                 input_scaling=None, 
                 output_scaling=None,
                 constrained=False):
        
        super().__init__()
        # Create Neural Network
        layers = []
        layers.append(nn.Linear(in_dim, h_dim))
        for i in range(num_layers):
            if i < num_layers-1:
                layers.append(nn.Softplus())
            else:
                # last layer has tanh as activation function
                # which acts as a regulator
                layers.append(nn.Tanh())
                break
            layers.append(nn.Linear(h_dim, h_dim))
        layers.append(nn.Linear(h_dim, out_dim))
        self.layers = nn.Sequential(*layers)
        
        # Initialize controller with zeros in the last layer
        if zero_init: self._init_zeros()
        self.zero_init = zero_init
        
        # Scaling
        if constrained is False and output_scaling is not None:
            warn("Output scaling has no effect without the `constrained` variable set to true")
        if input_scaling is None:
            input_scaling = torch.ones(in_dim)
        if output_scaling is None:
            # scaling[:, 0] -> min value
            # scaling[:, 1] -> max value
            output_scaling = torch.cat([-torch.ones(out_dim),
                                         torch.ones(out_dim)], -1)
        self.in_scaling = input_scaling
        self.out_scaling = output_scaling
        self.constrained = constrained
        
    def forward(self, t, x):
        x = self.layers(self.in_scaling.to(x)*x)
        if self.constrained:
            # We consider the constraints between -1 and 1
            # and then we rescale them
            x = tanh(x)
            # TODO: fix the tanh to clamp
#             x = torch.clamp(x, -1, 1) # not working in some applications
            x = self._rescale(x)
        return x
    
    def _rescale(self, x):
        s = self.out_scaling.to(x)
        return 0.5*(x + 1)*(s[...,1]-s[...,0]) + s[...,0]
    
    def _reset(self):
        '''Reinitialize layers'''
        for p in self.layers.children():
            if hasattr(p, 'reset_parameters'):
                p.reset_parameters()
        if self.zero_init: self._init_zeros()

    def _init_zeros(self):
        '''Reinitialize last layer with zeros'''
        for p in self.layers[-1].parameters(): 
            nn.init.zeros_(p)
            

class RandConstController(nn.Module):
    """Constant controller
    We can use this for residual propagation and MPC steps (forward propagation)"""
    def __init__(self, shape=(1,1), u_min=-1, u_max=1):
        super().__init__()
        self.u0 = torch.Tensor(*shape).uniform_(u_min, u_max)
        
    def forward(self, t, x):
        return self.u0