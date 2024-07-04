import torch
from torch import nn
import typing
from agent.baseModule import baseModule

from agent.OBC import outputBoundConfig_mp as outputBoundConfig

class polynomialLinearFeedback(baseModule):
    def __init__(self, 
                 n_feature:int, 
                 n_output:int,
                 order:int=1, 
                 upper_bounds:torch.Tensor=None,
                 lower_bounds:torch.Tensor=None,
                 ):
        super().__init__()
        self.n_feature = n_feature
        self.n_output = n_output
        self.order = order

        if upper_bounds is None:
            upper_bounds = torch.tensor([ torch.inf]*n_output)
        if lower_bounds is None:
            lower_bounds = torch.tensor([-torch.inf]*n_output)

        self.K = nn.Parameter(torch.randn(n_output, (n_feature*order)))
        self.obc = outputBoundConfig(upper_bounds, lower_bounds)

    def to(self, device:str, **kwargs):
        self.obc.to(device)
        return super().to(device=device, **kwargs)
    
    def forward(self, x:torch.Tensor):
        '''
            args:
                `x`: shape (batch_size, n_feature*order)
        '''
        shape = list(x.shape)
        shape[-1] *= self.order
        poly = torch.zeros(shape, device=x.device) # shape (batch_size, n_feature*order)
        for i in range(self.order):
            poly[:, i*self.n_feature:(i+1)*self.n_feature] = x**(i+1) # shape (batch_size, n_feature)
        u = poly@self.K.T
        u = self.obc(u)
        return u
        
        