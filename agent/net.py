import torch
from torch import nn
import typing

from agent.OBC import outputBoundConfig_mp as outputBoundConfig

class fcNet(nn.Module):
    def __init__(self, 
                 n_feature:int, 
                 n_output:int,
                 n_hiddens:typing.List[int], 
                 ):
        super().__init__()
        self.n_feature = n_feature
        self.n_output = n_output
        self.__dummy_param = nn.Parameter(torch.empty(0))
        
        fc = []
        for i in range(len(n_hiddens)):
            fc.append(nn.Linear(n_feature if i == 0 else n_hiddens[i-1], n_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(n_hiddens[-1], n_output))
        self.fc_layers = nn.Sequential(*fc)

    @property
    def device(self):
        return self.__dummy_param.device
    
    def forward(self, x):
        fc_out = self.fc_layers(x)
        return self.post_process(fc_out)

    def post_process(self, x):
        '''
            to be overloaded.
        '''
        return x
    
    def loss(self, x, target, **kwargs):
        '''
            to be overloaded.
        '''
        raise(NotImplementedError)
    
    def sample(self, output):
        '''
            to be overloaded when network output randomly.
        '''
        return output
    
    def distribution(self, output):
        '''
            to be overloaded when network output randomly.
        '''
        raise(NotImplementedError)
    
    def nominal_output(self, x, require_grad=True):
        '''
            determined output, e.g. mean of distribution.
            To be overloaded when network output randomly.
        '''
        raise(NotImplementedError)


class boundedFcNet(fcNet):
    def __init__(self,
                 n_feature:int,
                 n_output:int,
                 n_hiddens:typing.List[int],
                 upper_bounds:torch.Tensor,
                 lower_bounds:torch.Tensor,
                 ):
        super().__init__(n_feature, n_output, n_hiddens)
        self.obc = outputBoundConfig(upper_bounds, lower_bounds)
        '''
            see `outputBoundConfig`.
        '''

    def to(self, device:str, **kwargs):
        self.obc.to(device)
        return super().to(device=device, **kwargs)

    def post_process(self, x):
        return self.obc(x)
    
    def clip(self, x:torch.Tensor):
        '''
            clip output to output bounds.
        '''
        x = x.clamp(self.obc.lower_bounds, self.obc.upper_bounds)
        return x
    
class normalDistNet(boundedFcNet):
    def __init__(self, 
                 n_feature: int, 
                 n_sample: int, 
                 n_hiddens: typing.List[int], 
                 upper_bounds: torch.Tensor, 
                 lower_bounds: torch.Tensor
                 ):
        super().__init__(n_feature, 2*n_sample, n_hiddens, upper_bounds, lower_bounds)
        self.n_sample = n_sample

    def sample(self, output:torch.Tensor):
        '''
            sample from normal dist, output are mean and var.
        '''
        sample = self.distribution(output).sample()
        return self.clip(sample)
    
    def distribution(self, output):
        dist = torch.distributions.Normal(output[:,:self.n_sample], output[:,self.n_sample:])
        return dist
    
    def nominal_output(self, x, require_grad=True):
        '''
            mean of normal dist.
            To be overloaded when network output randomly.
        '''
        output = self.forward(x)
        nominal = output[:,:self.n_sample]
        if not require_grad:
            nominal = nominal.detach()
        return nominal
    
    def clip(self, x:torch.Tensor):
        '''
            clip output to output bounds.
        '''
        x = x.clamp(self.obc.lower_bounds[:self.n_sample], self.obc.upper_bounds[:self.n_sample])
        return x