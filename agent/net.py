import torch
from torch import nn
import typing

from utils import affine

class outputBoundConfig:
    def __init__(self, 
                 upper_bounds:torch.Tensor, 
                 lower_bounds:torch.Tensor
                ) -> None:
        if type(upper_bounds) is not torch.Tensor:
            upper_bounds = torch.tensor(upper_bounds)
        if type(lower_bounds) is not torch.Tensor:
            lower_bounds = torch.tensor(lower_bounds)
        upper_bounds = upper_bounds.flatten().to(torch.float32)
        lower_bounds = lower_bounds.flatten().to(torch.float32)
        if upper_bounds.shape[0]!=lower_bounds.shape[0]:
            raise(ValueError("upper_bounds' and lower_bounds' shape incompatible."))
        self.n_output = upper_bounds.shape[0]
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.activations = []
        for i in range(self.n_output):
            if torch.isfinite(self.upper_bounds[i]) and torch.isfinite(self.lower_bounds[i]):
                self.activations.append(lambda x:affine(torch.tanh(x), -1, 1, self.upper_bounds[i].item(), self.lower_bounds[i].item()))
            elif torch.isfinite(self.upper_bounds[i]) and (not torch.isfinite(self.lower_bounds[i])):
                self.activations.append(lambda x:(-torch.relu(x)+self.upper_bounds[i].item()))
            elif (not torch.isfinite(self.upper_bounds[i])) and torch.isfinite(self.lower_bounds[i]):
                self.activations.append(lambda x:( torch.relu(x)+self.lower_bounds[i].item()))
            else:
                self.activations.append(lambda x:x)

    def __call__(self, x:torch.Tensor):
        y = torch.zeros_like(x)
        for i in range(self.n_output):
            y[:,i] = self.activations[i](x[:,i])
        return y
    
    def uniSample(self, size:int):
        '''
            sample uniformally between output bounds.
        '''
        if self.upper_bounds.isinf().any() or self.lower_bounds.isinf().any():
            raise(ValueError("output bounds are not finite."))
        return torch.rand(size, self.n_output)*torch.abs(self.upper_bounds-self.lower_bounds)+self.lower_bounds

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

    def post_process(self, x):
        return self.obc(x)
    
class normalNet(boundedFcNet):
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
        mean = output[:,:self.n_sample]
        var = output[:,self.n_sample:]
        dist = torch.distributions.Normal(mean, var)
        return dist.sample()