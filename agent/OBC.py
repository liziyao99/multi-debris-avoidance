'''
    output bound config, defining classes that map networks' output to a bounded set.
    NOTICE: lambda function is unavailable in subprocess, so do `outputBoundConfig`. Please use `outputBoundConfig_mp`.
'''
import torch
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
                self.activations.append(lambda x:affine(torch.tanh(x), -1, 1, self.lower_bounds[i].item(), self.upper_bounds[i].item()))
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
        return torch.rand(size, self.n_output).to(self.device)*torch.abs(self.upper_bounds-self.lower_bounds)+self.lower_bounds

    def to(self, device:str):
        self.upper_bounds = self.upper_bounds.to(device)
        self.lower_bounds = self.lower_bounds.to(device)
        return self
    
    @property
    def max_norm(self):
        bounds = torch.vstack((self.upper_bounds, self.lower_bounds))
        bounds = torch.abs(bounds)
        bigger = torch.max(bounds, dim=0)[0]
        norm = torch.norm(bigger)
        return norm.item()

    @property
    def device(self):
        return self.upper_bounds.device
    

class outputBoundConfig_mp(outputBoundConfig):
    '''
        python can't pickle lambda hence `outputBoundConfig` is not available for multiprocessing. 
        `outputBoundConfig_mp` is a wrapper of `outputBoundConfig` that can be pickled.
    '''
    def __init__(self, upper_bounds: torch.Tensor, lower_bounds: torch.Tensor) -> None:
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
        self.activation_types = []
        for i in range(self.n_output):
            if torch.isfinite(self.upper_bounds[i]) and torch.isfinite(self.lower_bounds[i]):
                self.activation_types.append(0)
            elif torch.isfinite(self.upper_bounds[i]) and (not torch.isfinite(self.lower_bounds[i])):
                self.activation_types.append(1)
            elif (not torch.isfinite(self.upper_bounds[i])) and torch.isfinite(self.lower_bounds[i]):
                self.activation_types.append(2)
            else:
                self.activation_types.append(3)

    def activate(self, x:torch.Tensor, type:int, idx:int):
        if type==0: # both side
            # span = self.upper_bounds[idx].item()-self.lower_bounds[idx].item()
            return affine(torch.tanh(x), -1, 1, self.lower_bounds[idx].item(), self.upper_bounds[idx].item())
        elif type==1: # only upper
            return (-torch.relu(x)+self.upper_bounds[idx].item())
        elif type==2: # only lower
            return ( torch.relu(x)+self.lower_bounds[idx].item())
        else: # no bound
            return x
        
    def __call__(self, x:torch.Tensor):
        y = torch.zeros_like(x)
        for i in range(self.n_output):
            y[:,i] = self.activate(x[:,i], type=self.activation_types[i], idx=i)
        return y