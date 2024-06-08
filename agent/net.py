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
        self._dummy_param = nn.Parameter(torch.empty(0))
        
        fc = []
        for i in range(len(n_hiddens)):
            fc.append(nn.Linear(n_feature if i == 0 else n_hiddens[i-1], n_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(n_hiddens[-1], n_output))
        self.fc_layers = nn.Sequential(*fc)

    @property
    def device(self):
        return self._dummy_param.device
    
    def forward(self, x):
        fc_out = self.fc_layers(x)
        return self.post_process(fc_out)

    def post_process(self, x):
        '''
            will be called after fc layers.
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
            see `outputBoundConfig_mp`.
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
    
    def uniSample(self, size:int):
        '''
            sample uniformly between output bounds.
        '''
        return self.obc.uniSample(size)
    
class boundedLSTM(boundedFcNet):
    def __init__(self, 
                 n_feature: int, 
                 n_output: int, 
                 n_lstm_hidden: int, 
                 n_lstm_layer: int,
                 fc_hiddens: typing.List[int], 
                 upper_bounds: torch.Tensor, 
                 lower_bounds: torch.Tensor,
                 batch_first=False,
                ):
        if len(fc_hiddens)==0:
            n_output = n_lstm_hidden
    
        nn.Module.__init__(self)
        self.n_feature = n_feature
        self.n_output = n_output
        self._dummy_param = nn.Parameter(torch.empty(0))

        self.n_feature = n_feature
        self.n_lstm_hidden = n_lstm_hidden
        self.n_lstm_layer = n_lstm_layer
        self.lstm = nn.LSTM(n_feature, n_lstm_hidden, n_lstm_layer, batch_first=batch_first)
        
        fc = []
        if len(fc_hiddens) > 0:
            for i in range(len(fc_hiddens)):
                fc.append(nn.Linear(n_lstm_hidden if i == 0 else fc_hiddens[i-1], fc_hiddens[i]))
                fc.append(nn.ReLU())
            fc.append(nn.Linear(fc_hiddens[-1], n_output))
        self.fc_layers = nn.Sequential(*fc)

        self.obc = outputBoundConfig(upper_bounds, lower_bounds)

    def forward(self, x, h0c0:typing.Tuple[torch.Tensor]=None):
        '''
            `x` shape: (batch, seq, feature)
            `h0` shape: (n_lstm_layer, batch, n_hidden)
            `c0` shape: (n_lstm_layer, batch, n_hidden)
        '''
        lstm_out, (hn, cn) = self.lstm(x, h0c0)
        fc_out = self.fc_layers(lstm_out)
        return self.post_process(fc_out)
    
    def forward_with_hidden(self, x, h0c0:typing.Tuple[torch.Tensor]=None):
        '''
            `x` shape: (batch, seq, feature)
            `h0` shape: (n_lstm_layer, batch, n_hidden)
            `c0` shape: (n_lstm_layer, batch, n_hidden)
        '''
        lstm_out, (hn, cn) = self.lstm(x, h0c0)
        fc_out = self.fc_layers(lstm_out)
        return self.post_process(fc_out), (hn, cn)

    
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
    
class tanhNormalDistNet(normalDistNet):
    def __init__(self, 
                 n_feature: int, 
                 n_sample: int, 
                 n_hiddens: typing.List[int], 
                 action_bounds: typing.List[float], 
                 sigma_upper_bounds: typing.List[float],
                 ):
        upper_bounds = [ torch.inf]*n_sample+sigma_upper_bounds
        lower_bounds = [-torch.inf]*n_sample+[1e-7]*n_sample
        super().__init__(n_feature, n_sample, n_hiddens, upper_bounds, lower_bounds)
        self.action_bounds = torch.tensor(action_bounds).reshape((1,n_sample))

    def to(self, device: str, **kwargs):
        self.action_bounds = self.action_bounds.to(device)
        return super().to(device, **kwargs)

    def sample(self, output:torch.Tensor):
        '''
            sample from normal dist, output are mean and var.
        '''
        normal_sample = self.distribution(output).rsample()
        return self.contract(normal_sample)

    def contract(self, normal_sample:torch.Tensor):
        contracted = torch.tanh(normal_sample)
        action = self.action_bounds*contracted
        return action
    
    def tanh_log_prob(self, output:torch.Tensor, normal_sample:torch.Tensor):
        '''
            log prob of tanh normal dist.
        '''
        contracted = torch.tanh(normal_sample)
        log_prob = self.distribution(output).log_prob(normal_sample)
        log_prob = log_prob - torch.log(1-torch.tanh(contracted)**2+1e-7)
        log_prob = torch.sum(log_prob, dim=-1, keepdim=True)
        return log_prob
    
    def tanh_sample(self, obss):
        '''
            returns: `actions`, `tanh_log_probs`
        '''
        outputs = self.forward(obss)
        normal_dist = self.distribution(outputs)
        normal_samples = normal_dist.rsample()
        actions = self.contract(normal_samples)
        tanh_log_probs = self.tanh_log_prob(outputs, normal_samples)
        return actions, tanh_log_probs
    

class trackNet(boundedFcNet):
    def __init__(self, 
                 n_state: int,
                 n_control: int, 
                 n_hiddens: typing.List[int], 
                 state_weights: torch.Tensor, 
                 control_weights: torch.Tensor,
                 upper_bounds: torch.Tensor, 
                 lower_bounds: torch.Tensor):
        super().__init__(n_state*2, n_control, n_hiddens, upper_bounds, lower_bounds)
        self.n_state = n_state
        self.state_weights = torch.diag(state_weights)
        self.control_weights = torch.diag(control_weights)

    def to(self, device:str, **kwargs):
        self.state_weights = self.state_weights.to(device=device)
        self.control_weights = self.control_weights.to(device=device)
        return super().to(device=device, **kwargs)

    def loss(self, state_seq:torch.Tensor, track_seq:torch.Tensor):
        '''
            args:
                `state_seq`: shape (horizon,batch_size,n_state), must be propagated by `propagatorT` and contain grad.
                `track_seq`: shape (horizon,batch_size,n_state).
        '''
        track_seq = track_seq.detach()
        x = torch.concat((state_seq,track_seq), dim=-1)
        error_seq = state_seq-track_seq # shape (horizon,batch_size,n_state)
        control_seq = self.forward(x) # shape (horizon,batch_size,n_control)
        state_loss = torch.sum((error_seq@self.state_weights)*error_seq, dim=(0,-1)) # shape (batch_size,)
        control_loss = torch.sum((control_seq@self.control_weights)*control_seq, dim=(0,-1)) # shape (batch_size,)
        total_loss = torch.mean(state_loss+control_loss)
        return total_loss

class QNet(fcNet):
    def __init__(self, 
                 n_obs:int, 
                 n_action:int,
                 n_hiddens:typing.List[int], 
                 ):
        super(fcNet, self).__init__()
        self.n_obs = n_obs
        self.n_action = n_action
        self.n_feature = n_obs+n_action
        self.n_output = 1
        self._dummy_param = nn.Parameter(torch.empty(0))
        
        fc = []
        for i in range(len(n_hiddens)):
            fc.append(nn.Linear(self.n_feature if i == 0 else n_hiddens[i-1], n_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(n_hiddens[-1], 1))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        fc_out = self.fc_layers(x)
        return self.post_process(fc_out)
    
    def forward_(self, obs_action):
        x = obs_action
        fc_out = self.fc_layers(x)
        return self.post_process(fc_out)