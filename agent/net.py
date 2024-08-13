import torch
from torch import nn
import typing
from agent.baseModule import baseModule
from agent.module import *
import utils

from agent.OBC import outputBoundConfig_mp as outputBoundConfig

class fcNet(baseModule):
    def __init__(self, 
                 n_feature:int, 
                 n_output:int,
                 n_hiddens:typing.List[int], 
                 ):
        super().__init__()
        self.n_feature = n_feature
        self.n_output = n_output
        
        fc = []
        for i in range(len(n_hiddens)):
            fc.append(nn.Linear(n_feature if i == 0 else n_hiddens[i-1], n_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(n_hiddens[-1], n_output))
        self.fc_layers = nn.Sequential(*fc)
    
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

class discretePolicyNet(fcNet):
    def __init__(self, n_feature: int, n_output: int, n_hiddens: torch.List[int]):
        super().__init__(n_feature, n_output, n_hiddens)

    def post_process(self, x):
        return nn.functional.softmax(x, dim=-1)
    
    def distribution(self, output):
        dist = torch.distributions.Categorical(output)
        return dist

    def sample(self, output):
        dist = self.distribution(output)
        return dist.sample()
    
    def nominal_output(self, x, require_grad=True):
        '''
            argmax of categorial dist.
        '''
        output = self.forward(x)
        nominal = torch.argmax(output, dim=-1)
        return nominal

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
    
class linearFeedback(boundedFcNet):
    def __init__(self, n_feature:int, n_output:int, bias=False, 
                 upper_bounds:torch.Tensor|None=None,
                 lower_bounds:torch.Tensor|None=None,):
        baseModule.__init__(self)
        self.n_feature = n_feature
        self.n_output = n_output
        self.linear = nn.Linear(n_feature, n_output, bias=bias)
        if upper_bounds is None:
            upper_bounds = torch.tensor([ torch.inf]*n_output)
        if lower_bounds is None:
            lower_bounds = torch.tensor([-torch.inf]*n_output)
        self.obc = outputBoundConfig(upper_bounds, lower_bounds)
        '''
            see `outputBoundConfig_mp`.
        '''

    def forward(self, x):
        u = self.linear(x)
        return self.post_process(u)
    
class quadraticFunc(baseModule):
    def __init__(self, n_feature:int, n_rank:int|None=None):
        super().__init__()
        self.n_feature = n_feature
        self.n_output = 1
        self.n_rank = n_feature if n_rank is None else n_rank
        self.L = nn.Parameter(torch.randn((self.n_feature, self.n_rank)))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x:torch.Tensor):
        return -torch.sum((x@self.L)**2, dim=-1, keepdim=True) + self.b
    
class ballFcNet(boundedFcNet):
    def __init__(self,
                 n_feature:int,
                 n_output:int,
                 n_hiddens:typing.List[int],
                 rad_bound:float
                 ):
        upper_bounds = [ torch.inf]*n_output + [rad_bound]
        lower_bounds = [-torch.inf]*n_output + [0]
        super().__init__(n_feature, n_output+1, n_hiddens, upper_bounds, lower_bounds)
        self.dim = n_output

    def post_process(self, x:torch.Tensor):
        shape = list(x.shape)
        shape[-1] = self.dim
        x = x.reshape(-1, self.n_output)
        x = self.obc(x)
        vec = x[:,:-1]
        rad = x[:,-1:]
        norm = torch.linalg.norm(vec, dim=1, keepdim=True)
        vec = vec/norm*rad
        vec = vec.reshape(shape)
        return vec
    
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
    
    def post_process(self, x):
        return self.obc(x)
    
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
        self.action_bounds = torch.tensor(action_bounds).reshape((1,n_sample)).float()

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
    
    def uniSample(self, size: int):
        bound = self.action_bounds.flatten()
        dist = torch.distributions.Uniform(-bound, bound)
        sampled = dist.sample((size,)).reshape((size, self.n_sample))
        return sampled
    

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
    

class dualNet(boundedFcNet):
    def __init__(self, 
                 n_feature: int, 
                 n_output_action: int, 
                 n_output_value: int, # =1
                 n_hiddens: torch.List[int], 
                 upper_bounds: torch.Tensor, 
                 lower_bounds: torch.Tensor
                ):
        n_output = n_output_action + n_output_value
        super().__init__(n_feature, n_output, n_hiddens, upper_bounds, lower_bounds)
        self.n_output_action = n_output_action
        self.n_output_value = n_output_value

    def split_output(self, output):
        '''
            return action part and value part of dual net output.
        '''
        return output[:,:self.n_output_action], output[:,self.n_output_action:]
    
    def sample_action(self, output_a):
        '''
            sample action from output.
        '''
        return output_a
    
    def sample_value(self, output_v):
        '''
            sample value from output.
        '''
        return output_v
    
    def sample(self, output):
        '''
            sample action and value from output.
        '''
        output_a, output_v = self.split_output(output)
        sample_a = self.sample_action(output_a)
        sample_v = self.sample_value(output_v)
        return sample_a, sample_v
    
    def clip(self, x:torch.Tensor):
        '''
            clip output to output bounds.
        '''
        x = x.clamp(self.obc.lower_bounds[:self.n_output_action], self.obc.upper_bounds[:self.n_output_action])
        return x
    
class deepSet(baseModule):
    def __init__(self, 
                 n_feature: int, 
                 n_fc0_output: int, 
                 n_fc0_hiddens: typing.List[int], 
                 n_fc1_output: int,
                 n_fc1_hiddens: typing.List[int],
                 upper_bounds: torch.Tensor=None, 
                 lower_bounds: torch.Tensor=None):
        '''
            TODO: `obc`
        '''
        super().__init__()
        self.n_feature = n_feature
        self.n_fc0_output = n_fc0_output
        self.n_fc1_output = n_fc1_output

        fc = []
        for i in range(len(n_fc0_hiddens)):
            fc.append(nn.Linear(n_feature if i == 0 else n_fc0_hiddens[i-1], n_fc0_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(n_fc0_hiddens[-1], n_fc0_output))
        self.fc0 = nn.Sequential(*fc)

        fc = []
        for i in range(len(n_fc1_hiddens)):
            fc.append(nn.Linear(n_fc0_output if i == 0 else n_fc1_hiddens[i-1], n_fc1_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(n_fc1_hiddens[-1], n_fc1_output))
        self.fc1 = nn.Sequential(*fc)

    def forward(self, x):
        '''
            x shape: (batch_size, set_size, n_feature)
        '''
        y = self.fc0(x) # (batch_size, set_size, n_fc0_output)
        y = torch.mean(y, dim=1) # (batch_size, n_fc0_output)
        y = self.fc1(y) # (batch_size, n_fc1_output)
        return y

class multiLayerDeepSet(baseModule):
    def __init__(self, 
                 n_feature: int, 
                 n_fc0_output: int, 
                 n_fc0_hiddens: typing.List[int], 
                 n_fc0_layers:int,
                 n_fc1_output: int,
                 n_fc1_hiddens: typing.List[int],
                 upper_bounds: torch.Tensor=None, 
                 lower_bounds: torch.Tensor=None):
        '''
            TODO: `obc`
        '''
        super().__init__()
        self.n_feature = n_feature
        self.n_fc0_output = n_fc0_output
        self.n_fc0_layers = n_fc0_layers
        self.n_fc1_output = n_fc1_output

        self.FC0 = []
        for j in range(n_fc0_layers):
            fc = []
            for i in range(len(n_fc0_hiddens)):
                fc.append(nn.Linear(n_feature if i == 0 else n_fc0_hiddens[i-1], n_fc0_hiddens[i]))
                fc.append(nn.ReLU())
            fc.append(nn.Linear(n_fc0_hiddens[-1], n_fc0_output))
            fc.append(nn.ReLU())
            _module = nn.Sequential(*fc)
            self.__setattr__(f"_fc0_{j}", _module)
            self.FC0.append(_module)

        fc = []
        for i in range(len(n_fc1_hiddens)):
            fc.append(nn.Linear(n_fc0_layers*n_fc0_output if i == 0 else n_fc1_hiddens[i-1], n_fc1_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(n_fc1_hiddens[-1], n_fc1_output))
        self.fc1 = nn.Sequential(*fc)

    def forward(self, x:torch.Tensor):
        '''
            x shape: (batch_size, set_size, n_feature)
        '''
        batch_size, set_size, n_feature = x.shape
        y = torch.zeros((batch_size, self.n_fc0_layers, set_size, self.n_fc0_output), device=self.device)
        for j in range(self.n_fc0_layers):
            y[:,j] = self.FC0[j](x)
        y, _ = torch.max(y, dim=2) # (batch_size, n_fc0_layers, n_fc0_output)
        y = y.view((batch_size, self.n_fc0_layers*self.n_fc0_output))
        y = self.fc1(y) # (batch_size, n_fc1_output)
        return y
    
class deepSet3(multiLayerDeepSet):
    def __init__(self,
                 n_feature: int,
                 n_fc0_output: int,
                 n_fc0_hiddens: typing.List[int],
                 n_fc1_output: int,
                 n_fc1_hiddens: typing.List[int],
                 share_fc0=True,
                 upper_bounds: torch.Tensor=None,
                 lower_bounds: torch.Tensor=None,):
        baseModule.__init__(self)
        self.n_feature = n_feature
        self.n_fc0_output = n_fc0_output
        self.n_fc1_output = n_fc1_output
        self._share_fc0 = share_fc0
        if share_fc0:
            self._n_fc0_layers = 1
        else:
            self._n_fc0_layers = 3
        self.n_fc0_layers = 3

        self.FC0 = []
        for j in range(self._n_fc0_layers):
            fc = []
            for i in range(len(n_fc0_hiddens)):
                fc.append(nn.Linear(n_feature if i == 0 else n_fc0_hiddens[i-1], n_fc0_hiddens[i]))
                fc.append(nn.ReLU())
            fc.append(nn.Linear(n_fc0_hiddens[-1], n_fc0_output))
            _module = nn.Sequential(*fc)
            self.__setattr__(f"_fc0_{j}", _module)
            self.FC0.append(_module)

        fc = []
        for i in range(len(n_fc1_hiddens)):
            fc.append(nn.Linear(self.n_fc0_layers*n_fc0_output if i == 0 else n_fc1_hiddens[i-1], n_fc1_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(n_fc1_hiddens[-1], n_fc1_output))
        self.fc1 = nn.Sequential(*fc)

    def forward(self, x):
        batch_size, set_size, n_feature = x.shape
        y = torch.zeros((batch_size, self.n_fc0_layers, self.n_fc0_output), device=self.device)
        if self._share_fc0:
            z = self.FC0[0](x)
            y[:,0] = torch.max(z, dim=-2)[0]
            y[:,1] = torch.min(z, dim=-2)[0]
            y[:,2] = torch.mean(z, dim=-2)
        else:
            y[:,0] = torch.max(self.FC0[0](x), dim=-2)[0]
            y[:,1] = torch.min(self.FC0[1](x), dim=-2)[0]
            y[:,2] = torch.mean(self.FC0[2](x), dim=-2)
        y = y.view((batch_size, self.n_fc0_layers*self.n_fc0_output))
        y = self.fc1(y) # (batch_size, n_fc1_output)
        return y
    
class powDeepSet(multiLayerDeepSet):
    '''
        TODO: debug
    '''
    def __init__(self,
                 n_feature: int,
                 n_fc0_output: int,
                 n_fc0_hiddens: typing.List[int],
                 n_fc0_layers:int,
                 fc0_pows:typing.List[int],
                 n_fc1_output: int,
                 n_fc1_hiddens: typing.List[int],
                 share_fc0=True,
                 upper_bounds: torch.Tensor=None,
                 lower_bounds: torch.Tensor=None,):
        if len(fc0_pows) != n_fc0_layers:
            raise ValueError("length of fc0_pows must equal to n_fc0_layers")
        baseModule.__init__(self)
        self.n_feature = n_feature
        self.n_fc0_output = n_fc0_output
        self.n_fc1_output = n_fc1_output
        self._share_fc0 = share_fc0
        if share_fc0:
            self._n_fc0_layers = 1
        else:
            self._n_fc0_layers = n_fc0_layers
        self.n_fc0_layers = n_fc0_layers
        self.fc0_pows = fc0_pows

        self.FC0 = []
        for j in range(self._n_fc0_layers):
            fc = []
            for i in range(len(n_fc0_hiddens)):
                fc.append(nn.Linear(n_feature if i == 0 else n_fc0_hiddens[i-1], n_fc0_hiddens[i]))
                fc.append(nn.ReLU())
            fc.append(nn.Linear(n_fc0_hiddens[-1], n_fc0_output))
            fc.append(nn.ReLU())
            _module = nn.Sequential(*fc)
            self.__setattr__(f"_fc0_{j}", _module)
            self.FC0.append(_module)

        fc = []
        for i in range(len(n_fc1_hiddens)):
            fc.append(nn.Linear(self.n_fc0_layers*n_fc0_output if i == 0 else n_fc1_hiddens[i-1], n_fc1_hiddens[i]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(n_fc1_hiddens[-1], n_fc1_output))
        self.fc1 = nn.Sequential(*fc)

    def forward(self, x):
        batch_size, set_size, n_feature = x.shape
        y = torch.zeros((batch_size, self.n_fc0_layers, self.n_fc0_output), device=self.device)
        if self._share_fc0:
            z = self.FC0[0](x)
            for i in range(self.n_fc0_layers):
                y[:,i] = utils.powMean(z, p=self.fc0_pows[i], dim=-2)
        else:
            for i in range(self.n_fc0_layers):
                y[:,i] = utils.powMean(self.FC0[i](x), p=self.fc0_pows[i], dim=-2)
        y = y.view((batch_size, self.n_fc0_layers*self.n_fc0_output))
        y = self.fc1(y) # (batch_size, n_fc1_output)
        return y
    
