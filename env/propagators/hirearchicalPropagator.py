import numpy as np
import torch
from env.dynamic import matrix
import typing

class H2Propagator:
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int, h1_step:int, h2_step:int) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.h1_step = h1_step
        self.h2_step = h2_step

    def getObss(self, states:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError
    
    def getH1Rewards(self, states:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError
    
    def getH2Rewards(self, states:torch.Tensor, h1_actions:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError

    def getTruncatedRewards(self, states:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError
    
    def getNextStates(self, states:torch.Tensor, actions:torch.Tensor, require_grad=False) -> torch.Tensor:
        raise NotImplementedError
    
    def getDones(self, states:torch.Tensor, require_grad=False) -> typing.Tuple[torch.Tensor]:
        '''
            returns: done and terminal reward.
        '''
        raise NotImplementedError
    
    def propagate(self, states:torch.Tensor, h1_actions:torch.Tensor, actions:torch.Tensor):
        '''
            returns: `next_states`, `next_obss`, `h1rewards`, `h2rewards`, `dones`, `terminal_rewards`
        '''
        next_states = self.getNextStates(states, actions, require_grad=True)
        h1rewards = self.getH1Rewards(states, actions)
        h2rewards = self.getH2Rewards(states, h1_actions, actions, require_grad=True)
        dones, terminal_rewards = self.getDones(states)
        next_obss = self.getObss(next_states)
        return next_states, next_obss, h1rewards, h2rewards, dones, terminal_rewards
    
    def randomInitStates(self, n:int) -> torch.Tensor:
        raise NotImplementedError
    
    def obssNormalize(self, obss:torch.Tensor, require_grad=False) -> torch.Tensor:
        with torch.set_grad_enabled(require_grad):
            return obss
    
    def statesDecode(self, states:torch.Tensor):
        raise NotImplementedError
    
    def statesEncode(self, datas:dict):
        raise NotImplementedError