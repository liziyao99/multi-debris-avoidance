import numpy as np
from env.propagator import *
from tree.tree import GST, stateDict
from agent.agent import rlAgent

class treeEnv:
    def __init__(self,
                 state_dim:int,
                 obs_dim:int,
                 action_dim:int,
                 population:int,
                 max_gen:int
                ) -> None:
        self.propagator = Propagator(state_dim, obs_dim, action_dim)
        self.tree = GST(population, max_gen, state_dim, obs_dim, action_dim)

    def reset(self, root_stateDict:stateDict):
        self.tree.reset(root_stateDict)

    def step(self, agent:rlAgent):
        '''
            return: `done`
        '''
        return self.tree.step(agent, self.propagator)
    
    def simulate(self, agent:rlAgent):
        while not self.step(agent):
            pass
        return self.tree.get_transDicts()
    
    @property
    def state_dim(self):
        return self.propagator.state_dim
    
    @property
    def obs_dim(self):
        return self.propagator.obs_dim
    
    @property
    def action_dim(self):
        return self.propagator.action_dim
    
class debugEnv(treeEnv):
    def __init__(self, population: int, max_gen: int) -> None:
        self.propagator = debugPropagator()
        self.tree = GST(population, max_gen, self.propagator.state_dim, self.propagator.obs_dim, self.propagator.action_dim)
    
    def reset(self, root_stateDict:stateDict=None):
        if root_stateDict is None:
            root_stateDict = stateDict(self.propagator.state_dim, self.propagator.obs_dim, self.propagator.action_dim)
        self.tree.reset(root_stateDict)
