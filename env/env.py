import numpy as np
from env.propagator import Propagator
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
        self.propagate = Propagator(state_dim, obs_dim, action_dim)
        self.tree = GST(population, max_gen, state_dim, obs_dim, action_dim)

    def reset(self, root_stateDict:stateDict):
        self.tree.reset(root_stateDict)

    def step(self, agent:rlAgent):
        '''
            return: `done`
        '''
        return self.tree.step(agent, self.propagate)
