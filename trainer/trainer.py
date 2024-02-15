from agent.agent import rlAgent
from env.env import treeEnv
from tree.tree import stateDict

import numpy as np

class treeTrainer:
    def __init__(self, env:treeEnv, agent:rlAgent, gamma=1.) -> None:
        self.env = env
        self.agent = agent
        self.gamma = gamma # discount factor

    @property
    def tree(self):
        return self.env.tree

    def new_state(self) -> stateDict:
        # for debug, to be overloaded.
        sd = stateDict(self.env.state_dim, self.env.obs_dim, self.env.action_dim)
        sd.state[:3] = np.random.uniform(low=-5, high=5, size=3)
        sd.state[3:] = np.random.uniform(low=-0.1, high=0.1, size=3)
        return sd

    def reset_env(self):
        sd = self.new_state()
        self.env.reset(sd)

    def simulate(self):
        self.reset_env()
        while not self.env.step(self.agent):
            pass
        dicts = self.tree.get_transDicts()
        return dicts
    
    def update(self, trans_dict:dict):
        self.agent.update(trans_dict)