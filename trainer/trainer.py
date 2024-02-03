from agent.agent import rlAgent
from env.env import treeEnv
from tree.tree import stateDict

class treeTrainer:
    def __init__(self, env:treeEnv, agent:rlAgent, gamma=1.) -> None:
        self.env = env
        self.agent = agent
        self.gamma = gamma

    @property
    def tree(self):
        return self.env.tree

    def new_state(self) -> stateDict:
        raise(NotImplementedError)

    def reset_env(self):
        sd = self.new_state()
        self.env.reset(sd)

    def simulate(self):
        self.reset_env()
        while not self.env.step():
            pass
        self.tree.get_transDicts()