from agent.agent import rlAgent
import torch
import typing

class hierarchicalAgent:
    def __init__(self, agents:typing.Tuple[rlAgent]):
        self.agents = tuple(agents)

    @property
    def device(self):
        return self.agents[0].device

    @property
    def nH(self):
        return len(self.agents)
    
    def act(self, obss:torch.Tensor):
        obss = self.obs_preprocess(obss)[0]
        return self.agents[0].act(obss)
    
    def act_chain(self, obss:torch.Tensor):
        obss_chain = self.obs_preprocess(obss)
        actions_chain = []
        for i in range(self.nH):
            agent = self.agents[i]
            if i==0:
                obss_ = obss_chain[0]
            else:
                obss_ = torch.hstack((obss_chain[i],actions_chain[i-1]))
            _, actions = agent.act(obss_)
            actions_chain.append(actions)

    def obs_preprocess(self, obss:torch.Tensor):
        '''
            return processed obsercations for each hierarchy.
        '''
        return [obss for _ in self.nH]