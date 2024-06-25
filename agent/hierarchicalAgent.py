from agent.agent import rlAgent
import agent.agent as A
import torch
import typing, os

class hierarchicalAgent:
    def __init__(self, agents:typing.Tuple[rlAgent]):
        self.agents = tuple(agents)
        for agent in agents:
            assert (agent.device == self.device)
        self.last_actions = [None for _ in agents] # last action for each hierarchy TODO
    
    def __getitem__(self, index) -> rlAgent:
        return self.agents[index]
    
    @property
    def device(self):
        return self.agents[0].device

    @property
    def nH(self):
        return len(self.agents)
    
    def act(self, obss:torch.Tensor, h=0, higher_action:torch.Tensor=None):
        if h==0:
            obss = self.obs_preprocess(obss)[0]
            return self.agents[0].act(obss)
        else:
            if higher_action is None:
                higher_action = self.act_chain(obss)[h-1]
            higher_action = higher_action.detach()
            return self.agents[h].act(self.oa_preprocess(obss, higher_action, h))
    
    def act_chain(self, obss:torch.Tensor) -> typing.List[torch.Tensor]:
        actions_chain = [None]
        for i in range(self.nH):
            agent = self.agents[i]
            _, actions = agent.act(self.oa_preprocess(obss, actions_chain[-1], i))
            actions_chain.append(actions)
        return actions_chain[1:]

    def obs_preprocess(self, obss:torch.Tensor):
        '''
            return processed obsercations for each hierarchy.
        '''
        return [obss for _ in self.nH]
    
    def oa_preprocess(self, obss, actions, h:int):
        if h==0:
            return obss
        else:
            return torch.hstack((obss, actions))
    
    def save(self, folder="../model/hierarchical/"):
        if not os.path.exists(folder):
            os.mkdir(folder)
        for i in range(self.nH):
            path = os.path.join(folder, f"h{i}.ptd")
            self.agents[i].save(path)

    def load(self, folder="../model/hierarchical/"):
        for i in range(self.nH):
            path = os.path.join(folder, f"h{i}.ptd")
            self.agents[i].load(path)

    def copy(self, other):
        for i in range(self.nH):
            self.agents[i].copy(other.agents[i])
    

class CWH2_DDPG(hierarchicalAgent):
    def __init__(self, 
                 obs_dim: int, 
                 h1a_hiddens: typing.List[int] = [128] * 5, 
                 h1c_hiddens: typing.List[int] = [128] * 5, 
                 h2a_hiddens: typing.List[int] = [128] * 5,
                 h1out_ub:torch.Tensor=None,
                 h1out_lb:torch.Tensor=None,
                 h2out_ub:torch.Tensor=None, 
                 h2out_lb:torch.Tensor=None, 
                 h1a_lr=1e-5, 
                 h2a_lr=1e-4, 
                 h1c_lr=1e-4, 
                 gamma=0.95,
                 sigma=0.1,
                 tau=0.005,
                 device=None
                ) -> None:
        self.h1obs_dim = obs_dim
        self.h1out_dim = 6
        self.h2obs_dim = 6
        self.h2out_dim = 3

        h1agent = A.DDPG(self.h1obs_dim,
                         self.h1out_dim,
                         gamma=gamma,
                         sigma=sigma,
                         tau=tau,
                         actor_hiddens=h1a_hiddens,
                         critic_hiddens=h1c_hiddens,
                         action_upper_bounds=h1out_ub,
                         action_lower_bounds=h1out_lb,
                         actor_lr=h1a_lr,
                         critic_lr=h1c_lr,
                         device=device)

        h2agent = A.boundedRlAgent(self.h2obs_dim+self.h1out_dim, 
                                   self.h2out_dim, 
                                   actor_hiddens=h2a_hiddens, 
                                   critic_hiddens=[128], # dummy
                                   action_upper_bounds=h2out_ub,
                                   action_lower_bounds=h2out_lb,
                                   actor_lr=h2a_lr,
                                   device=device
                                   )
        
        agents = [h1agent, h2agent]
        super().__init__(agents)

    def obs_preprocess(self, obss:torch.Tensor):
        '''
            return processed obsercations for each hierarchy.
        '''
        return [obss, obss[:,:self.h2obs_dim]]
    
    def oa_preprocess(self, obss, actions, h:int):
        if h==0:
            return obss
        else:
            primals = obss[:,:self.h2obs_dim]
            return torch.hstack((primals, primals+actions))
        
class thrustCW_DDPG(hierarchicalAgent):
    def __init__(self, 
                 obs_dim: int, 
                 h1a_hiddens: typing.List[int] = [128] * 5, 
                 h1c_hiddens: typing.List[int] = [128] * 5, 
                 h2a_hiddens: typing.List[int] = [128] * 5,
                 h1out_ub:torch.Tensor=None,
                 h1out_lb:torch.Tensor=None,
                 h2out_ub:torch.Tensor=None, 
                 h2out_lb:torch.Tensor=None, 
                 h1a_lr=1e-5, 
                 h2a_lr=1e-4, 
                 h1c_lr=1e-4, 
                 gamma=0.95,
                 sigma=0.1,
                 tau=0.005,
                 device=None
                ) -> None:
        self.h1obs_dim = obs_dim
        self.h1out_dim = 3
        self.h2obs_dim = 3 # dummy
        self.h2out_dim = 3 # dummy

        h1agent = A.DDPG(self.h1obs_dim,
                         self.h1out_dim,
                         gamma=gamma,
                         sigma=sigma,
                         tau=tau,
                         actor_hiddens=h1a_hiddens,
                         critic_hiddens=h1c_hiddens,
                         action_upper_bounds=h1out_ub,
                         action_lower_bounds=h1out_lb,
                         actor_lr=h1a_lr,
                         critic_lr=h1c_lr,
                         device=device)

        h2agent = A.boundedRlAgent(self.h2obs_dim+self.h1out_dim, 
                                   self.h2out_dim, 
                                   actor_hiddens=h2a_hiddens, 
                                   critic_hiddens=[128], # dummy
                                   action_upper_bounds=h2out_ub,
                                   action_lower_bounds=h2out_lb,
                                   actor_lr=h2a_lr,
                                   device=device
                                   )
        
        agents = [h1agent, h2agent]
        super().__init__(agents)

    def obs_preprocess(self, obss:torch.Tensor):
        '''
            return processed obsercations for each hierarchy.
        '''
        return [obss, obss[:,:self.h2obs_dim]]
    
    def oa_preprocess(self, obss, actions, h:int):
        if h==0:
            return obss
        else:
            primals = obss[:,:self.h2obs_dim]
            return torch.hstack((primals, primals+actions))
        

class thrustCW_SAC(hierarchicalAgent):
    def __init__(self, 
                 obs_dim: int, 
                 action_bounds:typing.List[float],
                 sigma_upper_bounds:typing.List[float],
                 h1a_hiddens: typing.List[int] = [128] * 5, 
                 h1c_hiddens: typing.List[int] = [128] * 5, 
                 h2a_hiddens: typing.List[int] = [128] * 5,
                 h2out_ub:torch.Tensor=None, 
                 h2out_lb:torch.Tensor=None, 
                 h1a_lr=1e-5, 
                 h2a_lr=1e-4, 
                 h1c_lr=1e-4, 
                 gamma=0.95,
                 tau=0.005,
                 device=None
                ) -> None:
        self.h1obs_dim = obs_dim
        self.h1out_dim = 3
        self.h2obs_dim = 3 # dummy
        self.h2out_dim = 3 # dummy

        h1agent = A.SAC(self.h1obs_dim,
                        self.h1out_dim,
                        action_bounds=action_bounds,
                        sigma_upper_bounds=sigma_upper_bounds,
                        gamma=gamma,
                        tau=tau,
                        actor_hiddens=h1a_hiddens,
                        critic_hiddens=h1c_hiddens,
                        actor_lr=h1a_lr,
                        critic_lr=h1c_lr,
                        device=device)
        
        h2agent = A.boundedRlAgent(self.h2obs_dim+self.h1out_dim, 
                                   self.h2out_dim, 
                                   actor_hiddens=h2a_hiddens, 
                                   critic_hiddens=[128], # dummy
                                   action_upper_bounds=h2out_ub,
                                   action_lower_bounds=h2out_lb,
                                   actor_lr=h2a_lr,
                                   device=device
                                   )
        
        agents = [h1agent, h2agent]
        super().__init__(agents)

    def obs_preprocess(self, obss:torch.Tensor):
        '''
            return processed obsercations for each hierarchy.
        '''
        return [obss, obss[:,:self.h2obs_dim]]
    
    def oa_preprocess(self, obss, actions, h:int):
        if h==0:
            return obss
        else:
            primals = obss[:,:self.h2obs_dim]
            return torch.hstack((primals, primals+actions))
        

class trackCW_SAC(hierarchicalAgent):
    def __init__(self,
                 obs_dim: int,
                 action_bounds:typing.List[float],
                 sigma_upper_bounds:typing.List[float],
                 h1a_hiddens: typing.List[int] = [128] * 5,
                 h1c_hiddens: typing.List[int] = [128] * 5,
                 h2a_hiddens: typing.List[int] = [128] * 5,
                 h2out_ub:torch.Tensor=None,
                 h2out_lb:torch.Tensor=None,
                 h1a_lr=1e-5,
                 h2a_lr=1e-4,
                 h1c_lr=1e-4,
                 gamma=0.95,
                 tau=0.005,
                 device=None
                ) -> None:
        self.h1obs_dim = obs_dim
        self.h1out_dim = 3 # target pos
        self.h2obs_dim = 6 # primal states, TODO: and time
        self.h2out_dim = 3 # thrust

        h1agent = A.SAC(self.h1obs_dim,
                        self.h1out_dim,
                        action_bounds=action_bounds,
                        sigma_upper_bounds=sigma_upper_bounds,
                        gamma=gamma,
                        tau=tau,
                        actor_hiddens=h1a_hiddens,
                        critic_hiddens=h1c_hiddens,
                        actor_lr=h1a_lr,
                        critic_lr=h1c_lr,
                        device=device)
        
        h2agent = A.boundedRlAgent(self.h2obs_dim+self.h1out_dim, 
                                   self.h2out_dim, 
                                   actor_hiddens=h2a_hiddens, 
                                   critic_hiddens=[128], # dummy
                                   action_upper_bounds=h2out_ub,
                                   action_lower_bounds=h2out_lb,
                                   actor_lr=h2a_lr,
                                   device=device
                                   )
        
        agents = [h1agent, h2agent]
        super().__init__(agents)

    def obs_preprocess(self, obss:torch.Tensor):
        '''
            return processed obsercations for each hierarchy.
        '''
        return [obss, obss[:,:self.h2obs_dim]]
    
    def oa_preprocess(self, obss, actions, h:int):
        if h==0:
            return obss
        else:
            primals = obss[:,:self.h2obs_dim]
            return torch.hstack((primals, actions))