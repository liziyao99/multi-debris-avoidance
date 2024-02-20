from agent.net import *

import typing
import torch
import torch.nn.functional as F

class rlAgent:
    def __init__(self,
                 obs_dim:int,
                 action_dim:int,
                 actor_hiddens:typing.List[int],
                 critic_hiddens:typing.List[int],
                 actor_lr = 1e-5,
                 critic_lr = 1e-4,
                 device=None,
                ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_actor(actor_hiddens, actor_lr)
        self._init_critic(critic_hiddens, critic_lr)

    def _init_actor(self, hiddens, lr):
        self.actor = fcNet(self.obs_dim, self.action_dim, hiddens).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def _init_critic(self, hiddens, lr):
        self.critic = fcNet(self.obs_dim, 1, hiddens).to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)


    def act(self, obs:torch.Tensor):
        obs = obs.to(self.device)
        output = self.actor(obs)
        sample = self.actor.sample(output)
        return output, sample

    def update(self, trans_dict) -> typing.Tuple[float]:
        '''
            returns: `actor_loss`, `critic_loss`.
        '''
        raise(NotImplementedError)
    
    def save(self, path="../model/dicts.ptd"):
        dicts = {
                "p_net": self.actor.state_dict(),
                "v_net": self.critic.state_dict(),
                "p_opt": self.actor_opt.state_dict(),
                "v_opt": self.critic_opt.state_dict(),
            }
        torch.save(dicts, path)

    def load(self, path="../model/dicts.ptd"):
        dicts = torch.load(path)
        self.actor.load_state_dict(dicts["p_net"])
        self.critic.load_state_dict(dicts["v_net"])
        self.actor_opt.load_state_dict(dicts["p_opt"])
        self.critic_opt.load_state_dict(dicts["v_opt"])

class boundedRlAgent(rlAgent):
    def __init__(self,
                 obs_dim:int = 6,
                 action_dim:int = 3,
                 actor_hiddens:typing.List[int] = [128]*5,
                 critic_hiddens:typing.List[int] = [128]*5,
                 action_upper_bounds = None,
                 action_lower_bounds = None,
                 actor_lr = 1e-5,
                 critic_lr = 1e-4,
                 device=None,
                ) -> None:
        if action_upper_bounds is None:
            action_upper_bounds = [torch.inf]*action_dim
        if action_lower_bounds is None:
            action_lower_bounds = [-torch.inf]*action_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self._init_actor(actor_hiddens, action_upper_bounds, action_lower_bounds, actor_lr)
        self._init_critic(critic_hiddens, critic_lr)

    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        self.actor = boundedFcNet(self.obs_dim, self.action_dim, hiddens, upper_bounds, lower_bounds).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def _init_critic(self, hiddens, lr):
        return super()._init_critic(hiddens, lr)

    def explore(self, size:int):
        output = self.actor.obc.uniSample(size).to(self.device)
        sample = self.actor.sample(output)
        return output, sample

class debugAgent(boundedRlAgent):
    def __init__(self,
                 obs_dim:int = 6,
                 action_dim:int = 3,
                 actor_hiddens:typing.List[int] = [128]*5,
                 critic_hiddens:typing.List[int] = [128]*5,
                 action_upper_bound = 1.,
                 action_lower_bound = -1.,
                 actor_lr = 1e-5,
                 critic_lr = 1e-4,
                 device=None,
                ) -> None:
        upper_bounds = [action_upper_bound]*action_dim + [1]*action_dim
        lower_bounds = [action_lower_bound]*action_dim + [0.1]*action_dim
        super().__init__(obs_dim, action_dim, actor_hiddens, critic_hiddens, upper_bounds, lower_bounds, actor_lr, critic_lr, device)
        
    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        self.actor = normalNet(self.obs_dim, self.action_dim, hiddens, upper_bounds, lower_bounds).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def update(self, trans_dict:dict):
        dict_torch = {}
        for key in trans_dict.keys():
            dict_torch[key] = torch.from_numpy(trans_dict[key]).to(self.device)

        paras = self.actor(dict_torch["obss"])
        dist = torch.distributions.Normal(paras[:,:self.action_dim], paras[:,self.action_dim:])
        log_probs = torch.sum(dist.log_prob(dict_torch["actions"]), dim=1)
        actor_loss = (log_probs*dict_torch["regrets"]).mean()
        # actor_loss = -log_probs.mean()

        undone_idx = torch.where(dict_torch["dones"]==False)[0]
        cl_obj = self.critic(dict_torch["obss"]).flatten()[undone_idx] # critic loss object
        cl_tgt = dict_torch["td_targets"][undone_idx] # critic loss target
        critic_loss = F.mse_loss(cl_obj, cl_tgt)

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()
        return actor_loss.item(), critic_loss.item()