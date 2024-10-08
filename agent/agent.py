from agent.net import *
from agent.setTransfomer import setTransformer
import data.dicts as D

import typing
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import agent.utils as autils
from tree.searchTree import searchTree

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
        self.modules = []
        self._init_actor(actor_hiddens, actor_lr)
        self._init_critic(critic_hiddens, critic_lr)

    def _init_actor(self, hiddens, lr):
        self.actor = fcNet(self.obs_dim, self.action_dim, hiddens).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.modules.append(self.actor)

    def _init_critic(self, hiddens, lr):
        self.critic = fcNet(self.obs_dim, 1, hiddens).to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.modules.append(self.critic)

    @property
    def trans_dict_keys(self) -> typing.List[str]:
        '''
            required keys in `trans_dict` of `update` method.
        '''
        raise NotImplementedError

    def to(self, device):
        for m in self.modules:
            m.to(device)

    def act(self, obs:torch.Tensor):
        '''
            returns:
                `output`: output of actor.
                `sample`: sampled of `output` if actor is random, else `output`.
        '''
        obs = obs.to(self.device)
        output = self.actor(obs)
        sample = self.actor.sample(output)
        return output, sample
    
    def nominal_act(self, obs:torch.Tensor, require_grad=True):
        '''
            returns: `output` of actor.
        '''
        obs = obs.to(self.device)
        return self.actor.nominal_output(obs, require_grad=require_grad)
    
    def tree_act(self, state:torch.Tensor, prop):
        if state.dim()==0:
            state = state.unsqueeze(0)
        if state.shape[0]>1:
            raise ValueError("only support single observation")
        raise NotImplementedError

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

    def actorDist(self, output):
        '''
            returns: distribution of actor.
        '''
        return self.actor.distribution(output)
    
    def share_memory(self):
        for i in range(len(self.modules)):
            self.modules[i].share_memory()

    def copy(self, other):
        for i in range(len(self.modules)):
            self.modules[i].load_state_dict(other.modules[i].state_dict())

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
            action_upper_bounds = [ torch.inf]*action_dim
        if action_lower_bounds is None:
            action_lower_bounds = [-torch.inf]*action_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.modules = []
        self._init_actor(actor_hiddens, action_upper_bounds, action_lower_bounds, actor_lr)
        self._init_critic(critic_hiddens, critic_lr)

    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        self.actor = boundedFcNet(self.obs_dim, self.action_dim, hiddens, upper_bounds, lower_bounds).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.modules.append(self.actor)

    def _init_critic(self, hiddens, lr):
        return super()._init_critic(hiddens, lr)

    def explore(self, size:int):
        output = self.actor.obc.uniSample(size).to(self.device)
        sample = self.actor.sample(output)
        return output, sample
    
class dualRlAgent(boundedRlAgent):
    def __init__(self, 
                 obs_dim: int = 6, 
                 action_dim: int = 3, 
                 dual_hiddens: typing.List[int] = [128] * 5, 
                 action_upper_bounds=None, 
                 action_lower_bounds=None, 
                 lr=0.0001, 
                 device=None) -> None:
        if action_upper_bounds is None:
            action_upper_bounds = [ torch.inf]*action_dim
        if action_lower_bounds is None:
            action_lower_bounds = [-torch.inf]*action_dim
        for _b in (action_upper_bounds, action_lower_bounds):
            if isinstance(_b, torch.Tensor):
                _b = list(_b.flatten())
        output_upper_bounds = action_upper_bounds + [ torch.inf]
        output_lower_bounds = action_lower_bounds + [-torch.inf]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.modules = []

        self._init_dual(hiddens=dual_hiddens, output_upper_bounds=output_upper_bounds, output_lower_bounds=output_lower_bounds, lr=lr)

    def _init_dual(self, hiddens, output_upper_bounds, output_lower_bounds, lr):
        self.dual = dualNet(self.obs_dim, self.action_dim, 1, hiddens, output_upper_bounds, output_lower_bounds).to(self.device)
        self.dual_opt = torch.optim.Adam(self.dual.parameters(), lr=lr)
        self.modules.append(self.dual)

    def act(self, obs:torch.Tensor):
        '''
            return action part of output and sampled action.
        '''
        obs = obs.to(self.device)
        output = self.dual(obs)
        output_a, _ = self.dual.split_output(output)
        sample_a = self.dual.sample_action(output_a)
        return output_a, sample_a
    
    def _critic(self, obs:torch.Tensor):
        '''
            return sampled value.
        '''
        obs = obs.to(self.device)
        output = self.dual(obs)
        _, output_v = self.dual.split_output(output)
        sample_v = self.dual.sample_action(output_v)
        return sample_v
    
    def a_c(self, obs:torch.Tensor):
        '''
            return sampled action and value.
        '''
        obs = obs.to(self.device)
        output = self.dual(obs)
        return self.dual.sample(output)
    
    def nominal_act(self, obs:torch.Tensor, require_grad=True):
        raise NotImplementedError

    def explore(self, size: int):
        indices = np.arange(self.action_dim)
        output = self.dual.obc.uniSample(size, indices=indices).to(self.device)
        sample = self.dual.sample_action(output)
        return output, sample

class normalDistAgent(boundedRlAgent):
    def __init__(self,
                 obs_dim:int = 6,
                 action_dim:int = 3,
                 actor_hiddens:typing.List[int] = [128]*5,
                 critic_hiddens:typing.List[int] = [128]*5,
                 action_upper_bound = 0.2,
                 action_lower_bound = -0.2,
                 actor_lr = 1e-5,
                 critic_lr = 1e-4,
                 device=None,
                ) -> None:
        bound = (action_upper_bound-action_lower_bound)/2
        upper_bounds = [action_upper_bound]*action_dim + [bound/3]*action_dim
        lower_bounds = [action_lower_bound]*action_dim + [bound/30]*action_dim
        super().__init__(obs_dim, action_dim, actor_hiddens, critic_hiddens, upper_bounds, lower_bounds, actor_lr, critic_lr, device)
        
    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        self.actor = normalDistNet(self.obs_dim, self.action_dim, hiddens, upper_bounds, lower_bounds).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.modules.append(self.actor)

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
        cl_obj = self.critic(dict_torch["obss"]).flatten() # critic loss object
        cl_tgt = dict_torch["td_targets"] # critic loss target
        critic_loss = F.mse_loss(cl_obj, cl_tgt)

        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_opt.step()
        self.critic_opt.step()
        return actor_loss.item(), critic_loss.item()

class PPO_discrete(rlAgent):
    def __init__(self, obs_dim: int, action_num: int, 
                 actor_hiddens: typing.List[int], critic_hiddens: typing.List[int], 
                 actor_lr=0.00001, critic_lr=0.0001, 
                 gamma=0.99, lmd=0.9, clip_eps=0.2, epochs=10,
                 device=None) -> None:
        self.action_num = action_num
        super().__init__(obs_dim, action_num, actor_hiddens, critic_hiddens, actor_lr, critic_lr, device)    
        self.gamma = gamma
        self.lmd = lmd
        self.clip_eps = clip_eps
        self.epochs = epochs

    def _init_actor(self, hiddens, lr):
        self.actor = discretePolicyNet(self.obs_dim, self.action_num, hiddens).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.modules.append(self.actor)

    def update(self, trans_dict):
        trans_dict = D.torch_dict(trans_dict, device=self.device)
        trans_dict = D.cut_dict(trans_dict)
        obss = trans_dict["obss"]
        next_obss = trans_dict["next_obss"]
        actions = trans_dict["actions"].to(torch.int64)
        dones = trans_dict["dones"].view((-1,1))
        rewards = trans_dict["rewards"].view((-1,1))
        
        td_targets = rewards + self.gamma*self.critic(next_obss)*(~dones)

        td_deltas = td_targets - self.critic(obss)
        advantage = utils.compute_advantage(self.gamma, self.lmd, td_deltas)
        old_log_probs = torch.log(self.actor(obss).gather(1, actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(obss).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = F.mse_loss(self.critic(obss), td_targets.detach())
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

        return critic_loss.item(), actor_loss.item()

class PPO(boundedRlAgent):
    def __init__(self, 
                 obs_dim: int = 6, 
                 action_dim: int = 3, 
                 actor_hiddens: typing.List[int] = [128] * 5, 
                 critic_hiddens: typing.List[int] = [128] * 5, 
                 action_upper_bounds:typing.List[float]|None=None, 
                 action_lower_bounds:typing.List[float]|None=None, 
                 actor_lr=1e-5, 
                 critic_lr=1e-4, 
                 gamma=0.99,
                 lmd=0.9,
                 clip_eps=0.2,
                 epochs=10,
                 device=None) -> None:
        if action_upper_bounds is None:
            action_upper_bounds = [ 1]*action_dim
        if action_lower_bounds is None:
            action_lower_bounds = [-1]*action_dim
        if not isinstance(action_upper_bounds, torch.Tensor):
            action_upper_bounds = torch.tensor(action_upper_bounds, dtype=torch.float32, device=device)
        if not isinstance(action_lower_bounds, torch.Tensor):
            action_lower_bounds = torch.tensor(action_lower_bounds, dtype=torch.float32, device=device)
        action_upper_bounds = action_upper_bounds.flatten()
        action_lower_bounds = action_lower_bounds.flatten()
        ranges = (action_upper_bounds-action_lower_bounds)/2
        upper_bounds = torch.cat((action_upper_bounds, ranges/3))
        lower_bounds = torch.cat((action_lower_bounds, ranges/30))
        super().__init__(obs_dim, action_dim, actor_hiddens, critic_hiddens, upper_bounds, lower_bounds, actor_lr, critic_lr, device)
        self.gamma = gamma
        self.lmd = lmd
        self.clip_eps = clip_eps
        self.epochs = epochs

    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        self.actor = normalDistNet(self.obs_dim, self.action_dim, hiddens, upper_bounds, lower_bounds).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.modules.append(self.actor)

    def update(self, trans_dict:dict):
        trans_dict = D.torch_dict(trans_dict, device=self.device)
        trans_dict = D.cut_dict(trans_dict)
        obss = trans_dict["obss"]
        next_obss = trans_dict["next_obss"]
        actions = trans_dict["actions"]
        dones = trans_dict["dones"].view((-1,1))
        rewards = trans_dict["rewards"].view((-1,1))
        
        td_targets = rewards + self.gamma*self.critic(next_obss)*(~dones)

        td_deltas = td_targets - self.critic(obss)
        advantage = utils.compute_advantage(self.gamma, self.lmd, td_deltas)

        old_output = self.actor(obss).detach()
        action_dists = self.actor.distribution(old_output)
        old_log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True).detach()

        for _ in range(self.epochs):
            output = self.actor(obss)
            action_dists = self.actor.distribution(output)
            log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = F.mse_loss(self.critic(obss), td_targets.detach())
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

        return critic_loss.item(), actor_loss.item()
    

class planTrackAgent(boundedRlAgent):
    def __init__(self, 
                 obs_dim: int, 
                 plan_dim: int,
                 control_dim: int,
                 pad_dim=0,
                 state_weights: torch.Tensor=None,
                 control_weights: torch.Tensor=None,
                 actor_hiddens: typing.List[int] = [128] * 5, 
                 tracker_hiddens: typing.List[int] = [128] * 5,
                 critic_hiddens: typing.List[int] = [128] * 5, 
                 plan_upper_bounds:torch.Tensor=None,
                 plan_lower_bounds:torch.Tensor=None,
                 control_upper_bounds:torch.Tensor=None, 
                 control_lower_bounds:torch.Tensor=None, 
                 actor_lr=1e-5, 
                 critic_lr=1e-4, 
                 tracker_lr=1e-4, 
                 device=None
                ) -> None:
        self.obs_dim = obs_dim
        self.plan_dim = plan_dim
        self.control_dim = control_dim
        self.pad_dim = pad_dim
        self.track_dim = plan_dim+pad_dim

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.modules = []

        if state_weights is None:
            state_weights = torch.tensor([1.]*self.track_dim, device=self.device)
        if control_weights is None:
            control_weights = torch.tensor([0.01]*control_dim, device=self.device)

        if control_upper_bounds is None:
            control_upper_bounds = [torch.inf]*control_dim
        if control_lower_bounds is None:
            control_lower_bounds = [-torch.inf]*control_dim
        if plan_upper_bounds is None:
            plan_upper_bounds = [torch.inf]*plan_dim
        if plan_lower_bounds is None:
            plan_lower_bounds = [-torch.inf]*plan_dim
        
        self._init_actor(actor_hiddens, plan_upper_bounds, plan_lower_bounds, actor_lr)
        self._init_critic(critic_hiddens, critic_lr)
        self._init_tracker(tracker_hiddens, state_weights, control_weights, control_upper_bounds, control_lower_bounds, tracker_lr)

    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        self.actor = boundedFcNet(self.obs_dim, self.plan_dim, hiddens, upper_bounds, lower_bounds).to(self.device)
        '''
            input obs and output plan (target state), calling `planner` is more appreciated.
        '''
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.modules.append(self.actor)

    def _init_critic(self, hiddens, lr):
        self.critic = QNet(self.obs_dim, self.plan_dim, hiddens).to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.modules.append(self.critic)
    
    def _init_tracker(self, hiddens, state_weights, control_weights, upper_bounds, lower_bounds, lr):
        self.tracker = trackNet(self.track_dim, self.control_dim, hiddens, state_weights, control_weights, upper_bounds, lower_bounds).to(self.device)
        self.tracker_opt = torch.optim.Adam(self.tracker.parameters(), lr=lr)

    def explore(self, size:int):
        output = self.actor.obc.uniSample(size).to(self.device)
        sample = self.actor.sample(output)
        return output, sample

    def extract_primal_obs(self, obss:torch.Tensor):
        return obss[:, :self.track_dim]
    
    def extract_primal_state(self, states:torch.Tensor):
        return states[:, :self.track_dim]
    
    @property
    def planner(self):
        '''
            wrapped `actor`.
            Input obs and output plan (target state to be tracked).
        '''
        return self.actor
    
    @property
    def planner_opt(self):
        return self.actor_opt
    
    @property
    def Q(self):
        '''
            wrapped `critic`.
            Input (obs, action) and output Q value.
        '''
        return self.critic
    
    @property
    def Q_opt(self):
        return self.critic_opt

    def act(self, obss:torch.Tensor):
        '''
            returns:
                `target_states`: state to be tracked.
                `control`: control to be applied.
        '''
        raise NotImplementedError("TODO: `track_target` output state, but obs needed.")
        obss = obss.to(self.device)
        primal_obss = self.extract_primal_obs(obss)
        target_states, _ = self.track_target(obss)
        tracker_input = torch.hstack((primal_obss, target_obss))
        control = self.tracker(tracker_input)
        return target_states, control
    
    def act_target(self, primal_obss:torch.Tensor, target_obss:torch.Tensor):
        tracker_input = torch.hstack((primal_obss, target_obss))
        control = self.tracker(tracker_input)
        return target_obss, control
    
    def track_target(self, obs:torch.Tensor):
        '''
            returns: `target_states` and `planed_states`.
        '''
        obs = obs.to(self.device)
        planed_states = self.planner(obs)
        pad_zeros = torch.zeros((obs.shape[0], self.pad_dim), device=self.device)
        target_states = torch.hstack((planed_states, pad_zeros))
        return target_states, planed_states
    
    def random_track_target(self, batch_size):
        '''
            returns: `target_states` and `planed_states`.
        '''
        planed_states = self.planner.obc.uniSample(batch_size).to(self.device)
        pad_zeros = torch.zeros((batch_size, self.pad_dim), device=self.device)
        target_states = torch.hstack((planed_states, pad_zeros))
        return target_states, planed_states
    
    def nominal_act(self, obs:torch.Tensor, require_grad=True):
        '''
            returns: `output` of actor.
        '''
        obs = obs.to(self.device)
        return self.actor.nominal_output(obs, require_grad=require_grad)
    
    def save(self, path="../model/dicts.ptd"):
        dicts = {
                "p_net": self.actor.state_dict(),
                "v_net": self.critic.state_dict(),
                "p_opt": self.actor_opt.state_dict(),
                "v_opt": self.critic_opt.state_dict(),
                "tracker": self.tracker.state_dict(),
                "tracker_opt": self.tracker_opt.state_dict()
            }
        torch.save(dicts, path)

    def load(self, path="../model/dicts.ptd"):
        dicts = torch.load(path)
        self.actor.load_state_dict(dicts["p_net"])
        self.critic.load_state_dict(dicts["v_net"])
        self.actor_opt.load_state_dict(dicts["p_opt"])
        self.critic_opt.load_state_dict(dicts["v_opt"])
        self.tracker.load_state_dict(dicts["tracker"])
        self.tracker_opt.load_state_dict(dicts["tracker_opt"])


class H2Agent(boundedRlAgent):
    def __init__(self, 
                 obs_dim: int, 
                 h1obs_dim: int,
                 h2obs_dim: int,
                 h1out_dim: int,
                 h2out_dim: int,
                 h1pad_dim=0,
                 h1a_hiddens: typing.List[int] = [128] * 5, 
                 h2a_hiddens: typing.List[int] = [128] * 5,
                 h1c_hiddens: typing.List[int] = [128] * 5, 
                 h1out_ub:torch.Tensor=None,
                 h1out_lb:torch.Tensor=None,
                 h2out_ub:torch.Tensor=None, 
                 h2out_lb:torch.Tensor=None, 
                 h1a_lr=1e-5, 
                 h2a_lr=1e-4, 
                 h1c_lr=1e-4, 
                 gamma=0.95,
                 device=None
                ) -> None:
        self.obs_dim = obs_dim
        self.h1obs_dim = h1obs_dim
        self.h2obs_dim = h2obs_dim
        self.h1out_dim = h1out_dim
        self.h2out_dim = h2out_dim
        self.pad_dim = h1pad_dim
        self.h1h2_dim = h1out_dim+h1pad_dim
        self.action_dim = h1out_dim

        self.gamma = gamma

        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.modules = []

        if h1out_ub is None:
            h1out_ub = [ torch.inf]*h1out_dim
        if h1out_lb is None:
            h1out_lb = [-torch.inf]*h1out_dim
        if h2out_ub is None:
            h2out_ub = [ torch.inf]*h2out_dim
        if h2out_lb is None:
            h2out_lb = [-torch.inf]*h2out_dim
        
        self._init_networks(h1a_hiddens, h2a_hiddens, h1c_hiddens, h1a_lr, h2a_lr, h1c_lr, h1out_ub, h1out_lb, h2out_ub, h2out_lb)

    def _init_networks(self, h1a_hiddens, h2a_hiddens, h1c_hiddens, h1a_lr, h2a_lr, h1c_lr, h1out_ub, h1out_lb, h2out_ub, h2out_lb):
        self.h1a = boundedFcNet(self.obs_dim, self.h1out_dim, h1a_hiddens, h1out_ub, h1out_lb).to(self.device)
        self.h1a_opt1 = torch.optim.Adam(self.h1a.parameters(), lr=h1a_lr)
        self.h1a_opt2 = torch.optim.Adam(self.h1a.parameters(), lr=h1a_lr)
        self.h1c = QNet(self.h1obs_dim, self.h1out_dim, h1c_hiddens).to(self.device)
        self.h1c_opt = torch.optim.Adam(self.h1c.parameters(), lr=h1c_lr)
        self.h2a = boundedFcNet(self.h2obs_dim+self.h1h2_dim, self.h2out_dim, h2a_hiddens, h2out_ub, h2out_lb).to(self.device)
        self.h2a_opt = torch.optim.Adam(self.h2a.parameters(), lr=h2a_lr)
        self.actor = self.h1a
        self.critic = self.h1c

    def h1explore(self, size:int):
        output = self.h1a.obc.uniSample(size).to(self.device)
        sample = self.h1a.sample(output)
        return output, sample
    
    @property
    def planner(self):
        '''
            wrapped `actor`.
            Input obs and output plan (target state to be tracked).
        '''
        return self.h1a
    
    @property
    def planner_opt(self):
        return self.h1a_opt1
    
    @property
    def Q(self):
        '''
            wrapped `critic`.
            Input (obs, action) and output Q value.
        '''
        return self.h1c
    
    @property
    def Q_opt(self):
        return self.h1c_opt

    def act(self, obss:torch.Tensor):
        return self.h1act(obss)

    def h1obs_preprocess(self, obss:torch.Tensor):
        return obss.to(self.device)

    def h1act(self, obss:torch.Tensor):
        '''
            returns: `output`, `sample`.
        '''
        obss = self.h1obs_preprocess(obss)
        output = self.h1a(obss)
        sample = self.h1a.sample(output)
        return output, sample
    
    def h2obs_preprocess(self, obss:torch.Tensor, h1_actions:torch.Tensor):
        obss = obss.to(self.device)
        obss = obss[:,:self.h2obs_dim]
        h2_input = torch.hstack((obss, h1_actions))
        return h2_input
    
    def h2act(self, obss:torch.Tensor, h1_action:torch.Tensor):
        '''
            returns: `output`, `sample`.
        '''
        h2_input = self.h2obs_preprocess(obss, h1_action)
        output = self.h2a(h2_input)
        sample = self.h2a.sample(output)
        return output, sample
    
    def save(self, path="../model/dicts.ptd"):
        dicts = {
                "h1a_net": self.h1a.state_dict(),
                "h1c_net": self.h1c.state_dict(),
                "h1a_opt1": self.h1a_opt1.state_dict(),
                "h1a_opt2": self.h1a_opt2.state_dict(),
                "h1c_opt": self.h1c_opt.state_dict(),
                "h2a_net": self.h2a.state_dict(),
                "h2a_opt": self.h2a_opt.state_dict()
            }
        torch.save(dicts, path)

    def load(self, path="../model/dicts.ptd"):
        dicts = torch.load(path)
        self.h1a.load_state_dict(dicts["h1a_net"])
        self.h1c.load_state_dict(dicts["h1c_net"])
        self.h1a_opt1.load_state_dict(dicts["h1a_opt1"])
        self.h1a_opt2.load_state_dict(dicts["h1a_opt2"])
        self.h1c_opt.load_state_dict(dicts["h1c_opt"])
        self.h2a.load_state_dict(dicts["h2a_net"])
        self.h2a_opt.load_state_dict(dicts["h2a_opt"])

    def h1update(self, trans_dict):
        '''
            returns: `Q_loss`, `mc_loss`, `ddpg_loss`.
        '''
        trans_dict = D.torch_dict(trans_dict, device=self.device)
        rewards = trans_dict["rewards"].reshape((-1, 1))
        dones = trans_dict["dones"].reshape((-1, 1))
        terminal_rewards = trans_dict["terminal_rewards"].reshape((-1, 1))

        # Q_values = self.Q(trans_dict["obss"], trans_dict["actions"])
        # Q_loss = F.mse_loss(Q_values, trans_dict["Q_targets"].reshape(Q_values.shape))
        # self.Q_opt.zero_grad()
        # Q_loss.backward()
        # self.Q_opt.step()

        Q_values = self.Q(trans_dict["obss"], trans_dict["actions"])
        next_Q_values = self.Q(trans_dict["next_obss"], self.h1a(trans_dict["next_obss"]))*(~dones) + terminal_rewards*dones
        next_Q_values = next_Q_values.detach()
        td_Q_targets = rewards + self.gamma*next_Q_values
        Q_loss = F.mse_loss(Q_values, td_Q_targets)
        self.Q_opt.zero_grad()
        Q_loss.backward()
        self.Q_opt.step()

        actions = self.h1a(trans_dict["obss"])
        mc_loss = torch.mean(-trans_dict["regret_mc"]*torch.norm(actions-trans_dict["actions"],dim=-1))
        self.h1a_opt1.zero_grad()
        mc_loss.backward()
        self.h1a_opt1.step()

        actions = self.h1a(trans_dict["obss"])
        ddpg_loss = torch.mean(-self.Q(trans_dict["obss"], actions))
        self.h1a_opt2.zero_grad()
        ddpg_loss.backward()
        self.h1a_opt2.step()

        return Q_loss.item(), mc_loss.item(), ddpg_loss.item()
    
    def update(self,trans_dict):
        '''
            see `h1update`.
        '''
        return self.h1update(trans_dict)


class SAC(boundedRlAgent):
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int, 
                 action_bounds: typing.List[float], 
                 sigma_upper_bounds: typing.List[float],
                 actor_hiddens: typing.List[int] = [128] * 5, 
                 critic_hiddens: typing.List[int] = [128] * 5, 
                 actor_lr=0.00001, 
                 critic_lr=0.0001, 
                 alpha_lr=0.00001,
                 target_entropy:float=None,
                 gamma=0.99,
                 tau=0.005,
                 device=None) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.modules = []
        self._init_actor(actor_hiddens, action_bounds, sigma_upper_bounds, actor_lr)
        self._init_critic(critic_hiddens, critic_lr)
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float, device=device)
        self.log_alpha.requires_grad = True
        self.log_alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy if target_entropy is not None else -self.action_dim
        self.gamma = gamma
        self.tau = tau

    def _init_actor(self, hiddens, action_bound, sigma_upper_bound, lr):
        self.actor = tanhNormalDistNet(self.obs_dim, self.action_dim, hiddens, action_bound, sigma_upper_bound).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.modules.append(self.actor)

    def _init_critic(self, hiddens, lr):
        self.critic1 = QNet(self.obs_dim, self.action_dim, hiddens).to(self.device)
        self.critic1_opt = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2 = QNet(self.obs_dim, self.action_dim, hiddens).to(self.device)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.target_critic1 = QNet(self.obs_dim, self.action_dim, hiddens).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2 = QNet(self.obs_dim, self.action_dim, hiddens).to(self.device)
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.modules += [self.critic1, self.critic2, self.target_critic1, self.target_critic2]
        self.critic = self.critic1

    def QMin(self, obss, actions, target=False):
        obss = obss.to(self.device)
        actions = actions.to(self.device)
        if target:
            q1 = self.target_critic1(obss, actions)
            q2 = self.target_critic2(obss, actions)
        else:
            q1 = self.critic1(obss, actions)
            q2 = self.critic2(obss, actions)
        return torch.min(q1, q2)
    
    def QEntropy(self, obss, actions, entropy, target=False):
        obss = obss.to(self.device)
        actions = actions.to(self.device)
        return self.log_alpha.exp()*entropy+self.QMin(obss, actions, target=target)
    
    def V(self, obss, target=False):
        actions, log_probs = self.actor.tanh_sample(obss)
        values = self.QEntropy(obss, actions, -log_probs, target=target)
        return values

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data*(1.0-self.tau) + param.data*self.tau)

    def update(self, trans_dict):
        trans_dict = D.torch_dict(trans_dict, device=self.device)
        rewards = trans_dict["rewards"].reshape((-1, 1))
        dones = trans_dict["dones"].reshape((-1, 1))

        next_actions, next_log_probs = self.actor.tanh_sample(trans_dict["next_obss"])
        next_QEntropy = self.QEntropy(trans_dict["next_obss"], next_actions, -next_log_probs, target=True)
        td_targets = rewards + self.gamma*next_QEntropy*(~dones)
        critic1_loss = F.mse_loss(self.critic1(trans_dict["obss"], trans_dict["actions"]), td_targets.detach())
        critic2_loss = F.mse_loss(self.critic2(trans_dict["obss"], trans_dict["actions"]), td_targets.detach())
        self.critic1_opt.zero_grad()
        critic1_loss.backward()
        self.critic1_opt.step()
        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        new_actions, new_log_probs = self.actor.tanh_sample(trans_dict["obss"])
        new_QEntropy = self.QEntropy(trans_dict["obss"], new_actions, -new_log_probs, target=False)
        actor_loss = -new_QEntropy.mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -torch.mean((new_log_probs+self.target_entropy).detach()*self.log_alpha.exp())
        self.log_alpha_opt.zero_grad()
        alpha_loss.backward()
        self.log_alpha_opt.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        critic_loss = (critic1_loss+critic2_loss)/2
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()

    def save(self, path="../model/dicts.ptd"):
        dicts = {
                "actor": self.actor.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic1_opt": self.critic1_opt.state_dict(),
                "critic2": self.critic2.state_dict(),
                "critic2_opt": self.critic2_opt.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "log_alpha": self.log_alpha,
                "log_alpha_opt": self.log_alpha_opt.state_dict(),
            }
        torch.save(dicts, path)

    def load(self, path="../model/dicts.ptd"):
        dicts = torch.load(path)
        self.actor.load_state_dict(dicts["actor"])
        self.actor_opt.load_state_dict(dicts["actor_opt"])
        self.critic1.load_state_dict(dicts["critic1"])
        self.critic1_opt.load_state_dict(dicts["critic1_opt"])
        self.critic2.load_state_dict(dicts["critic2"])
        self.critic2_opt.load_state_dict(dicts["critic2_opt"])
        self.target_critic1.load_state_dict(dicts["target_critic1"])
        self.target_critic2.load_state_dict(dicts["target_critic2"])
        self.log_alpha = dicts["log_alpha"].to(self.device)
        self.log_alpha_opt.load_state_dict(dicts["log_alpha_opt"])

    
class DDPG(boundedRlAgent):
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int, 
                 gamma = 0.99,
                 sigma = 0.1,
                 tau = 0.005,
                 actor_hiddens: typing.List[int] = [128] * 5, 
                 critic_hiddens: typing.List[int] = [128] * 5, 
                 action_upper_bounds=None, 
                 action_lower_bounds=None, 
                 actor_lr=0.00001, 
                 critic_lr=0.0001, 
                 device=None) -> None:
        super().__init__(obs_dim, action_dim, actor_hiddens, critic_hiddens, action_upper_bounds, action_lower_bounds, actor_lr, critic_lr, device)
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau

        self.OU_noise = torch.randn((0, action_dim), device=self.device)
        self.OU_mu = torch.zeros((1, action_dim), device=self.device)
        self.OU_theta = 1e-2*torch.ones((1, action_dim), device=self.device)
        self.OU_sigma = 2e-2*torch.ones((1, action_dim), device=self.device)
        self._OU = True

    def init_OU_noise(self, size, scale=1.):
        self.OU_noise = torch.randn((size, self.action_dim), device=self.device)*scale

    def propagate_OU_noise(self):
        delta_noise = self.OU_theta*(self.OU_mu-self.OU_noise)+self.OU_sigma*torch.randn_like(self.OU_noise)
        self.OU_noise = self.OU_noise+delta_noise
        return self.OU_noise
    
    def test_OU_noise(self, size=1, init_scale=1., horizon=1000):
        noise = torch.zeros((horizon, self.action_dim), device=self.device)
        self.init_OU_noise(size=size, scale=init_scale)
        for i in range(horizon):
            noise[i] = self.propagate_OU_noise()
        return noise

    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        self.actor = boundedFcNet(self.obs_dim, self.action_dim, hiddens, upper_bounds, lower_bounds).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.target_actor = boundedFcNet(self.obs_dim, self.action_dim, hiddens, upper_bounds, lower_bounds).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.modules += [self.actor, self.target_actor]
    
    def _init_critic(self, hiddens, lr):
        self.critic = QNet(self.obs_dim, self.action_dim, hiddens).to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.target_critic = QNet(self.obs_dim, self.action_dim, hiddens).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.modules += [self.critic, self.target_critic]

    def act(self, obs:torch.Tensor, with_OU_noise=False):
        '''
            returns:
                `output`: output of actor.
                `sample`: sampled of `output` if actor is random, else `output`.
        '''
        obs = obs.to(self.device)
        output = self.actor(obs)
        if with_OU_noise and self._OU:
            noise = self.propagate_OU_noise()
            assert noise.shape == output.shape
        else: # white noise
            noise = torch.randn_like(output)*self.sigma
        sample = output + noise
        sample = self.actor.clip(sample)
        return output, sample

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data*(1.0-self.tau) + param.data*self.tau)

    def update(self, trans_dict):
        _OU = self._OU
        self._OU = False
        trans_dict = D.torch_dict(trans_dict, device=self.device)
        rewards = trans_dict["rewards"].reshape((-1, 1))
        dones = trans_dict["dones"].reshape((-1, 1))

        next_q_values = self.target_critic(trans_dict["next_obss"], self.target_actor(trans_dict["next_obss"]))
        # next_q_values = self.critic(trans_dict["next_obss"], self.actor(trans_dict["next_obss"]))
        q_targets = rewards + self.gamma*next_q_values*(~dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(trans_dict["obss"], trans_dict["actions"]), q_targets.detach()))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -torch.mean(self.critic(trans_dict["obss"], self.actor(trans_dict["obss"])))
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        self._OU = _OU

        return critic_loss.item(), actor_loss.item()
    
    def save(self, path="../model/dicts.ptd"):
        dicts = {
                "actor": self.actor.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
            }
        torch.save(dicts, path)

    def load(self, path="../model/dicts.ptd"):
        dicts = torch.load(path)
        self.actor.load_state_dict(dicts["actor"])
        self.actor_opt.load_state_dict(dicts["actor_opt"])
        self.critic.load_state_dict(dicts["critic"])
        self.critic_opt.load_state_dict(dicts["critic_opt"])
        self.target_actor.load_state_dict(dicts["target_actor"])
        self.target_critic.load_state_dict(dicts["target_critic"])

class DDPG_V(DDPG):
    def __init__(self, 
                 obs_dim: int = 6, 
                 action_dim: int = 3, 
                 gamma = 0.99,
                 sigma = 0.1,
                 tau = 0.005,
                 actor_hiddens: typing.List[int] = [128] * 5, 
                 critic_hiddens: typing.List[int] = [128] * 5, 
                 action_upper_bounds=None, 
                 action_lower_bounds=None, 
                 actor_lr=0.00001, 
                 critic_lr=0.0001, 
                 device=None) -> None:
        boundedRlAgent.__init__(self, obs_dim, action_dim, actor_hiddens, critic_hiddens, action_upper_bounds, action_lower_bounds, actor_lr, critic_lr, device)
        self.gamma = gamma
        self.sigma = sigma
        self.tau = tau
    
    def _init_critic(self, hiddens, lr):
        self.critic = fcNet(self.obs_dim, 1, hiddens).to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.target_critic = fcNet(self.obs_dim, 1, hiddens).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.modules += [self.critic, self.target_critic]

    def update(self, trans_dict, prop, step:int=10):
        _OU = self._OU
        self._OU = False
        trans_dict = D.torch_dict(trans_dict, device=self.device)
        # rewards = trans_dict["rewards"].reshape((-1, 1))
        # dones = trans_dict["dones"].reshape((-1, 1))
        batch_size = trans_dict["obss"].shape[0]

        critic_loss = torch.mean(F.mse_loss(self.critic(trans_dict["obss"]), trans_dict["values"]))
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        states = trans_dict["states"]
        obss = prop.getObss(states)
        actor_loss = torch.zeros((batch_size, 1), device=self.device)
        done_flags = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.device)
        for i in range(step):
            _, actions = self.act(obss)
            states, rewards, dones, obss = prop.propagate(states, actions, require_grad=True)
            actor_loss = actor_loss - self.critic(obss)*~done_flags
            done_flags |= dones.unsqueeze(dim=1)
        actor_loss = actor_loss/step
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        self._OU = _OU

        return critic_loss.item(), actor_loss.item()


class lstmDDPG(DDPG):
    def __init__(self,
                 main_obs_dim: int,
                 sub_obs_dim: int,
                 sub_feature_dim: int,
                 lstm_num_layers: int,
                 action_dim: int,
                 gamma = 0.99,
                 sigma = 0.1,
                 tau = 0.005,
                 actor_hiddens: typing.List[int] = [128] * 5,
                 critic_hiddens: typing.List[int] = [128] * 5,
                 action_upper_bounds=None,
                 action_lower_bounds=None,
                 actor_lr=0.00001,
                 critic_lr=0.0001,
                 partial_dim = None,
                 partial_hiddens = [128],
                 partial_lr=0.0001,
                 device=None) -> None:
        self.main_obs_dim = main_obs_dim
        self.sub_obs_dim = sub_obs_dim
        '''
            input size of lstm
        '''
        self.sub_feature_dim = sub_feature_dim
        '''
            hidden size of lstm
        '''
        self.fc_in_dim = main_obs_dim+sub_feature_dim
        DDPG.__init__(self, self.fc_in_dim, action_dim, gamma, sigma, tau, actor_hiddens, critic_hiddens, action_upper_bounds, action_lower_bounds, actor_lr, critic_lr, device)
        self.lstm = nn.LSTM(input_size=self.sub_obs_dim, hidden_size=self.sub_feature_dim, num_layers=lstm_num_layers, batch_first=True).to(device)
        self.target_lstm = nn.LSTM(input_size=self.sub_obs_dim, hidden_size=self.sub_feature_dim, num_layers=lstm_num_layers, batch_first=True).to(device)
        self.lstm_opt = torch.optim.Adam(self.lstm.parameters(), lr=critic_lr)
        self.modules.append(self.lstm)

        # self.actor_opt = torch.optim.Adam(list(self.actor.parameters())+list(self.lstm.parameters()), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(list(self.critic.parameters())+list(self.lstm.parameters()), lr=critic_lr)

        self.partial_dim = partial_dim
        if partial_dim is not None:
            self.partial_fc = fcNet(self.fc_in_dim, partial_dim, partial_hiddens).to(device)
            self.modules.append(self.partial_fc)
            self.partial_opt = torch.optim.Adam(list(self.lstm.parameters())+list(self.partial_fc.parameters()), lr=partial_lr)
        else:
            self.partial_fc = None
            self.partial_opt = None

        self._permute = True
        '''
            randomly permute input seq for lstm.
        '''

    def get_fc_input(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False, permute=None):
        '''
            `main_obs`: shape (batch_size, main_obs_dim)
            `sub_seq`: shape (batch_size, seq_len, sub_obs_dim)
        '''
        lstm = self.target_lstm if target else self.lstm
        permute = self._permute if permute is None else permute
        if isinstance(sub_seq, torch.Tensor): 
            if permute:
                permute_idx = np.random.choice(np.arange(sub_seq.shape[1]), sub_seq.shape[1], replace=False)
                sub_seq = sub_seq[:, permute_idx]
            sub_feature = lstm(sub_seq.to(self.device))[0][:,-1] # shape: (batch_size, sub_feature_dim)
        elif isinstance(sub_seq, typing.Iterable):
            batch_size = len(sub_seq)
            sub_feature = torch.zeros((batch_size, self.sub_feature_dim), device=self.device)
            stacked, indices = autils.var_len_seq_sort(sub_seq)
            for i in range(len(stacked)):
                if permute:
                    permute_idx = np.random.choice(np.arange(stacked[i].shape[1]), stacked[i].shape[1], replace=False)
                    stacked[i] = stacked[i][:, permute_idx]
                sub_feature[indices[i]] = lstm(stacked[i].to(self.device))[0][:,-1]
        if self.main_obs_dim==0:
            fc_in = sub_feature
        else:
            fc_in = torch.cat([main_obs.to(self.device), sub_feature], dim=-1)
        return fc_in
    
    def partial_output(self, main_obs:torch.Tensor, sub_seq:torch.Tensor):
        if self.partial_fc is None:
            raise RuntimeError("no `partial_fc`")
        else:
            fc_in = self.get_fc_input(main_obs, sub_seq)
            return self.partial_fc(fc_in)
    
    def act(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False, 
            permute=None, with_OU_noise=True):
        '''
            returns:
                `output`: output of actor.
                `sample`: sampled of `output` if actor is random, else `output`.
        '''
        actor = self.target_actor if target else self.actor
        obs = self.get_fc_input(main_obs, sub_seq, target=target, permute=permute)
        output = actor(obs)
        if with_OU_noise and self._OU:
            noise = self.propagate_OU_noise()
        else:
            noise = torch.randn_like(output)*self.sigma
        sample = output + noise
        sample = actor.clip(sample)
        return output, sample

    
    def _critic(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], actions:torch.Tensor, target=False,
                permute=None):
        critic = self.target_critic if target else self.critic
        obs = self.get_fc_input(main_obs, sub_seq, target=target, permute=permute)
        q_values = critic(obs, actions)
        return q_values
    
    def update(self, trans_dict, n_update=1):
        _OU = self._OU
        self._OU = False
        rewards = torch.stack(trans_dict["rewards"]).reshape((-1, 1)).to(self.device)
        dones = torch.stack(trans_dict["dones"]).reshape((-1, 1)).to(self.device)
        main_obss = torch.stack(trans_dict["primal_obss"]).to(self.device)
        next_main_obss = torch.stack(trans_dict["next_primal_obss"]).to(self.device)
        sub_obss = trans_dict["debris_obss"]
        next_sub_obss = trans_dict["next_debris_obss"]
        actions = torch.stack(trans_dict["actions"]).to(self.device)

        for _ in range(n_update):
            _, next_actions = self.act(next_main_obss, next_sub_obss, target=True)
            next_q_values = self._critic(next_main_obss, next_sub_obss, next_actions.detach(), target=True)
            q_targets = rewards + self.gamma*next_q_values*(~dones)
            critic_loss = torch.mean(F.mse_loss(self._critic(main_obss, sub_obss, actions), q_targets.detach()))
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            critic_loss = critic_loss.item()

            curr_actions, _ = self.act(main_obss, sub_obss)
            actor_loss = -torch.mean(self._critic(main_obss, sub_obss, curr_actions))
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss = actor_loss.item()

            if "partial_label" in trans_dict.keys():
                partial_out = self.partial_output(main_obss, sub_obss)
                partial_loss = F.mse_loss(partial_out, trans_dict["partial_label"])
                self.partial_opt.zero_grad()
                partial_loss.backward()
                self.partial_opt.step()
                partial_loss = partial_loss.item()
            else:
                partial_loss = None

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.lstm, self.target_lstm)
        self._OU = _OU

        return critic_loss, actor_loss, partial_loss
    
    def save(self, path="../model/dicts.ptd"):
        dicts = {
                "actor": self.actor.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "lstm": self.lstm.state_dict(),
            }
        torch.save(dicts, path)

    def load(self, path="../model/dicts.ptd"):
        dicts = torch.load(path)
        self.actor.load_state_dict(dicts["actor"])
        self.actor_opt.load_state_dict(dicts["actor_opt"])
        self.critic.load_state_dict(dicts["critic"])
        self.critic_opt.load_state_dict(dicts["critic_opt"])
        self.target_actor.load_state_dict(dicts["target_actor"])
        self.target_critic.load_state_dict(dicts["target_critic"])
        self.lstm.load_state_dict(dicts["lstm"])

class lstmDDPG_V(lstmDDPG, DDPG_V):
    def __init__(self,
                 main_obs_dim: int,
                 sub_obs_dim: int,
                 sub_feature_dim: int,
                 lstm_num_layers: int,
                 action_dim: int,
                 gamma = 0.99,
                 sigma = 0.1,
                 tau = 0.005,
                 actor_hiddens: typing.List[int] = [128] * 5,
                 critic_hiddens: typing.List[int] = [128] * 5,
                 action_upper_bounds=None,
                 action_lower_bounds=None,
                 actor_lr=0.00001,
                 critic_lr=0.0001,
                 partial_dim = None,
                 partial_hiddens = [128],
                 partial_lr=0.0001,
                 device=None) -> None:
        lstmDDPG.__init__(self, main_obs_dim, sub_obs_dim, sub_feature_dim, lstm_num_layers, action_dim, gamma, sigma, tau, actor_hiddens, critic_hiddens, action_upper_bounds, action_lower_bounds, actor_lr, critic_lr, partial_dim, partial_hiddens, partial_lr, device)

    def _init_critic(self, hiddens, lr):
        return DDPG_V._init_critic(self, hiddens, lr)
    
    def _critic(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False, permute=None):
        critic = self.target_critic if target else self.critic
        obs = self.get_fc_input(main_obs, sub_seq, target=target, permute=None)
        values = critic(obs)
        return values
    
    def tree_act(self, main_state:torch.Tensor, sub_state:torch.Tensor, prop, 
                 search_step:int, search_population:int,
                 max_gen=10, select_explore_eps=0.2):
        if main_state.dim()==0:
            main_state = main_state.unsqueeze(0)
        if main_state.shape[0]>1:
            raise ValueError("only support single observation")
        state = (main_state, sub_state)
        obs = prop.getObss(main_state, sub_state)
        tree = searchTree.from_data(state, obs, max_gen=max_gen, gamma=self.gamma, select_explore_eps=select_explore_eps, device=self.device)
        tree.root.V_critic = self._critic(obs[0], obs[1])
        tree.backup()
        for _ in range(search_step):
            selected = tree.select(1, "value")[0]
            parents = [selected]*search_population
            main_state = selected.state[0]
            main_states = torch.cat([main_state]*search_population, dim=0)
            sub_state = selected.state[1]
            main_obss, sub_obs = prop.getObss(main_states, sub_state)
            _, actions = self.act(main_obss, sub_obs)
            actions[1:,:] = self.actor.uniSample(search_population-1)
            states, rewards, dones, obss = prop.propagate(main_states, sub_state, actions)
            V_critics = self._critic(obss[0], obss[1]).detach()

            states = list(states)
            states[0] = states[0].unsqueeze(1)
            states[1] = torch.stack([states[1]]*search_population, dim=0)
            states = list(zip(*states))
            obss = list(obss)
            obss[0] = obss[0].unsqueeze(1)
            obss = list(zip(*obss))
            
            tree.extend(states, obss, actions.unsqueeze(1), rewards.unsqueeze(1), dones.unsqueeze(1), parents=parents, V_critics=V_critics.unsqueeze(1))
            tree.backup()
        return tree.best_action()
    
    def MPC(self, sp0:torch.Tensor, sd0:torch.Tensor, prop, horizon=10, opt_step=10):
        '''
            Model Predictive Control.
            returns: `actions`, `actor_loss.item()`.
        '''
        _OU = self._OU
        self._OU = False
        batch_size = sp0.shape[0]
        if horizon>0:
            for _ in range(opt_step):
                Rewards = torch.zeros((horizon, batch_size), device=self.device)
                truncated_values = torch.zeros(batch_size, device=self.device)
                Values = torch.zeros((horizon, batch_size), device=self.device)
                done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                truncated_steps = (horizon-1)*torch.ones(batch_size, dtype=torch.int32, device=self.device)
                sp, sd = sp0.clone(), sd0.clone()
                op, od = prop.getObss(sp, sd, batch_debris_obss=True)
                for i in range(horizon):
                    _, actions = self.act(op.detach(), od.detach(), permute=False, with_OU_noise=False)
                    (sp, sd), rewards, dones, (op, od) = prop._propagate(sp, sd, actions, new_debris=True, batch_debris_obss=True, require_grad=True)
                    Rewards[i, ~done_flags] = rewards[~done_flags]
                    now_done = dones&~done_flags
                    truncated_values[now_done] = self._critic(op[now_done], od[now_done], permute=False).squeeze(-1)
                    truncated_steps[now_done] = i
                    done_flags = done_flags|dones
                truncated_values[~done_flags] = self._critic(op[~done_flags], od[~done_flags], permute=False).squeeze(-1)
                for i in range(horizon-1, -1, -1):
                    if (i>truncated_steps).all():
                        continue
                    trunc_flags = i==truncated_steps
                    disct_flags = i< truncated_steps
                    Values[i, trunc_flags] = Rewards[i, trunc_flags] + self.gamma*truncated_values[trunc_flags]
                    Values[i, disct_flags] = Rewards[i, disct_flags] + self.gamma*Values[i, disct_flags]
                actor_loss = -Values[0].mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
            actor_loss = actor_loss.item()
        else:
            actor_loss = None
        op, od = prop.getObss(sp0, sd0, batch_debris_obss=True)
        actions, _ = self.act(op, od, permute=False, with_OU_noise=False)
        self._OU = _OU
        return actions, actor_loss
    
    def update(self, trans_dict, prop, horizon:int=1, n_update=10):
        _OU = self._OU
        self._OU = False

        main_obss = torch.stack(trans_dict["primal_obss"]).to(self.device)
        sub_obss = trans_dict["debris_obss"]
        target_values = torch.stack(trans_dict["values"]).to(self.device).reshape((-1,1))
        batch_size = main_obss.shape[0]

        for _ in range(n_update):
            critic_loss = torch.mean(F.mse_loss(self._critic(main_obss, sub_obss, permute=True), target_values.detach()))
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            critic_loss = critic_loss.item()

            sp = torch.stack(trans_dict["primal_states"]).to(self.device)
            # sd = trans_dict["debris_states"] # NOTICE: list of tensor with different length
            sd = torch.cat(trans_dict["debris_states"], dim=0).to(self.device) # shape: (sum(n_debris), 6)
            nd = np.random.randint(1, prop.max_n_debris+1)
            indices = np.random.choice(sd.shape[0], size=(min(nd, sd.shape[0]),), replace=False)
            # sd = prop.randomDebrisStates(nd)
            sd = sd[indices]
            _, actor_loss = self.MPC(sp, sd, prop, horizon=horizon, opt_step=1)

            if "partial_label" in trans_dict.keys():
                partial_out = self.partial_output(main_obss, sub_obss)
                partial_loss = F.mse_loss(partial_out, trans_dict["partial_label"])
                self.partial_opt.zero_grad()
                partial_loss.backward()
                self.partial_opt.step()
                partial_loss = partial_loss.item()
            else:
                partial_loss = None

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        self.soft_update(self.lstm, self.target_lstm)

        self._OU = _OU

        return critic_loss, actor_loss, partial_loss
    
class deepSetDDPG_V(lstmDDPG_V):
    def __init__(self, main_obs_dim: int, sub_obs_dim: int, sub_feature_dim: int, 
                 action_dim: int, 
                 gamma=0.99, sigma=0.1, tau=0.005, 
                 n_fc0_output=12,
                 n_fc0_layers=8,
                 actor_hiddens: typing.List[int] = [128]*5, 
                 critic_hiddens: typing.List[int] = [128]*5, 
                 n_fc0_hiddens: typing.List[int] = [128],
                 n_fc1_hiddens: typing.List[int] = [128]*4,
                 action_upper_bounds=None, 
                 action_lower_bounds=None, 
                 actor_lr=0.00001, 
                 critic_lr=0.0001, 
                 device=None) -> None:
        self.main_obs_dim = main_obs_dim
        self.sub_obs_dim = sub_obs_dim
        self.sub_feature_dim = sub_feature_dim
        self.fc_in_dim = main_obs_dim+sub_feature_dim
        DDPG.__init__(self, self.fc_in_dim, action_dim, gamma, sigma, tau, actor_hiddens, critic_hiddens, action_upper_bounds, action_lower_bounds, actor_lr, critic_lr, device)

        self.deepSet = deepSet3(n_feature=sub_obs_dim, 
                                n_fc0_output=n_fc0_output,
                                n_fc0_hiddens=n_fc0_hiddens,
                                n_fc1_output=sub_feature_dim,
                                n_fc1_hiddens=n_fc1_hiddens).to(device)
        self.critic_opt = torch.optim.Adam(list(self.critic.parameters())+list(self.deepSet.parameters()), lr=critic_lr)
        self.modules.append(self.deepSet)

    def get_fc_input(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False, permute=None):
        '''
            `main_obs`: shape (batch_size, main_obs_dim)
            `sub_seq`: shape (batch_size, seq_len, sub_obs_dim)
        '''
        if isinstance(sub_seq, torch.Tensor): 
            sub_feature = self.deepSet(sub_seq.to(self.device)) # shape: (batch_size, sub_feature_dim)
        elif isinstance(sub_seq, typing.Iterable):
            batch_size = len(sub_seq)
            sub_feature = torch.zeros((batch_size, self.sub_feature_dim), device=self.device)
            stacked, indices = autils.var_len_seq_sort(sub_seq)
            for i in range(len(stacked)):
                sub_feature[indices[i]] = self.deepSet(stacked[i].to(self.device))
        if self.main_obs_dim==0:
            fc_in = sub_feature
        else:
            fc_in = torch.cat([main_obs.to(self.device), sub_feature], dim=-1)
        return fc_in
    
    def update(self, trans_dict, prop, horizon:int=1, n_update=1):
        main_obss = torch.stack(trans_dict["primal_obss"]).to(self.device)
        sub_obss = trans_dict["debris_obss"]
        target_values = torch.stack(trans_dict["values"]).to(self.device).reshape((-1,1))
        batch_size = main_obss.shape[0]

        for _ in range(n_update):
            critic_loss = torch.mean(F.mse_loss(self._critic(main_obss, sub_obss), target_values.detach()))
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            critic_loss = critic_loss.item()

            sp = torch.stack(trans_dict["primal_states"]).to(self.device)
            # sd = trans_dict["debris_states"] # NOTICE: list of tensor with different length
            sd = torch.cat(trans_dict["debris_states"], dim=0).to(self.device) # shape: (sum(n_debris), 6)
            nd = np.random.randint(1, prop.max_n_debris+1)
            indices = np.random.choice(sd.shape[0], size=(min(nd, sd.shape[0]),), replace=False)
            # sd = prop.randomDebrisStates(nd)
            sd = sd[indices]
            _, actor_loss = self.MPC(sp, sd, prop, horizon=horizon, opt_step=1)

            partial_loss = None

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return critic_loss, actor_loss, partial_loss
    
    def save(self, path="../model/dicts.ptd"):
        dicts = {
                "actor": self.actor.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "deepSet": self.deepSet.state_dict(),
            }
        torch.save(dicts, path)

    def load(self, path="../model/dicts.ptd"):
        dicts = torch.load(path)
        self.actor.load_state_dict(dicts["actor"])
        self.actor_opt.load_state_dict(dicts["actor_opt"])
        self.critic.load_state_dict(dicts["critic"])
        self.critic_opt.load_state_dict(dicts["critic_opt"])
        self.target_actor.load_state_dict(dicts["target_actor"])
        self.target_critic.load_state_dict(dicts["target_critic"])
        self.deepSet.load_state_dict(dicts["deepSet"])

class setTransDDPG_V(deepSetDDPG_V):
    def __init__(self, main_obs_dim: int, sub_obs_dim: int, sub_feature_dim: int,
                 action_dim: int,
                 num_heads: int,
                 encoder_depth: int,
                 gamma=0.99, sigma=0.1, tau=0.005,
                 actor_hiddens: typing.List[int] = [128]*5,
                 critic_hiddens: typing.List[int] = [128]*5,
                 encoder_fc_hiddens: typing.List[int] = [128]*2,
                 initial_fc_hiddens:typing.List[int]|None = None,
                 initial_fc_output:int|None = None,
                 pma_fc_hiddens: typing.List[int] = [128]*2,
                 pma_mab_fc_hiddens: typing.List[int] = [128]*2,
                 action_upper_bounds=None,
                 action_lower_bounds=None,
                 actor_lr=0.00001,
                 critic_lr=0.0001,
                 device=None) -> None:
        self.main_obs_dim = main_obs_dim
        self.sub_obs_dim = sub_obs_dim
        self.sub_feature_dim = sub_feature_dim
        self.fc_in_dim = main_obs_dim+sub_feature_dim
        DDPG.__init__(self, self.fc_in_dim, action_dim, gamma, sigma, tau, actor_hiddens, critic_hiddens, action_upper_bounds, action_lower_bounds, actor_lr, critic_lr, device)
        self._init_setTrans(num_heads, encoder_fc_hiddens, encoder_depth, initial_fc_hiddens, initial_fc_output, pma_fc_hiddens, pma_mab_fc_hiddens, critic_lr)

    def _init_setTrans(self,
                       num_heads, encoder_fc_hiddens, encoder_depth,
                       initial_fc_hiddens, initial_fc_output,
                       pma_fc_hiddens, pma_mab_fc_hiddens, lr):
        self.setTrans = setTransformer(n_feature=self.sub_obs_dim,
                                       num_heads=num_heads,
                                       encoder_fc_hiddens=encoder_fc_hiddens,
                                       encoder_depth=encoder_depth,
                                       initial_fc_hiddens=initial_fc_hiddens,
                                       initial_fc_output=initial_fc_output,
                                       n_output=self.sub_feature_dim,
                                       pma_fc_hiddens=pma_fc_hiddens,
                                       pma_mab_fc_hiddens=pma_mab_fc_hiddens,
                                       ).to(self.device)
        self.critic_opt = torch.optim.Adam(list(self.critic.parameters())+list(self.setTrans.parameters()), lr=lr)
        self.modules.append(self.setTrans)
        
    def get_fc_input(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False, permute=None):
        '''
            `main_obs`: shape (batch_size, main_obs_dim)
            `sub_seq`: shape (batch_size, seq_len, sub_obs_dim)
        '''
        if isinstance(sub_seq, torch.Tensor): 
            sub_feature = self.setTrans(sub_seq.to(self.device)) # shape: (batch_size, sub_feature_dim)
        elif isinstance(sub_seq, typing.Iterable):
            batch_size = len(sub_seq)
            sub_feature = torch.zeros((batch_size, self.sub_feature_dim), device=self.device)
            stacked, indices = autils.var_len_seq_sort(sub_seq)
            for i in range(len(stacked)):
                sub_feature[indices[i]] = self.setTrans(stacked[i].to(self.device))
        if self.main_obs_dim==0:
            fc_in = sub_feature
        else:
            fc_in = torch.cat([main_obs.to(self.device), sub_feature], dim=-1)
        return fc_in
        
    def save(self, path="../model/dicts.ptd"):
        dicts = {
                "actor": self.actor.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "deepSet": self.setTrans.state_dict(),
            }
        torch.save(dicts, path)

    def load(self, path="../model/dicts.ptd"):
        dicts = torch.load(path)
        self.actor.load_state_dict(dicts["actor"])
        self.actor_opt.load_state_dict(dicts["actor_opt"])
        self.critic.load_state_dict(dicts["critic"])
        self.critic_opt.load_state_dict(dicts["critic_opt"])
        self.target_actor.load_state_dict(dicts["target_actor"])
        self.target_critic.load_state_dict(dicts["target_critic"])
        self.setTrans.load_state_dict(dicts["deepSet"])

class setTransDDPG(setTransDDPG_V, DDPG):
    def __init__(self, main_obs_dim: int, sub_obs_dim: int, sub_feature_dim: int, 
                 action_dim: int, 
                 num_heads: int, encoder_depth: int, 
                 gamma=0.99, sigma=0.1, tau=0.005, 
                 actor_hiddens: typing.List[int] = [128] * 5, 
                 critic_hiddens: typing.List[int] = [128] * 5, 
                 encoder_fc_hiddens: typing.List[int] = [128] * 2, 
                 initial_fc_hiddens: typing.List[int] | None = None, 
                 initial_fc_output: int | None = None, 
                 pma_fc_hiddens: typing.List[int] = [128] * 2, 
                 pma_mab_fc_hiddens: typing.List[int] = [128] * 2, 
                 action_upper_bounds=None, 
                 action_lower_bounds=None, 
                 actor_lr=0.00001, 
                 critic_lr=0.0001, 
                 device=None) -> None:
        super().__init__(main_obs_dim, sub_obs_dim, sub_feature_dim, 
                         action_dim, num_heads, encoder_depth, 
                         gamma, sigma, tau, 
                         actor_hiddens, critic_hiddens, encoder_fc_hiddens, 
                         initial_fc_hiddens, initial_fc_output, pma_fc_hiddens, pma_mab_fc_hiddens, 
                         action_upper_bounds, action_lower_bounds, actor_lr, critic_lr, device)
        
    def _init_critic(self, hiddens, lr):
        return DDPG._init_critic(self, hiddens, lr)
    
    def _critic(self, main_obs: torch.Tensor, sub_seq: torch.Tensor | typing.Iterable[torch.Tensor], actions:torch.Tensor,
                target=False, permute=None):
        critic = self.target_critic if target else self.critic
        obs = self.get_fc_input(main_obs, sub_seq, target, permute)
        Q = critic(obs, actions)
        return Q
    
    def tree_act(self, main_state:torch.Tensor, sub_state:torch.Tensor, prop, 
                 search_step:int, search_population:int,
                 max_gen=10, select_explore_eps=0.2):
        if main_state.dim()==0:
            main_state = main_state.unsqueeze(0)
        if main_state.shape[0]>1:
            raise ValueError("only support single observation")
        state = (main_state, sub_state)
        obs = prop.getObss(main_state, sub_state)
        tree = searchTree.from_data(state, obs, max_gen=max_gen, gamma=self.gamma, select_explore_eps=select_explore_eps, device=self.device)
        tree.root.V_target = torch.zeros((1,1), device=self.device) # dummy value for select
        for _ in range(search_step):
            selected = tree.select(1, "value")[0]
            parents = [selected]*search_population
            main_state = selected.state[0]
            main_states = torch.cat([main_state]*search_population, dim=0)
            sub_state = selected.state[1]
            main_obss, sub_obs = prop.getObss(main_states, sub_state)
            _, actions = self.act(main_obss, sub_obs)
            actions[1:,:] = self.actor.uniSample(search_population-1)
            states, rewards, dones, obss = prop.propagate(main_states, sub_state, actions)
            Q_critics = self._critic(main_obss, sub_obs, actions).detach()

            states = list(states)
            states[0] = states[0].unsqueeze(1)
            states[1] = torch.stack([states[1]]*search_population, dim=0)
            states = list(zip(*states))
            obss = list(obss)
            obss[0] = obss[0].unsqueeze(1)
            obss = list(zip(*obss))
            
            tree.extend(states, obss, actions.unsqueeze(1), rewards.unsqueeze(1), dones.unsqueeze(1), parents=parents, Q_critics=Q_critics.unsqueeze(1))
            tree.backupQ()
        return tree.best_action()
    
    def MPC(self, sp0:torch.Tensor, sd0:torch.Tensor, prop, horizon=10, opt_step=10):
        '''
            Model Predictive Control.
            returns: `actions`, `actor_loss.item()`.
        '''
        _OU = self._OU
        self._OU = False
        batch_size = sp0.shape[0]
        if horizon>0:
            for _ in range(opt_step):
                Rewards = torch.zeros((horizon, batch_size), device=self.device)
                truncated_Q = torch.zeros(batch_size, device=self.device)
                Q = torch.zeros((horizon, batch_size), device=self.device)
                done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                truncated_steps = (horizon-1)*torch.ones(batch_size, dtype=torch.int32, device=self.device)
                sp, sd = sp0.clone(), sd0.clone()
                op, od = prop.getObss(sp, sd, batch_debris_obss=True)
                for i in range(horizon):
                    _, actions = self.act(op.detach(), od.detach(), with_OU_noise=False)
                    (sp, sd), rewards, dones, (op, od) = prop.propagate(sp, sd, actions, new_debris=True, batch_debris_obss=True, require_grad=True)
                    Rewards[i, ~done_flags] = rewards[~done_flags]
                    now_done = dones&~done_flags
                    truncated_Q[now_done] = self._critic(op[now_done], od[now_done], actions).squeeze(-1)
                    truncated_steps[now_done] = i
                    done_flags = done_flags|dones
                truncated_Q[~done_flags] = self._critic(op[~done_flags], od[~done_flags], actions).squeeze(-1)
                for i in range(horizon-1, -1, -1):
                    if (i>truncated_steps).all():
                        continue
                    trunc_flags = i==truncated_steps
                    disct_flags = i< truncated_steps
                    Q[i, trunc_flags] = Rewards[i, trunc_flags] + self.gamma*truncated_Q[trunc_flags]
                    Q[i, disct_flags] = Rewards[i, disct_flags] + self.gamma*Q[i, disct_flags]
                actor_loss = -Q[0].mean()
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()
            op, od = prop.getObss(sp0, sd0, batch_debris_obss=True)
            actions, _ = self.act(op, od, permute=False, with_OU_noise=False)
        else:
            op, od = prop.getObss(sp0, sd0, batch_debris_obss=True)
            _, actions = self.act(op.detach(), od.detach(), with_OU_noise=False)
            Q = self._critic(op, od, actions)
            actor_loss = -Q.mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        actor_loss = actor_loss.item()
        self._OU = _OU
        return actions, actor_loss
    
    def update(self, trans_dict, n_update=1, mode="mc"):
        if mode not in ("mc", "td"):
            raise ValueError("mode must be either `mc` or `td`")
        main_obss = torch.stack(trans_dict["primal_obss"]).to(self.device)
        sub_obss = trans_dict["debris_obss"]
        actions = torch.stack(trans_dict["actions"]).to(self.device)
        batch_size = main_obss.shape[0]

        for _ in range(n_update):
            if mode=="td":
                next_main_obss = torch.stack(trans_dict["next_primal_obss"]).to(self.device)
                next_sub_obss = trans_dict["next_debris_obss"]
                actions = torch.stack(trans_dict["actions"]).to(self.device)
                rewards = torch.stack(trans_dict["rewards"]).to(self.device).reshape((-1, 1))
                dones = torch.stack(trans_dict["dones"]).to(self.device).reshape((-1, 1))
                next_actions, _ = self.act(next_main_obss, next_sub_obss, target=True, with_OU_noise=False)
                next_q_values = self._critic(next_main_obss, next_sub_obss, next_actions, target=True).detach()
                q_targets = rewards + self.gamma*next_q_values*(~dones)
                critic_loss = F.mse_loss(self._critic(main_obss, sub_obss, actions), q_targets)
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()
                critic_loss = critic_loss.item()
            elif mode=="mc":
                q_targets = torch.stack(trans_dict["target_Q"]).to(self.device).reshape((-1, 1))
                critic_loss = torch.mean(F.mse_loss(self._critic(main_obss, sub_obss, actions), q_targets.detach()))
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()
                critic_loss = critic_loss.item()

            actions, _ = self.act(main_obss, sub_obss, with_OU_noise=False)
            Q = self._critic(main_obss, sub_obss, actions)
            actor_loss = -Q.mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            actor_loss = actor_loss.item()

            partial_loss = None

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

        return critic_loss, actor_loss, partial_loss
    
class setTransDDPG_L(setTransDDPG):
    def __init__(self, main_obs_dim: int, sub_obs_dim: int, sub_feature_dim: int, action_dim: int, 
                 num_heads: int, encoder_depth: int, 
                 gamma=0.99, sigma=0.1, tau=0.005, 
                 encoder_fc_hiddens: typing.List[int] = [128] * 2, 
                 initial_fc_hiddens: typing.List[int] | None = None, 
                 initial_fc_output: int | None = None, 
                 pma_fc_hiddens: typing.List[int] = [128] * 2, 
                 pma_mab_fc_hiddens: typing.List[int] = [128] * 2, 
                 action_upper_bounds=None, action_lower_bounds=None, 
                 actor_lr=0.00001, critic_lr=0.0001, device=None) -> None:
        actor_hiddens, critic_hiddens = None, None
        super().__init__(main_obs_dim, sub_obs_dim, sub_feature_dim, action_dim, num_heads, encoder_depth, gamma, sigma, tau, actor_hiddens, critic_hiddens, encoder_fc_hiddens, initial_fc_hiddens, initial_fc_output, pma_fc_hiddens, pma_mab_fc_hiddens, action_upper_bounds, action_lower_bounds, actor_lr, critic_lr, device)

    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        self.actor = linearFeedback(self.fc_in_dim, self.action_dim, upper_bounds=upper_bounds, lower_bounds=lower_bounds).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.target_actor = linearFeedback(self.fc_in_dim, self.action_dim, upper_bounds=upper_bounds, lower_bounds=lower_bounds).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.modules += [self.actor, self.target_actor]
    
    def _init_critic(self, hiddens, lr):
        self.critic = quadraticFunc(self.fc_in_dim+self.action_dim).to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.target_critic = quadraticFunc(self.fc_in_dim+self.action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.modules += [self.critic, self.target_critic]

    def _init_setTrans(self,
                       num_heads, encoder_fc_hiddens, encoder_depth,
                       initial_fc_hiddens, initial_fc_output,
                       pma_fc_hiddens, pma_mab_fc_hiddens, lr):
        self.setTrans = setTransformer(n_feature=self.sub_obs_dim,
                                       num_heads=num_heads,
                                       encoder_fc_hiddens=encoder_fc_hiddens,
                                       encoder_depth=encoder_depth,
                                       initial_fc_hiddens=initial_fc_hiddens,
                                       initial_fc_output=initial_fc_output,
                                       n_output=self.sub_feature_dim,
                                       pma_fc_hiddens=pma_fc_hiddens,
                                       pma_mab_fc_hiddens=pma_mab_fc_hiddens,
                                       ).to(self.device)
        self.critic_opt = torch.optim.Adam(list(self.critic.parameters())+list(self.setTrans.parameters()), lr=lr)
        self.target_setTrans = setTransformer(n_feature=self.sub_obs_dim,
                                              num_heads=num_heads,
                                              encoder_fc_hiddens=encoder_fc_hiddens,
                                              encoder_depth=encoder_depth,
                                              initial_fc_hiddens=initial_fc_hiddens,
                                              initial_fc_output=initial_fc_output,
                                              n_output=self.sub_feature_dim,
                                              pma_fc_hiddens=pma_fc_hiddens,
                                              pma_mab_fc_hiddens=pma_mab_fc_hiddens,
                                              ).to(self.device)
        self.target_setTrans.load_state_dict(self.setTrans.state_dict())
        self.modules += [self.setTrans, self.target_setTrans]

    def get_fc_input(self, main_obs: torch.Tensor, sub_seq: torch.Tensor | typing.Iterable[torch.Tensor], target=False, permute=None):
        '''
            `main_obs`: shape (batch_size, main_obs_dim)
            `sub_seq`: shape (batch_size, seq_len, sub_obs_dim)
        '''
        aggregator = self.target_setTrans if target else self.setTrans
        if isinstance(sub_seq, torch.Tensor): 
            sub_feature = aggregator(sub_seq.to(self.device)) # shape: (batch_size, sub_feature_dim)
        elif isinstance(sub_seq, typing.Iterable):
            batch_size = len(sub_seq)
            sub_feature = torch.zeros((batch_size, self.sub_feature_dim), device=self.device)
            stacked, indices = autils.var_len_seq_sort(sub_seq)
            for i in range(len(stacked)):
                sub_feature[indices[i]] = aggregator(stacked[i].to(self.device))
        if self.main_obs_dim==0:
            fc_in = sub_feature
        else:
            fc_in = torch.cat([main_obs.to(self.device), sub_feature], dim=-1)
        return fc_in
    
    def _critic(self, main_obs: torch.Tensor, sub_seq: torch.Tensor | typing.Iterable[torch.Tensor], actions:torch.Tensor,
                target=False, permute=None):
        critic = self.target_critic if target else self.critic
        obs = self.get_fc_input(main_obs, sub_seq, target, permute)
        Q = critic(torch.cat((obs, actions), dim=-1))
        return Q
    
    def update(self, trans_dict, n_update=1, mode="mc"):
        res = super().update(trans_dict, n_update, mode)
        self.soft_update(self.setTrans, self.target_setTrans)
        return res
    
class setTransSAC(setTransDDPG, SAC):
    def __init__(self, main_obs_dim: int, sub_obs_dim: int, sub_feature_dim: int, action_dim: int, 
                 num_heads: int, 
                 encoder_depth: int, 
                 action_bounds: typing.List[float], 
                 sigma_upper_bounds: typing.List[float],
                 gamma=0.99, 
                 tau=0.005, 
                 actor_hiddens: typing.List[int] = [128] * 5, 
                 critic_hiddens: typing.List[int] = [128] * 5, 
                 encoder_fc_hiddens: typing.List[int] = [128] * 2, 
                 initial_fc_hiddens: typing.List[int] | None = None, 
                 initial_fc_output: int | None = None, 
                 pma_fc_hiddens: typing.List[int] = [128] * 2, 
                 pma_mab_fc_hiddens: typing.List[int] = [128] * 2, 
                 target_entropy:float=None,
                 actor_lr=0.00001, critic_lr=0.0001, device=None) -> None:
        self.main_obs_dim = main_obs_dim
        self.sub_obs_dim = sub_obs_dim
        self.sub_feature_dim = sub_feature_dim
        self.fc_in_dim = main_obs_dim+sub_feature_dim
        SAC.__init__(self, self.fc_in_dim, action_dim, action_bounds, sigma_upper_bounds, 
                     gamma=gamma, tau=tau, 
                     actor_hiddens=actor_hiddens, critic_hiddens=critic_hiddens, 
                     target_entropy=target_entropy, actor_lr=actor_lr, critic_lr=critic_lr, device=device)
        self._init_setTrans(num_heads, encoder_fc_hiddens, encoder_depth, initial_fc_hiddens, initial_fc_output, pma_fc_hiddens, pma_mab_fc_hiddens, critic_lr)

    def _init_critic(self, hiddens, lr):
        return SAC._init_critic(self, hiddens, lr)

    def _init_actor(self, hiddens, action_bound, sigma_upper_bound, lr):
        return SAC._init_actor(self, hiddens, action_bound, sigma_upper_bound, lr)

    def _critic(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], actions, idx:typing.Literal[1,2]=1, target=False):
        if idx==1:
            if target:
                critic = self.target_critic1
            else:
                critic = self.critic1
        elif idx==2:
            if target:
                critic = self.target_critic2
            else:
                critic = self.critic2
        else:
            raise ValueError("idx must be either 1 or 2")
        obss = self.get_fc_input(main_obs, sub_seq)
        return critic(obss, actions)
    
    def act(self, main_obs, sub_seq):
        obss = self.get_fc_input(main_obs, sub_seq)
        return SAC.act(self, obss)

    def QMin(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], actions, target=False):
        obss = self.get_fc_input(main_obs, sub_seq)
        actions = actions.to(self.device)
        if target:
            q1 = self.target_critic1(obss, actions)
            q2 = self.target_critic2(obss, actions)
        else:
            q1 = self.critic1(obss, actions)
            q2 = self.critic2(obss, actions)
        return torch.min(q1, q2)
    
    def QEntropy(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], actions, entropy, target=False):
        return self.log_alpha.exp()*entropy+self.QMin(main_obs, sub_seq, actions, target=target)
    
    def V(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False):
        obss = self.get_fc_input(main_obs, sub_seq)
        actions, log_probs = self.actor.tanh_sample(obss)
        values = self.QEntropy(obss, actions, -log_probs, target=target)
        return values

    def update(self, trans_dict, n_update=1, mode="mc"):
        if mode not in ("mc", "td"):
            raise ValueError("mode must be either `mc` or `td`")
        main_obss = torch.stack(trans_dict["primal_obss"]).to(self.device)
        sub_obss = trans_dict["debris_obss"]
        actions = torch.stack(trans_dict["actions"]).to(self.device)

        for _ in range(n_update):
            if mode=="td":
                rewards = torch.stack(trans_dict["rewards"]).reshape((-1, 1)).to(self.device)
                next_main_obss = torch.stack(trans_dict["next_primal_obss"]).to(self.device)
                next_sub_obss = trans_dict["next_debris_obss"]
                dones = torch.stack(trans_dict["dones"]).reshape((-1, 1)).to(self.device)

                next_actions, next_log_probs = self.actor.tanh_sample(self.get_fc_input(next_main_obss, next_sub_obss))
                next_QEntropy = self.QEntropy(next_main_obss, next_sub_obss, next_actions, -next_log_probs, target=True)
                td_targets = rewards + self.gamma*next_QEntropy*(~dones)
                critic1_loss = F.mse_loss(self._critic(main_obss, sub_obss, actions, 1), td_targets.detach())
                critic2_loss = F.mse_loss(self._critic(main_obss, sub_obss, actions, 2), td_targets.detach())
                self.critic1_opt.zero_grad()
                critic1_loss.backward()
                self.critic1_opt.step()
                self.critic2_opt.zero_grad()
                critic2_loss.backward()
                self.critic2_opt.step()

            elif mode=="mc":
                q_targets = torch.stack(trans_dict["target_Q"]).to(self.device).reshape((-1, 1))
                critic1_loss = F.mse_loss(self._critic(main_obss, sub_obss, actions, 1), q_targets.detach())
                critic2_loss = F.mse_loss(self._critic(main_obss, sub_obss, actions, 2), q_targets.detach())
                self.critic1_opt.zero_grad()
                critic1_loss.backward()
                self.critic1_opt.step()
                self.critic2_opt.zero_grad()
                critic2_loss.backward()
                self.critic2_opt.step()

            new_actions, new_log_probs = self.actor.tanh_sample(self.get_fc_input(main_obss, sub_obss))
            new_QEntropy = self.QEntropy(main_obss, sub_obss, new_actions, -new_log_probs, target=False)
            actor_loss = -new_QEntropy.mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            alpha_loss = -torch.mean((new_log_probs+self.target_entropy).detach()*self.log_alpha.exp())
            self.log_alpha_opt.zero_grad()
            alpha_loss.backward()
            self.log_alpha_opt.step()

        self.soft_update(self.critic1, self.target_critic1)
        self.soft_update(self.critic2, self.target_critic2)

        critic_loss = (critic1_loss+critic2_loss)/2
        return critic_loss.item(), actor_loss.item(), alpha_loss.item()
    
class setTransPPO(setTransDDPG, PPO):
    def __init__(self, main_obs_dim: int, sub_obs_dim: int, sub_feature_dim: int, action_dim: int, 
                 num_heads: int, 
                 encoder_depth: int, 
                 gamma=0.99, 
                 lmd=0.9, 
                 clip_eps=0.2,
                 epochs=10,
                 action_upper_bounds:typing.List[float] = None,
                 action_lower_bounds:typing.List[float] = None,
                 actor_hiddens: typing.List[int] = [128] * 5, 
                 critic_hiddens: typing.List[int] = [128] * 5, 
                 encoder_fc_hiddens: typing.List[int] = [128] * 2, 
                 initial_fc_hiddens: typing.List[int] | None = None, 
                 initial_fc_output: int | None = None, 
                 pma_fc_hiddens: typing.List[int] = [128] * 2, 
                 pma_mab_fc_hiddens: typing.List[int] = [128] * 2,
                 actor_lr=0.00001, critic_lr=0.0001, device=None) -> None:
        self.main_obs_dim = main_obs_dim
        self.sub_obs_dim = sub_obs_dim
        self.sub_feature_dim = sub_feature_dim
        self.fc_in_dim = main_obs_dim+sub_feature_dim
        PPO.__init__(self, self.fc_in_dim, action_dim, 
                     gamma=gamma, lmd=lmd, clip_eps=clip_eps, epochs=epochs,
                     action_upper_bounds=action_upper_bounds, action_lower_bounds=action_lower_bounds, 
                     actor_hiddens=actor_hiddens, critic_hiddens=critic_hiddens, 
                     actor_lr=actor_lr, critic_lr=critic_lr, device=device)
        self._init_setTrans(num_heads, encoder_fc_hiddens, encoder_depth, initial_fc_hiddens, initial_fc_output, pma_fc_hiddens, pma_mab_fc_hiddens, critic_lr)

    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        return PPO._init_actor(self, hiddens, upper_bounds, lower_bounds, lr)

    def _init_critic(self, hiddens, lr):
        return PPO._init_critic(self, hiddens, lr)
    
    def _critic(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor]):
        obss = self.get_fc_input(main_obs, sub_seq)
        return self.critic(obss)
    
    def act(self, main_obs, sub_seq):
        obss = self.get_fc_input(main_obs, sub_seq)
        return PPO.act(self, obss)
    
    def update(self, trans_dict:dict):
        trans_dict = D.cut_dict(trans_dict)
        main_obss = torch.stack(trans_dict["primal_obss"]).to(self.device)
        sub_obss = trans_dict["debris_obss"]
        actions = torch.stack(trans_dict["actions"]).to(self.device)
        dones = torch.stack(trans_dict["dones"]).view((-1,1)).to(self.device)
        rewards = torch.stack(trans_dict["rewards"]).view((-1,1)).to(self.device)
        next_main_obss = torch.stack(trans_dict["next_primal_obss"]).to(self.device)
        next_sub_obss = trans_dict["next_debris_obss"]

        obss = self.get_fc_input(main_obss, sub_obss)
        next_obss = self.get_fc_input(next_main_obss, next_sub_obss)
        
        td_targets = rewards + self.gamma*self.critic(next_obss)*(~dones)

        td_deltas = td_targets - self.critic(obss)
        advantage = utils.compute_advantage(self.gamma, self.lmd, td_deltas)

        old_output = self.actor(obss).detach()
        action_dists = self.actor.distribution(old_output)
        old_log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True).detach()

        for _ in range(self.epochs):
            obss = self.get_fc_input(main_obss, sub_obss)
            critic_loss = F.mse_loss(self.critic(obss), td_targets.detach())
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            obss = self.get_fc_input(main_obss, sub_obss)
            output = self.actor(obss)
            action_dists = self.actor.distribution(output)
            log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

        return critic_loss.item(), actor_loss.item()
    
class setTransPPO2(setTransPPO):
    def __init__(self, main_obs_dim: int, sub_obs_dim: int, sub_feature_dim: int, action_dim: int,
                 num_heads: int,
                 encoder_depth: int,
                 gamma=0.99,
                 lmd=0.9,
                 clip_eps=0.2,
                 epochs=10,
                 action_upper_bounds:typing.List[float] = None,
                 action_lower_bounds:typing.List[float] = None,
                 actor_hiddens: typing.List[int] = [128] * 5,
                 critic_hiddens: typing.List[int] = [128] * 5,
                 encoder_fc_hiddens: typing.List[int] = [128] * 2,
                 initial_fc_hiddens: typing.List[int] | None = None,
                 initial_fc_output: int | None = None,
                 pma_fc_hiddens: typing.List[int] = [128] * 2,
                 pma_mab_fc_hiddens: typing.List[int] = [128] * 2,
                 actor_lr=0.00001, critic_lr=0.0001, device=None) -> None:
        self.critic_lr0 = critic_lr
        self.actor_lr0 = actor_lr
        super().__init__(main_obs_dim, sub_obs_dim, sub_feature_dim, action_dim,
                 num_heads,
                 encoder_depth,
                 gamma=gamma,
                 lmd=lmd,
                 clip_eps=clip_eps,
                 epochs=epochs,
                 action_upper_bounds=action_upper_bounds,
                 action_lower_bounds=action_lower_bounds,
                 actor_hiddens=actor_hiddens,
                 critic_hiddens=critic_hiddens,
                 encoder_fc_hiddens=encoder_fc_hiddens,
                 initial_fc_hiddens=initial_fc_hiddens,
                 initial_fc_output=initial_fc_output,
                 pma_fc_hiddens=pma_fc_hiddens,
                 pma_mab_fc_hiddens=pma_mab_fc_hiddens,
                 actor_lr=actor_lr,
                 critic_lr=critic_lr,
                 device=device)
        
    def _init_setTrans(self, num_heads, encoder_fc_hiddens, encoder_depth, initial_fc_hiddens, initial_fc_output, pma_fc_hiddens, pma_mab_fc_hiddens, lr):
        self.setTrans = None
        self.setTransC = setTransformer(n_feature=self.sub_obs_dim,
                                        num_heads=num_heads,
                                        encoder_fc_hiddens=encoder_fc_hiddens,
                                        encoder_depth=encoder_depth,
                                        initial_fc_hiddens=initial_fc_hiddens,
                                        initial_fc_output=initial_fc_output,
                                        n_output=self.sub_feature_dim,
                                        pma_fc_hiddens=pma_fc_hiddens,
                                        pma_mab_fc_hiddens=pma_mab_fc_hiddens,
                                        ).to(self.device)
        self.critic_opt = torch.optim.Adam(list(self.critic.parameters())+list(self.setTransC.parameters()), lr=self.critic_lr0)
        self.modules.append(self.setTransC)

        self.setTransA = setTransformer(n_feature=self.sub_obs_dim,
                                        num_heads=num_heads,
                                        encoder_fc_hiddens=encoder_fc_hiddens,
                                        encoder_depth=encoder_depth,
                                        initial_fc_hiddens=initial_fc_hiddens,
                                        initial_fc_output=initial_fc_output,
                                        n_output=self.sub_feature_dim,
                                        pma_fc_hiddens=pma_fc_hiddens,
                                        pma_mab_fc_hiddens=pma_mab_fc_hiddens,
                                        ).to(self.device)
        self.actor_opt = torch.optim.Adam(list(self.actor.parameters())+list(self.setTransA.parameters()), lr=self.actor_lr0)
        self.modules.append(self.setTransA)

    def get_fc_input(self, main_obs: torch.Tensor, sub_seq: torch.Tensor | typing.Iterable[torch.Tensor], idx:typing.Literal['c','a']='c'):
        '''
            `main_obs`: shape (batch_size, main_obs_dim)
            `sub_seq`: shape (batch_size, seq_len, sub_obs_dim)
        '''
        aggregator = self.setTransC if idx=='c' else self.setTransA
        if isinstance(sub_seq, torch.Tensor): 
            sub_feature = aggregator(sub_seq.to(self.device)) # shape: (batch_size, sub_feature_dim)
        elif isinstance(sub_seq, typing.Iterable):
            batch_size = len(sub_seq)
            sub_feature = torch.zeros((batch_size, self.sub_feature_dim), device=self.device)
            stacked, indices = autils.var_len_seq_sort(sub_seq)
            for i in range(len(stacked)):
                sub_feature[indices[i]] = aggregator(stacked[i].to(self.device))
        if self.main_obs_dim==0:
            fc_in = sub_feature
        else:
            fc_in = torch.cat([main_obs.to(self.device), sub_feature], dim=-1)
        return fc_in

    def _critic(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor]):
        obss = self.get_fc_input(main_obs, sub_seq, idx=1)
        return self.critic(obss)
    
    def act(self, main_obs, sub_seq):
        obss = self.get_fc_input(main_obs, sub_seq, idx=2)
        return PPO.act(self, obss)
    
    def update(self, trans_dict:dict):
        trans_dict = D.cut_dict(trans_dict)
        main_obss = torch.stack(trans_dict["primal_obss"]).to(self.device)
        sub_obss = trans_dict["debris_obss"]
        actions = torch.stack(trans_dict["actions"]).to(self.device)
        dones = torch.stack(trans_dict["dones"]).view((-1,1)).to(self.device)
        rewards = torch.stack(trans_dict["rewards"]).view((-1,1)).to(self.device)
        next_main_obss = torch.stack(trans_dict["next_primal_obss"]).to(self.device)
        next_sub_obss = trans_dict["next_debris_obss"]

        obss = self.get_fc_input(main_obss, sub_obss, idx='c')
        next_obss = self.get_fc_input(next_main_obss, next_sub_obss, idx='c')
        td_targets = rewards + self.gamma*self.critic(next_obss)*(~dones)
        td_deltas = td_targets - self.critic(obss)
        advantage = utils.compute_advantage(self.gamma, self.lmd, td_deltas)

        obss = self.get_fc_input(main_obss, sub_obss, idx='a')
        old_output = self.actor(obss).detach()
        action_dists = self.actor.distribution(old_output)
        old_log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True).detach()

        for _ in range(self.epochs):
            obss = self.get_fc_input(main_obss, sub_obss, idx='c')
            critic_loss = F.mse_loss(self.critic(obss), td_targets.detach())
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            obss = self.get_fc_input(main_obss, sub_obss, idx='a')
            output = self.actor(obss)
            action_dists = self.actor.distribution(output)
            log_probs = action_dists.log_prob(actions).sum(dim=1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

        return critic_loss.item(), actor_loss.item()