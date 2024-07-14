from agent.net import *
import data.dicts as D

import typing
import numpy as np
import torch
import torch.nn.functional as F
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
    

class PPOClipAgent(boundedRlAgent):
    def __init__(self, 
                 obs_dim: int = 6, 
                 action_dim: int = 3, 
                 actor_hiddens: typing.List[int] = [128] * 5, 
                 critic_hiddens: typing.List[int] = [128] * 5, 
                 action_upper_bound=0.2, 
                 action_lower_bound=-0.2, 
                 actor_lr=1e-5, 
                 critic_lr=1e-4, 
                 gamma=0.99,
                 lmd=0.99,
                 clip_eps=0.2,
                 epochs=10,
                 device=None) -> None:
        bound = (action_upper_bound-action_lower_bound)/2
        upper_bounds = [action_upper_bound]*action_dim + [bound/3]*action_dim
        lower_bounds = [action_lower_bound]*action_dim + [bound/30]*action_dim
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
        dict_torch = {}
        for key in trans_dict.keys():
            dict_torch[key] = torch.from_numpy(trans_dict[key]).to(self.device)

        td_targets = dict_torch["rewards"].unsqueeze(dim=1) + self.gamma*self.critic(dict_torch["next_obss"])*(1-dict_torch["dones"].to(torch.float32).unsqueeze(dim=1))
        td_deltas = td_targets - self.critic(dict_torch["obss"])

        # advantage = utils.compute_advantage(self.gamma, self.lmd, td_deltas.cpu()).to(self.device)
        advantage = td_deltas.detach()
        old_output = self.actor(dict_torch["obss"]).detach()
        action_dists = self.actor.distribution(old_output)
        old_log_probs = action_dists.log_prob(dict_torch["actions"]).sum(dim=1, keepdim=True)

        for _ in range(self.epochs):
            output = self.actor(dict_torch["obss"])
            action_dists = self.actor.distribution(output)
            log_probs = action_dists.log_prob(dict_torch["actions"]).sum(dim=1, keepdim=True)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(dict_torch["obss"]), td_targets.detach()))
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_opt.step()
            self.critic_opt.step()

        return actor_loss.item(), critic_loss.item()
    

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

    def act(self, obs:torch.Tensor):
        '''
            returns:
                `output`: output of actor.
                `sample`: sampled of `output` if actor is random, else `output`.
        '''
        obs = obs.to(self.device)
        output = self.actor(obs)
        noise = torch.randn_like(output)*self.sigma
        sample = output + noise
        sample = self.actor.clip(sample)
        return output, sample

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data*(1.0-self.tau) + param.data*self.tau)

    def update(self, trans_dict):
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
        super().__init__(self.fc_in_dim, action_dim, gamma, sigma, tau, actor_hiddens, critic_hiddens, action_upper_bounds, action_lower_bounds, actor_lr, critic_lr, device)
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
            permute=None):
        '''
            returns:
                `output`: output of actor.
                `sample`: sampled of `output` if actor is random, else `output`.
        '''
        actor = self.target_actor if target else self.actor
        obs = self.get_fc_input(main_obs, sub_seq, target=target, permute=permute)
        output = actor(obs)
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
    
    def _critic(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False):
        critic = self.target_critic if target else self.critic
        obs = self.get_fc_input(main_obs, sub_seq, target=target)
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
            V_critics = self._critic(obss[0], obss[1])

            states = list(states)
            states[0] = states[0].unsqueeze(1)
            states[1] = torch.stack([states[1]]*search_population, dim=0)
            states = list(zip(*states))
            obss = list(obss)
            obss[0] = obss[0].unsqueeze(1)
            obss = list(zip(*obss))
            
            tree.extend(states, obss, actions.unsqueeze(1), rewards.unsqueeze(1), dones.unsqueeze(1), V_critics.unsqueeze(1), parents=parents)
            tree.backup()
        return tree.best_action()
    
    def update(self, trans_dict, prop, n_step:int=1, n_update=10):
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

            Rewards = torch.zeros((n_step, batch_size), device=self.device)
            truncated_values = torch.zeros(batch_size, device=self.device)
            Values = torch.zeros((n_step, batch_size), device=self.device)
            done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            truncated_steps = (n_step-1)*torch.ones(batch_size, dtype=torch.int32, device=self.device)
            sp = torch.stack(trans_dict["primal_states"]).to(self.device)
            # sd = trans_dict["debris_states"] # NOTICE: list of tensor with different length
            sd = torch.cat(trans_dict["debris_states"], dim=0).to(self.device) # shape: (sum(n_debris), 6)
            nd = np.random.randint(1, prop.max_n_debris+1)
            indices = np.random.choice(sd.shape[0], size=(min(nd, sd.shape[0]),), replace=False)
            # sd = prop.randomDebrisStates(nd)
            sd = sd[indices]
            op, od = prop.getObss(sp, sd, batch_debris_obss=True)
            for i in range(n_step):
                _, actions = self.act(op, od)
                (sp, sd), rewards, dones, (op, od) = prop.propagate(sp, sd, actions, new_debris=True, batch_debris_obss=True, require_grad=True)
                Rewards[i, ~done_flags] = rewards[~done_flags]
                now_done = dones&~done_flags
                truncated_values[now_done] = self._critic(op[now_done], od[now_done], target=True).squeeze(-1)
                truncated_steps[now_done] = i
                done_flags = done_flags|dones
            truncated_values[~done_flags] = self._critic(op[~done_flags], od[~done_flags], target=True).squeeze(-1)
            for i in range(n_step-1, -1, -1):
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

        return critic_loss, actor_loss, partial_loss
    

class dualLstmDDPG(lstmDDPG_V, dualRlAgent):
    def __init__(self,
                 main_obs_dim: int,
                 sub_obs_dim: int,
                 sub_feature_dim: int,
                 lstm_num_layers: int,
                 action_dim: int,
                 gamma = 0.99,
                 sigma = 0.1,
                 tau = 0.005,
                 dual_hiddens: typing.List[int] = [128] * 5,
                 action_upper_bounds=None,
                 action_lower_bounds=None,
                 lr=0.0001,
                 partial_dim = None,
                 partial_hiddens = [128],
                 partial_lr=0.0001,
                 device=None) -> None:
        lstmDDPG.__init__(self, main_obs_dim=main_obs_dim, sub_obs_dim=sub_obs_dim, sub_feature_dim=sub_feature_dim, 
                          lstm_num_layers=lstm_num_layers, action_dim=action_dim, 
                          gamma=gamma, sigma=sigma, tau=tau, 
                          actor_hiddens=None, critic_hiddens=None, 
                          partial_dim=partial_dim, partial_hiddens=partial_hiddens, partial_lr=partial_lr,
                          device=device)
        
        if action_upper_bounds is None:
            action_upper_bounds = [ torch.inf]*action_dim
        if action_lower_bounds is None:
            action_lower_bounds = [-torch.inf]*action_dim
        for _b in (action_upper_bounds, action_lower_bounds):
            if isinstance(_b, torch.Tensor):
                _b = list(_b.flatten())
        output_upper_bounds = action_upper_bounds + [ torch.inf]
        output_lower_bounds = action_lower_bounds + [-torch.inf]
        self._init_dual(dual_hiddens, output_upper_bounds, output_lower_bounds, lr)
        
    def _init_actor(self, hiddens, upper_bounds, lower_bounds, lr):
        self.actor = torch.nn.Module()
        self.target_actor = torch.nn.Module()

    def _init_critic(self, hiddens, lr):
        self.critic = torch.nn.Module()
        self.target_critic = torch.nn.Module()

    def _init_dual(self, hiddens, output_upper_bounds, output_lower_bounds, lr):
        self.dual = dualNet(self.obs_dim, self.action_dim, 1, hiddens, output_upper_bounds, output_lower_bounds).to(self.device)
        self.target_dual = dualNet(self.obs_dim, self.action_dim, 1, hiddens, output_upper_bounds, output_lower_bounds).to(self.device)
        params = list(self.dual.parameters())+list(self.lstm.parameters())
        self.dual_opt = torch.optim.Adam(params, lr=lr)
        self.actor_opt = torch.optim.Adam(params, lr=lr)
        self.critic_opt = torch.optim.Adam(params, lr=lr)
        self.modules += [self.dual, self.target_dual]

    def act(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False, 
            permute=None):
        '''
            returns:
                `output`: output of actor.
                `sample`: sampled of `output` if actor is random, else `output`.
        '''
        dual = self.target_dual if target else self.dual
        obs = self.get_fc_input(main_obs, sub_seq, target=target, permute=permute)
        output = dual(obs)
        output_a, _ = dual.split_output(output)
        noise = torch.randn_like(output_a)*self.sigma
        sample_a = output_a + noise
        sample_a = dual.clip(sample_a)
        return output_a, sample_a
    
    def _critic(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False):
        dual = self.target_dual if target else self.dual
        obs = self.get_fc_input(main_obs, sub_seq, target=target)
        output = dual(obs)
        _, output_v = dual.split_output(output)
        values = dual.sample_value(output_v)
        return values
    
    def a_c(self, main_obs:torch.Tensor, sub_seq:torch.Tensor|typing.Iterable[torch.Tensor], target=False):
        '''
            return sampled action and value.
        '''
        dual = self.target_dual if target else self.dual
        obs = self.get_fc_input(main_obs, sub_seq, target=target)
        output = dual(obs)
        return dual.sample(output)
    
    def update(self, trans_dict, prop, n_step:int=1, n_update=10):
        main_obss = torch.stack(trans_dict["primal_obss"]).to(self.device)
        sub_obss = trans_dict["debris_obss"]
        target_values = torch.stack(trans_dict["values"]).to(self.device).reshape((-1,1))
        batch_size = main_obss.shape[0]

        for _ in range(n_update):
            critic_loss = torch.mean(F.mse_loss(self._critic(main_obss, sub_obss), target_values.detach()))
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            Rewards = torch.zeros((n_step, batch_size), device=self.device)
            truncated_values = torch.zeros(batch_size, device=self.device)
            Values = torch.zeros((n_step, batch_size), device=self.device)
            done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            truncated_steps = (n_step-1)*torch.ones(batch_size, dtype=torch.int32, device=self.device)
            sp = torch.stack(trans_dict["primal_states"]).to(self.device)
            # sd = trans_dict["debris_states"] # NOTICE: list of tensor with different length
            sd = torch.cat(trans_dict["debris_states"], dim=0).to(self.device) # shape: (sum(n_debris), 6)
            nd = np.random.randint(1, prop.max_n_debris+1)
            indices = np.random.choice(sd.shape[0], size=(min(nd, sd.shape[0]),), replace=False)
            sd = sd[indices]
            op, od = prop.getObss(sp, sd, batch_debris_obss=True)
            for i in range(n_step):
                _, actions = self.act(op, od)
                (sp, sd), rewards, dones, (op, od) = prop.propagate(sp, sd, actions, new_debris=True, batch_debris_obss=True, require_grad=True)
                Rewards[i, ~done_flags] = rewards[~done_flags]
                now_done = dones&~done_flags
                truncated_values[now_done] = self._critic(op[now_done], od[now_done], target=True).squeeze(-1)
                truncated_steps[now_done] = i
                done_flags = done_flags|dones
            truncated_values[~done_flags] = self._critic(op[~done_flags], od[~done_flags], target=True).squeeze(-1)
            for i in range(n_step-1, -1, -1):
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

            # total_loss = critic_loss+actor_loss
            # self.dual_opt.zero_grad()
            # total_loss.backward()
            # self.dual_opt.step()

            if "partial_label" in trans_dict.keys():
                partial_out = self.partial_output(main_obss, sub_obss)
                partial_loss = F.mse_loss(partial_out, trans_dict["partial_label"])
                self.partial_opt.zero_grad()
                partial_loss.backward()
                self.partial_opt.step()
                partial_loss = partial_loss.item()
            else:
                partial_loss = None

        self.soft_update(self.dual, self.target_dual)
        self.soft_update(self.lstm, self.target_lstm)

        return critic_loss.item(), actor_loss.item(), partial_loss
    
    def save(self, path="../model/dicts.ptd"):
        dicts = {
                "dual": self.dual.state_dict(),
                "dual_opt": self.dual_opt.state_dict(),
                "target_dual": self.target_dual.state_dict(),
                "lstm": self.lstm.state_dict(),
            }
        torch.save(dicts, path)

    def load(self, path="../model/dicts.ptd"):
        dicts = torch.load(path)
        self.dual.load_state_dict(dicts["dual"])
        self.dual_opt.load_state_dict(dicts["dual_opt"])
        self.target_dual.load_state_dict(dicts["target_dual"])
        self.lstm.load_state_dict(dicts["lstm"])