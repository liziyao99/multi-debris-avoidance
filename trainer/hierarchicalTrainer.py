import typing
import torch
import numpy as np
from rich.progress import Progress

from env.propagators.hierarchicalPropagator import H2Propagator
from data.buffer import replayBuffer
import data.dicts as D
import agent.agent as A
import agent.hierarchicalAgent as HA

class H2Trainer:
    def __init__(self, 
                 prop:H2Propagator,
                 hAgent:HA.hierarchicalAgent,
                 buffer:replayBuffer=None,
                 loss_keys:typing.List[str]=[]) -> None:
        self.prop = prop
        self.hAgent = hAgent
        self.buffer = buffer
        self.loss_keys = loss_keys

    @property
    def device(self):
        return self.hAgent.device

    def h2Sim(self, 
              states0:torch.Tensor, 
              h1actions:torch.Tensor=None, 
              h1noise=True, 
              prop_with_grad=True):
        '''
            returns: `h2transdict`, `next_states`, `h2rewards`(with grad if `prop_with_grad`), `h1actions`, `h1rewards`, `dones`, `terminal_rewards`.
        '''
        batch_size = states0.shape[0]
        trans_dict = D.init_transDictBatch(self.prop.h2_step, batch_size, self.prop.state_dim, self.prop.obs_dim, self.prop.h2_action_dim,
                                           struct="torch", device=self.device)
        step = 0
        done = False
        done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        obss = self.prop.getObss(states0, require_grad=True)
        
        if h1actions is None:
            _, h1actions = self.hAgent[0].act(obss)
        if h1noise:
            noise = self.hAgent[0].actor.uniSample(batch_size)/10
            h1actions = h1actions + noise
        h1actions = self.hAgent[0].actor.clip(h1actions)
        targets = (h1actions+states0[:,:6]).detach() # increment control
        
        states = states0.clone()
        next_states = states0.clone()
        h1rewards = torch.zeros(batch_size, device=self.device)
        h2rewards = torch.zeros(batch_size, device=self.device, requires_grad=prop_with_grad)
        terminal_rewards = torch.zeros(batch_size, device=self.device)
        actions_penalty = []
        while not done and step<self.prop.h2_step:
            trans_dict["states"][step,...] = states[...].detach()
            trans_dict["obss"][step,...] = obss[...].detach()
            _, actions = self.hAgent.act(obss, h=1, higher_action=h1actions)
            actions_penalty.append(-torch.norm(actions, dim=-1))
            states, obss, h1rs, h2rs, dones, trs = self.prop.propagate(states, targets, actions, require_grad=prop_with_grad)
            terminal_rewards = torch.where(done_flags, terminal_rewards, trs)
            next_states = torch.where(done_flags.unsqueeze(1), next_states, states)
            done_flags = done_flags | dones
            trans_dict["actions"][step,...] = actions[...].detach()
            trans_dict["next_states"][step,...] = states[...].detach()
            trans_dict["next_obss"][step,...] = obss[...].detach()
            trans_dict["rewards"][step,...] = h2rs[...].detach()
            trans_dict["dones"][step,...] = dones[...].detach()
            h1rewards = h1rewards + h1rs*(~done_flags)
            h2rewards = h2rs
            done = torch.all(dones)
            step += 1
        actions_penalty = torch.stack(actions_penalty, dim=0)
        actions_penalty = torch.sum(actions_penalty, dim=0)
        h2rewards = h2rewards + actions_penalty
        return trans_dict, next_states, h2rewards, h1actions, h1rewards, done_flags, terminal_rewards
    
    def h1Sim(self, states0:torch.Tensor=None, h1_explore_eps=0., states_num=1, train_h2a=False, prop_with_grad=False):
        if states0 is None:
            states0 = self.prop.randomInitStates(states_num)
        prop_with_grad = prop_with_grad or train_h2a
        batch_size = states0.shape[0]
        h1td = D.init_transDictBatch(self.prop.h1_step, batch_size, self.prop.state_dim, self.prop.obs_dim, self.prop.h1_action_dim,
                                     items=("terminal_rewards",),
                                     struct="torch", device=self.device)
        h2Loss = []
        step = 0
        done = False
        done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        states = states0
        obss = self.prop.getObss(states)
        while not done and step<self.prop.h1_step:
            h1td["states"][step, ...] = states[...].detach()
            h1td["obss"][step, ...] = obss[...].detach()
            if np.random.rand()<h1_explore_eps:
                h1actions = self.hAgent[0].actor.uniSample(batch_size)
            else:
                h1actions = None
            _, states, h2rs, h1as, h1rs, dones, trs = self.h2Sim(states, h1actions, prop_with_grad=prop_with_grad)
            obss = self.prop.getObss(states)
            h1td["actions"][step, ...] = h1as[...].detach()
            h1td["next_states"][step, ...] = states[...].detach()
            h1td["next_obss"][step, ...] = obss[...].detach()
            h1td["rewards"][step, ...] = h1rs[...].detach()
            h1td["dones"][step, ...] = dones[...].detach()
            h1td["terminal_rewards"][step, ...] = trs[...].detach()

            h2loss = -h2rs.mean()
            h2Loss.append(h2loss)

            done_flags = done_flags | dones
            done = torch.all(done_flags)
            step += 1
        h2Loss = torch.mean(torch.stack(h2Loss))
        if train_h2a and prop_with_grad:
            self.hAgent[1].actor_opt.zero_grad()
            h2loss.backward()
            self.hAgent[1].actor_opt.step()
            h2loss = h2loss.detach()
        return h1td, h2Loss
        
    def train(self, epoch:int, episode:int, select_itr:int, select_size:int, explore_eps=0.5, batch_size=640):
        data_list = {
            "Q_loss": [],
            "mc_loss": [],
            "ddpg_loss": [],
            "h2_loss": [],
            "true_value": []
        }
        with Progress() as progress:
            task = progress.add_task("H2TreeTrainer training:", total=episode)
            for i in range(epoch):
                progress.tasks[task].description = "epoch {0} of {1}".format(i+1, epoch)
                progress.tasks[task].completed = 0
                for _ in range(episode):
                    trans_dict, h2loss, root_value = self.treeSim(select_itr, select_size, h1_explore_eps=explore_eps, train_h2a=True)
                    dicts = D.split_dict(trans_dict, batch_size)
                    _Q, _mc, _ddpg = [], [], []
                    for dict in dicts:
                        Q_loss, mc_loss, ddpg_loss = self.agent.h1update(dict)
                        _Q.append(Q_loss)
                        _mc.append(mc_loss)
                        _ddpg.append(ddpg_loss)
                    data_list["Q_loss"].append(np.mean(_Q))
                    data_list["mc_loss"].append(np.mean(_mc))
                    data_list["ddpg_loss"].append(np.mean(_ddpg))
                    data_list["h2_loss"].append(np.mean(h2loss))
                    data_list["true_value"].append(root_value)

                    progress.update(task, advance=1)
        return data_list
    
    def h2Pretrain(self, episode:int, horizon=None, states_num=256, h1=True):
        Loss = []
        h1actor = self.hAgent[0].actor if h1 else None
        with Progress() as progress:
            task = progress.add_task("H2TreeTrainer pretraining:", total=episode)
            for _ in range(episode):
                loss = self.prop.episodeTrack(self.hAgent[1].actor, h1actor, horizon=horizon, batch_size=states_num)
                self.hAgent[1].actor_opt.zero_grad()
                loss.backward()
                self.hAgent[1].actor_opt.step()
                Loss.append(loss.item())
                progress.update(task, advance=1)
        return Loss
    
    def offPolicySim(self, states0:torch.Tensor=None, h1_explore_eps=0., states_num=1, 
                     off_policy_train=True,
                     prop_with_grad=False):
        if states0 is None:
            states0 = self.prop.randomInitStates(states_num)
        batch_size = states0.shape[0]
        h1td = D.init_transDictBatch(self.prop.h1_step, batch_size, self.prop.state_dim, self.prop.obs_dim, self.prop.h1_action_dim,
                                     items=("terminal_rewards",),
                                     struct="torch", device=self.device)
        h2Loss = []
        step = 0
        done = False
        done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        states = states0
        obss = self.prop.getObss(states)
        while not done and step<self.prop.h1_step:
            h1td["states"][step, ...] = states[...].detach()
            h1td["obss"][step, ...] = obss[...].detach()
            if np.random.rand()<h1_explore_eps:
                h1actions = self.hAgent[0].actor.uniSample(batch_size)
            else:
                h1actions = None
            _, states, h2rs, h1as, h1rs, dones, trs = self.h2Sim(states, h1actions, prop_with_grad=prop_with_grad)
            obss = self.prop.getObss(states)
            h1td["actions"][step, ...] = h1as[...].detach()
            h1td["next_states"][step, ...] = states[...].detach()
            h1td["next_obss"][step, ...] = obss[...].detach()
            h1td["rewards"][step, ...] = h1rs[...].detach()
            h1td["dones"][step, ...] = dones[...].detach()
            h1td["terminal_rewards"][step, ...] = trs[...].detach()

            if off_policy_train and self.buffer.size>self.buffer.minimal_size:
                self.hAgent[0].update(self.buffer.sample())

            h2loss = -h2rs.mean()
            if prop_with_grad:
                self.hAgent[1].actor_opt.zero_grad()
                h2loss.backward()
                self.hAgent[1].actor_opt.step()
                h2loss = h2loss.detach()
                states = states.detach()
            h2Loss.append(h2loss)

            done_flags = done_flags | dones
            done = torch.all(done_flags)
            step += 1
        return h1td, torch.stack(h2Loss).mean().item()
    
    def offPolicyTrain(self, n_epoch:int, n_episode:int, states_num=1, h1_explore_eps=0.4, prop_with_grad=False):
        keys = ["total_rewards", "h2_loss"] + self.loss_keys
        log_dict = dict(zip( keys, [[] for _ in range(len(keys))] ))
        with Progress() as progress:
            task = progress.add_task("epoch{0}".format(0), total=n_episode)
            for i in range(n_epoch):
                progress.tasks[task].description = "epoch {0} of {1}".format(i+1, n_epoch)
                progress.tasks[task].completed = 0
                for _ in range(n_episode):
                    trans_dict, h2_loss = self.offPolicySim(states_num=states_num, h1_explore_eps=h1_explore_eps, prop_with_grad=prop_with_grad)
                    tds = D.deBatch_dict(trans_dict)
                    for d in tds:
                        Loss = self.hAgent[0].update(d)
                        d = D.torch_dict(d, device="cpu")
                        self.buffer.from_dict(d)
                    total_reward = trans_dict["rewards"].sum().item()/states_num
                    log_dict["total_rewards"].append(total_reward)
                    log_dict["h2_loss"].append(h2_loss)
                    for i in range(len(self.loss_keys)):
                        if isinstance(Loss, tuple):
                            log_dict[self.loss_keys[i]].append(Loss[i])
                        elif isinstance(Loss, dict):
                            log_dict[self.loss_keys[i]].append(Loss[self.loss_keys[i]])
                    progress.update(task, advance=1)
        return log_dict
    
class thrustCWTrainer(H2Trainer):
    def __init__(self, prop: H2Propagator, hAgent: HA.hierarchicalAgent, buffer: replayBuffer = None, loss_keys: typing.List[str] = []) -> None:
        super().__init__(prop, hAgent, buffer, loss_keys)

    def h2Sim(self, 
              states0:torch.Tensor, 
              h1actions:torch.Tensor=None, 
              h1noise=False, 
              prop_with_grad=True):
        '''
            returns: `h2transdict`, `next_states`, `h2rewards`(with grad if `prop_with_grad`), `h1actions`, `h1rewards`, `dones`, `terminal_rewards`.
        '''
        batch_size = states0.shape[0]
        trans_dict = D.init_transDictBatch(self.prop.h2_step, batch_size, self.prop.state_dim, self.prop.obs_dim, self.prop.h2_action_dim,
                                           struct="torch", device=self.device)
        step = 0
        done = False
        done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        obss = self.prop.getObss(states0, require_grad=True)
        
        if h1actions is None:
            _, h1actions = self.hAgent[0].act(obss)
        if h1noise:
            noise = self.hAgent[0].actor.uniSample(batch_size)/10
            h1actions = h1actions + noise
        h1actions = self.hAgent[0].actor.clip(h1actions)
        thrust_todo = h1actions
        
        states = states0.clone()
        next_states = states0.clone()
        h1rewards = torch.zeros(batch_size, device=self.device)
        h2rewards = torch.zeros(batch_size, device=self.device, requires_grad=False)
        terminal_rewards = torch.zeros(batch_size, device=self.device)
        actions_penalty = []
        while not done and step<self.prop.h2_step:
            trans_dict["states"][step,...] = states[...].detach()
            trans_dict["obss"][step,...] = obss[...].detach()
            # _, actions = self.hAgent.act(obss, h=1, higher_action=h1actions)
            actions = self.hAgent[1].actor.clip(thrust_todo)
            actions_penalty.append(-torch.norm(actions, dim=-1))
            states, obss, h1rs, h2rs, dones, trs = self.prop.propagate(states, thrust_todo, actions, require_grad=False)
            thrust_todo = thrust_todo - actions*self.prop.dt
            terminal_rewards = torch.where(done_flags, terminal_rewards, trs)
            next_states = torch.where(done_flags.unsqueeze(1), next_states, states)
            done_flags = done_flags | dones
            trans_dict["actions"][step,...] = actions[...].detach()
            trans_dict["next_states"][step,...] = states[...].detach()
            trans_dict["next_obss"][step,...] = obss[...].detach()
            trans_dict["rewards"][step,...] = h2rs[...].detach()
            trans_dict["dones"][step,...] = dones[...].detach()
            h1rewards = h1rewards + h1rs*(~done_flags)
            h2rewards = h2rs
            done = torch.all(dones)
            step += 1
        actions_penalty = torch.stack(actions_penalty, dim=0)
        actions_penalty = torch.sum(actions_penalty, dim=0)
        h2rewards = h2rewards + actions_penalty
        return trans_dict, next_states, h2rewards, h1actions, h1rewards, done_flags, terminal_rewards
