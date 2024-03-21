from agent import agent
from env import env as ENV
from env import propagator
from env import propagatorT
from tree.tree import stateDict
import data.dicts as D
from data.buffer import replayBuffer
from plotting.dataplot import dataPlot


from rich.progress import Progress
import numpy as np
import torch
import torch.nn.functional as F

class dummyTrainer:
    def __init__(self, env:ENV.dummyEnv, agent:agent.rlAgent) -> None:
        self.env = env
        self.agent = agent
        self.buffer = replayBuffer(keys=("dummy",))

    def new_state(self) -> stateDict:
        sd = stateDict(self.env.state_dim, self.env.obs_dim, self.env.action_dim)
        state = self.propagator.randomInitStates(1)
        obs = self.env.propagator.getObss(state)
        sd.state[:] = state.flatten()
        sd.obs[:] = obs.flatten()
        return sd

    def reset_env(self):
        sd = self.new_state()
        self.env.reset(sd)

    def simulate(self):
        raise(NotImplementedError)

    def update(self, trans_dict:dict):
        '''
            returns: `actor_loss`, `critic_loss`.
        '''
        return self.agent.update(trans_dict)
    
    def train(self, n_epoch=10, n_episode=100):
        raise(NotImplementedError)

    def test(self, **kwargs):
        raise(NotImplementedError)
    
    @property
    def propagator(self):
        return self.env.propagator

class treeTrainer(dummyTrainer):
    def __init__(self, env:ENV.treeEnv, agent:agent.rlAgent, gamma=1., batch_size=1280) -> None:
        self.env = env
        '''
            env.treeEnv
        '''
        self.agent = agent
        self.env.tree.gamma = gamma # discount factor
        self.testEnv = ENV.singleEnv.from_propagator(env.propagator, env.tree.max_gen)

        self.plot = dataPlot(("true_values", "actor_loss", "critic_loss"))

        buffer_keys = list(D.BASIC_KEYS) + list(env.tree.item_keys)
        self.buffer = replayBuffer(keys=buffer_keys, capacity=100*batch_size, minimal_size=10*batch_size, batch_size=batch_size)
        self.batch_size = batch_size

    @property
    def tree(self):
        return self.env.tree

    def simulate(self, dicts_redundant=False, n_bufferTrain=0):
        self.reset_env()
        while not self.env.step(self.agent):
            if self.buffer.size >= n_bufferTrain:
                for _ in range(n_bufferTrain):
                    self.agent.update(self.buffer.sample(self.batch_size))
        dicts = self.tree.get_transDicts(redundant=dicts_redundant)
        return dicts

    def train(self, n_epoch=10, n_episode=100, n_bufferTrain=5):
        true_values = []
        actor_loss = []
        critic_loss = []
        with Progress() as progress:
            task = progress.add_task("epoch{0}".format(0), total=n_episode)
            for i in range(n_epoch):
                progress.tasks[task].description = "epoch {0} of {1}".format(i+1, n_epoch)
                progress.tasks[task].completed = 0
                for _ in range(n_episode):
                    al, cl = [], []
                    trans_dicts = self.simulate(dicts_redundant=False, n_bufferTrain=n_bufferTrain)
                    trans_dict = D.concat_dicts(trans_dicts)
                    self.buffer.from_dict(trans_dict)
                    dict_batches = D.batch_dict(trans_dict, batch_size=self.batch_size)
                    for d in dict_batches:
                        al_, cl_ = self.agent.update(d)
                        al.append(al_)
                        cl.append(cl_)
                    total_reward = trans_dicts[0]["rewards"].sum()
                    progress.update(task, advance=1)
                    true_values.append(total_reward)
                    actor_loss.append(np.mean(al))
                    critic_loss.append(np.mean(cl))
                self.agent.save("../model/check_point{0}.ptd".format(i))
                np.savez("../model/log.npz", 
                         true_values = np.array(true_values),
                         actor_loss = np.array(actor_loss),
                         critic_loss = np.array(critic_loss)
                        )
                self.plot.set_data((np.array(true_values),np.array(actor_loss),np.array(critic_loss)))
                self.plot.save_fig("../model/log.png")
                
    def test(self, decide_mode="time", t_max=0.02, g_max=5):
        '''
            args:
                `decide_mode`: `time`, `gen`, `determined`.
                `t_max`: max searching time each step, second.
                `pick_mode`: see `GST_0.pick`.
        '''
        trans_dict = D.init_transDict(self.tree.max_gen+1, self.env.state_dim, self.env.obs_dim, self.env.action_dim)
        sd = self.new_state()
        self.env.reset(sd)
        self.testEnv.reset(sd.state)
        done = False
        while not done:
            if decide_mode!="determined":
                sd = self.tree.decide(sd, self.agent, self.env.propagator, decide_mode=decide_mode, t_max=t_max, g_max=g_max)
                action = sd.action
            else: # determined
                state = self.testEnv.state.reshape((1,-1))
                obs = self.propagator.getObss(state)
                obs = torch.from_numpy(obs).to(self.agent.device)
                action = self.agent.nominal_act(obs).detach().cpu().numpy()
            transit = self.testEnv.step(action)
            done = transit[-1]
            self.testEnv.fill_dict(trans_dict, transit)
        total_rewards = trans_dict["rewards"].sum()
        return total_rewards, trans_dict
        

class singleTrainer(dummyTrainer):
    def __init__(self, env:ENV.singleEnv, agent:agent.rlAgent) -> None:
        self.env = env
        self.agent = agent

    @property
    def obs(self):
        state = np.expand_dims(self.env.state, axis=0)
        obs = self.env.propagator.getObss(state)
        return obs.flatten()
    
    def reset_env(self):
        sd = self.new_state()
        self.env.reset(sd.state)
     
    def simulate(self):
        trans_dict = D.init_transDict(self.env.max_stage+1, self.env.state_dim, self.env.obs_dim, self.env.action_dim,
                                          items=("advantages",))
        sd = self.new_state()
        self.env.reset(sd.state)
        done = False
        while not done:
            _, action = self.agent.act(torch.from_numpy(self.obs).unsqueeze(0))
            transit = self.env.step(action.detach().cpu().numpy())
            done = transit[-1]
            self.env.fill_dict(trans_dict, transit)
        self.env.cut_dict(trans_dict)
        return trans_dict
    
    def train(self, n_epoch=10, n_episode=100):
        true_values = []
        actor_loss = []
        critic_loss = []
        with Progress() as progress:
            task = progress.add_task("epoch{0}".format(0), total=n_episode)
            for i in range(n_epoch):
                progress.tasks[task].description = "epoch {0} of {1}".format(i+1, n_epoch)
                progress.tasks[task].completed = 0
                for _ in range(n_episode):
                    trans_dict = self.simulate()
                    al, cl = self.update(trans_dict)

                    progress.update(task, advance=1)
                    true_values.append(np.sum(trans_dict["rewards"]))
                    actor_loss.append(al)
                    critic_loss.append(cl)
                self.agent.save("../model/check_point{0}.ptd".format(i))
                np.savez("../model/log.npz", 
                         true_values = np.array(true_values),
                         actor_loss = np.array(actor_loss),
                         critic_loss = np.array(critic_loss)
                        )
                

class planTrackTrainer:
    def __init__(self, mainPorp:propagator.Propagator, trackProp:propagatorT.PropagatorT, agent:agent.planTrackAgent) -> None:
        self.mainProp = mainPorp
        self.trackProp = trackProp
        self.agent = agent

    def trainTracker(self, horizon:int, episode=100, batch_size=256, target_mode="random"):
        loss_list = []
        with Progress() as progress:
            task = progress.add_task("training tracker", total=episode)
            for _ in range(episode):
                states = self.trackProp.randomInitStates(batch_size).to(self.agent.device)
                if target_mode=="random":
                    targets = self.trackProp.randomInitStates(batch_size).to(self.agent.device)
                    targets[:,3:] = 0 # vel zero, to be overloaded
                elif target_mode=="plan":
                    obss = self.trackProp.getObss(states)
                    targets, _ = self.agent.track_target(obss)
                else:
                    raise(ValueError, "unknown target mode")
                targets_seq = torch.tile(targets, (horizon,1,1)).detach()
                main_noise = torch.randn_like(targets_seq[:,:,:self.agent.plan_dim]) * 0.1
                sub_noise  = torch.randn_like(targets_seq[:,:,self.agent.plan_dim:]) * 0.01
                noise = torch.cat([main_noise, sub_noise], dim=2)
                targets_seq += noise # add noise
                loss = self.trackProp.seqTrack(states, targets_seq, self.agent.tracker)
                self.agent.tracker_opt.zero_grad()
                loss.backward()
                self.agent.tracker_opt.step()
                loss_list.append(loss.item())
                progress.update(task, advance=1)
        return loss_list
    
    def trainPlaner(self, episode=100, epoch=10, batch_size=256, explore_eps=0.5):
        actor_loss_list = []
        critic_loss_list = []
        total_rewards = []
        with Progress() as progress:
            task = progress.add_task("training planer", total=episode)
            for i in range(epoch):
                progress.tasks[task].description = "epoch {0} of {1}".format(i+1, epoch)
                progress.tasks[task].completed = 0
                for _ in range(episode):
                    states = self.mainProp.randomInitStates(batch_size)
                    obss = self.mainProp.getObss(states)
                    obss_ = torch.from_numpy(obss).float().to(self.agent.device)
                    if torch.rand(1) < explore_eps:
                        targets_, planed_ = self.agent.random_track_target(batch_size)
                    else:
                        targets_, planed_ = self.agent.track_target(obss_)
                    targets = targets_.detach().cpu().numpy()

                    rewards = self.mainProp.getPlanRewards(states, targets)
                    rewards_ = torch.from_numpy(rewards).float().to(self.agent.device)
                    Qvalues_ = self.agent.critic(obss_, planed_)
                    rewards_ = rewards_.reshape(Qvalues_.shape)
                    critic_loss = F.mse_loss(rewards_, Qvalues_)
                    self.agent.critic_opt.zero_grad()
                    critic_loss.backward()
                    self.agent.critic_opt.step()

                    targets_, planed_ = self.agent.track_target(obss_)
                    rewards = self.mainProp.getPlanRewards(states, targets_.detach().cpu().numpy())
                    Qvalues_ = self.agent.critic(obss_, planed_)
                    actor_loss = -Qvalues_.mean()
                    self.agent.actor_opt.zero_grad()
                    actor_loss.backward()
                    self.agent.actor_opt.step()

                    actor_loss_list.append(actor_loss.item())
                    critic_loss_list.append(critic_loss.item())
                    total_rewards.append(rewards.mean().item())
                    progress.update(task, advance=1)
        return actor_loss_list, critic_loss_list, total_rewards
    
    def test(self, horizon:int):
        '''
            debug
        '''
        trans_dict = D.init_transDict(horizon, self.mainProp.state_dim, self.mainProp.obs_dim, self.mainProp.action_dim, 
                                      other_terms={"target_states":(self.trackProp.state_dim,)})
        state = self.mainProp.randomInitStates(1)
        obs = self.mainProp.getObss(state)

        obs_0 = torch.from_numpy(obs).float().to(self.agent.device)
        primal_state_0 = self.agent.extract_primal_state(torch.from_numpy(state).float().to(self.agent.device))
        target_state_0, _ = self.agent.act(obs_0, primal_state_0)
        target_obs_0 = self.trackProp.getObss(target_state_0)
        for i in range(horizon):
            trans_dict["states"][i,...] = state
            trans_dict["obss"][i,...] = obs
            decoded = self.mainProp.statesDecode(state)
            forecast_time = decoded["forecast_time"].squeeze()
            is_approaching = forecast_time.max()>0
            primal_state_ = self.agent.extract_primal_state(torch.from_numpy(state).float().to(self.agent.device))
            primal_obs_ = self.trackProp.getObss(primal_state_)

            if is_approaching:
                # obs_ = torch.from_numpy(obs).float().to(self.agent.device)
                # target_state_, action_ = self.agent.act(obs_, primal_state_)
                # target_obs_ = self.trackProp.getObss(target_state_)
                target_state_ = target_state_0
                target_obs_ = target_obs_0
                _, action_ = self.agent.act_target(target_obs_, primal_obs_)
            else:
                target_state_ = torch.zeros_like(primal_state_)
                target_obs_ = self.trackProp.getObss(target_state_)
                _, action_ = self.agent.act_target(target_obs_, primal_obs_)
            action = action_.detach().cpu().numpy()
            state, _, done, obs = self.mainProp.propagate(state, action)
            trans_dict["target_states"][i,...] = target_state_.detach().cpu().numpy()
            trans_dict["actions"][i,...] = action
            trans_dict["next_states"][i,...] = state
            trans_dict["next_obss"][i,...] = obs
            if done.all():
                break
        return trans_dict, min(i,horizon)
    
def CWPTT(n_debris, device):
    mainProp = propagator.CWPlanTrackPropagator(n_debris)
    trackProp = propagatorT.CWPropagatorT(device=device)
    cub = torch.tensor([ 0.06]*3)
    clb = torch.tensor([-0.06]*3)
    pub = torch.tensor([ mainProp.max_dist]*3)
    plb = torch.tensor([-mainProp.max_dist]*3)
    a = agent.planTrackAgent(obs_dim=mainProp.obs_dim,
                             plan_dim=3,
                             control_dim=mainProp.action_dim,
                             pad_dim=3,
                             actor_hiddens=[512]*8,
                             tracker_hiddens=[512]*4,
                             critic_hiddens=[512]*8,
                             control_upper_bounds=cub,
                             control_lower_bounds=clb,
                             plan_upper_bounds=pub,
                             plan_lower_bounds=plb,
                             device=device)
    trainer = planTrackTrainer(mainProp, trackProp, a)
    return trainer