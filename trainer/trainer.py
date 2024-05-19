from agent import agent
from env import env as ENV
from env.propagators import propagator
from env.propagators import propagatorT
from env.propagators import hirearchicalPropagator
from tree.geneticTree import stateDict
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
                    dict_batches = D.split_dict(trans_dict, batch_size=self.batch_size)
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
    
    def trainPlanner(self, episode=100, epoch=10, batch_size=256, explore_eps=0.5):
        actor_loss_list = []
        critic_loss_list = []
        total_rewards = []
        with Progress() as progress:
            task = progress.add_task("training planner", total=episode)
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
    
    def trainUnity(self, horizon:int, episode=100, epoch=10, batch_size=256, explore_eps=0.5):
        '''
            returns: `tracker_loss_list`, `planner_loss_list`, `critic_loss_list`, `total_rewards`
        '''
        tracker_loss_list = []
        planner_loss_list = []
        critic_loss_list = []
        total_rewards = []
        max_fuel_consume = self.agent.tracker.obc.max_norm*horizon
        with Progress() as progress:
            task = progress.add_task("training unity", total=episode)
            for i in range(epoch):
                progress.tasks[task].description = "epoch {0} of {1}".format(i+1, epoch)
                progress.tasks[task].completed = 0
                for _ in range(episode):
                    # plan
                    states0 = self.mainProp.randomInitStates(batch_size)
                    obss0 = self.mainProp.getObss(states0)
                    obss0_ = torch.from_numpy(obss0).float().to(self.agent.device)
                    if torch.rand(1) < explore_eps: # explore
                        targets0_, planed0_ = self.agent.random_track_target(batch_size)
                    else: # exploit
                        targets0_, planed0_ = self.agent.track_target(obss0_)
                    main_noise = torch.randn_like(targets0_[:,:self.agent.plan_dim]) * 10
                    sub_noise  = torch.randn_like(targets0_[:,self.agent.plan_dim:]) * 0.1
                    noise = torch.cat([main_noise, sub_noise], dim=-1)
                    planed0_ = planed0_+main_noise
                    targets0_ = targets0_+noise
                    targets0 = targets0_.detach().cpu().numpy()

                    # track init
                    trans_dict = D.init_transDictBatch(horizon, batch_size, self.mainProp.state_dim, self.mainProp.obs_dim, self.mainProp.action_dim,)
                    pstates_seq = [] # sequence of primal states, tensor require grad
                    targets_seq = [] # sequence of primal targets, tensor
                    states = states0.copy()
                    states_ = torch.from_numpy(states).float().to(self.agent.device)
                    obss = self.mainProp.getObss(states)
                    pstates_ = self.agent.extract_primal_state(states_) # primal states
                    pobss_ = self.trackProp.getObss(pstates_) # primal obss, NOTE: 2 prop have different `obssNormalize`, may lead to bug
                    # track simulate
                    for t in range(horizon):
                        trans_dict["states"][t,...] = states
                        trans_dict["obss"][t,...] = obss

                        decoded = self.mainProp.statesDecode(states)
                        forecast_time = decoded["forecast_time"] # shape (batch_size,n_debris,1)
                        is_approaching = np.max(forecast_time>0, axis=1) # shape (batch_size,1)
                        targets = np.where(is_approaching, targets0, np.zeros_like(targets0)) # shape (batch_size, agent.track_dim)
                        targets_ = torch.from_numpy(targets).float().to(self.agent.device)
                        tobss_ = self.trackProp.getObss(targets_) # target obss, NOTE: 2 prop have different `obssNormalize`, may lead to bug
                        pstates_seq.append(pstates_)
                        targets_seq.append(targets_.detach())

                        tracker_input = torch.hstack((pobss_,tobss_))
                        control_ = self.agent.tracker(tracker_input)
                        control = control_.detach().cpu().numpy()
                        pstates_, _, _, pobss_ = self.trackProp.propagate(pstates_, control_)
                        states, rewards, dones, obss = self.mainProp.propagate(states, control)
                        trans_dict["actions"][t,...] = control
                        trans_dict["next_states"][t,...] = states
                        trans_dict["next_obss"][t,...] = obss
                        trans_dict["rewards"][t,...] = rewards
                        trans_dict["dones"][t,...] = dones

                    # update tracker
                    pstates_seq = torch.stack(pstates_seq, dim=0)
                    targets_seq = torch.stack(targets_seq, dim=0)
                    track_loss = self.agent.tracker.loss(pstates_seq, targets_seq)
                    self.agent.tracker_opt.zero_grad()
                    track_loss.backward()
                    self.agent.tracker_opt.step()

                    # update critic
                    fuel_consume = np.linalg.norm(trans_dict["actions"], axis=-1) # shape (horizon, batch_size)
                    fuel_consume = np.sum(fuel_consume, axis=0) # shape (batch_size,)
                    plan_rewards = -fuel_consume/max_fuel_consume
                    fail = np.sum(trans_dict["dones"], axis=0) # fail if done before max horizon, shape (batch_size,)
                    plan_rewards = np.where(~fail, plan_rewards, -1)
                    plan_rewards_ = torch.from_numpy(plan_rewards).float().to(self.agent.device)
                    Qvalues_ = self.agent.critic(obss0_, planed0_)
                    plan_rewards_ = plan_rewards_.reshape(Qvalues_.shape)
                    critic_loss = F.mse_loss(plan_rewards_, Qvalues_)
                    self.agent.critic_opt.zero_grad()
                    critic_loss.backward()
                    self.agent.critic_opt.step()

                    # update actor (planner)
                    _, planed_ = self.agent.track_target(obss0_)
                    Qvalues_ = self.agent.critic(obss0_, planed_)
                    actor_loss = -Qvalues_.mean()
                    self.agent.actor_opt.zero_grad()
                    actor_loss.backward()
                    self.agent.actor_opt.step()

                    tracker_loss_list.append(track_loss.item())
                    planner_loss_list.append(actor_loss.item())
                    critic_loss_list.append(critic_loss.item())
                    total_rewards.append(np.mean(plan_rewards))
                    progress.update(task, advance=1)
        return tracker_loss_list, planner_loss_list, critic_loss_list, total_rewards
    
    def test(self, horizon:int, target_mode="static"):
        '''
            debug
        '''
        if target_mode not in ("static", "moving", "mc"):
            raise ValueError("target_mode must be one of (static, moving, mc)")
        trans_dict = D.init_transDict(horizon, self.mainProp.state_dim, self.mainProp.obs_dim, self.mainProp.action_dim, 
                                      other_terms={"target_states":(self.trackProp.state_dim,)})
        state = self.mainProp.randomInitStates(1)
        obs = self.mainProp.getObss(state)
        obs_0 = torch.from_numpy(obs).float().to(self.agent.device)

        if target_mode=="static":
            target_state_0, _ = self.agent.track_target(obs_0)
            target_obs_0 = self.trackProp.getObss(target_state_0)
        elif target_mode=="mc":
            n = 1000
            target_states = self.trackProp.randomInitStates(n)
            plans = target_states[:,:self.agent.plan_dim]
            obss = torch.tile(obs_0, (n, 1))
            Qs = self.agent.critic(obss, plans)
            target_state_0 = target_states[Qs.argmax()].unsqueeze(0)
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
                if target_mode in ("static", "mc"):
                    target_state_ = target_state_0
                    target_obs_ = target_obs_0
                elif target_mode=="moving":
                    obs_ = torch.from_numpy(obs).float().to(self.agent.device)
                    target_state_, _ = self.agent.track_target(obs_)
                    target_obs_ = self.trackProp.getObss(target_state_)
                _, action_ = self.agent.act_target(primal_obs_, target_obs_)
            else:
                target_state_ = torch.zeros_like(primal_state_)
                target_obs_ = self.trackProp.getObss(target_state_)
                _, action_ = self.agent.act_target(primal_obs_, target_obs_)
            action = action_.detach().cpu().numpy()
            state, _, done, obs = self.mainProp.propagate(state, action)
            trans_dict["target_states"][i,...] = target_state_.detach().cpu().numpy()
            trans_dict["actions"][i,...] = action
            trans_dict["next_states"][i,...] = state
            trans_dict["next_obss"][i,...] = obs
            if done.all():
                break
        return trans_dict, min(i,horizon)
    
def CWPTT(n_debris, device, load=""):
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
    if load:
        a.load(load)
    trainer = planTrackTrainer(mainProp, trackProp, a)
    return trainer, a

from tree.undoTree import undoTree
class H2TreeTrainer:
    def __init__(self, 
                 prop:hirearchicalPropagator.H2Propagator,
                 agent:agent.H2Agent,
                 h2a_bound=0.06) -> None:
        self.prop = prop
        self.agent = agent
        self.tree = undoTree(prop.h1_step, prop.state_dim, prop.obs_dim, prop.h1_action_dim, gamma=agent.gamma)
        self.h2a_bound = h2a_bound

    @property
    def device(self):
        return self.agent.device
    
    def h2a_clip(self, actions):
        return torch.clamp(actions, -self.h2a_bound, self.h2a_bound)

    def h2Sim(self, states0:torch.Tensor, h1actions:torch.Tensor=None, h1noise=True, prop_with_grad=True):
        '''
            returns: `h2transdict`, `next_states`, `h2rewards`(with grad if `prop_with_grad`), `h1actions`, `h1rewards`, `dones`, `terminal_rewards`.
        '''
        batch_size = states0.shape[0]
        trans_dict = D.init_transDictBatch(self.prop.h2_step, batch_size, self.prop.state_dim, self.prop.obs_dim, self.prop.h2_action_dim,
                                           struct="torch", device=self.agent.device)
        step = 0
        done = False
        done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.agent.device)
        obss = self.prop.getObss(states0, require_grad=True)
        
        if h1actions is None:
            _, h1actions = self.agent.act(obss)
        if h1noise:
            noise = self.agent.actor.uniSample(batch_size)/10
            h1actions = h1actions + noise
        h1actions = self.agent.actor.clip(h1actions)
        h1actions_incremented = (h1actions+states0[:,:6]).detach() # increment control
        
        states = states0.clone()
        next_states = states0.clone()
        h1rewards = torch.zeros(batch_size, device=self.agent.device)
        h2rewards = torch.zeros(batch_size, device=self.agent.device, requires_grad=prop_with_grad)
        terminal_rewards = torch.zeros(batch_size, device=self.agent.device)
        while not done and step<self.prop.h2_step:
            trans_dict["states"][step,...] = states[...].detach()
            trans_dict["obss"][step,...] = obss[...].detach()
            _, actions = self.agent.h2act(obss, h1actions_incremented)
            states, obss, h1rs, h2rs, dones, trs = self.prop.propagate(states, h1actions_incremented, actions, require_grad=prop_with_grad)
            terminal_rewards = torch.where(done_flags, terminal_rewards, trs)
            next_states = torch.where(done_flags.unsqueeze(1), next_states, states)
            done_flags = done_flags | dones
            trans_dict["actions"][step,...] = actions[...].detach()
            trans_dict["next_states"][step,...] = states[...].detach()
            trans_dict["next_obss"][step,...] = obss[...].detach()
            trans_dict["rewards"][step,...] = h2rs[...].detach()
            trans_dict["dones"][step,...] = dones[...].detach()
            h1rewards = h1rewards + h1rs*(~done_flags)
            # h2rewards = h2rewards + h2rs
            h2rewards = h2rs
            done = torch.all(dones)
            step += 1
        return trans_dict, next_states, h2rewards, h1actions, h1rewards, done_flags, terminal_rewards
    
    def h1Sim(self, states0:torch.Tensor=None, h1_explore_eps=0., states_num=1, train_h2a=False, prop_with_grad=True):
        if states0 is None:
            states0 = self.prop.randomInitStates(states_num)
        batch_size = states0.shape[0]
        h1td = D.init_transDictBatch(self.prop.h1_step, batch_size, self.prop.state_dim, self.prop.obs_dim, self.prop.h1_action_dim,
                                     items=("terminal_rewards",),
                                     struct="torch", device=self.agent.device)
        h2Loss = []
        step = 0
        done = False
        done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.agent.device)
        states = states0
        obss = self.prop.getObss(states)
        while not done and step<self.prop.h1_step:
            h1td["states"][step, ...] = states[...].detach()
            h1td["obss"][step, ...] = obss[...].detach()
            if np.random.rand()<h1_explore_eps:
                h1actions = self.agent.actor.uniSample(batch_size)
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
            self.agent.h2a_opt.zero_grad()
            h2loss.backward()
            self.agent.h2a_opt.step()
            h2loss = h2loss.detach()
        return h1td, h2Loss
    
    def treeSim(self, select_itr:int, select_size:int, state0:torch.Tensor=None, h1_explore_eps=0., train_h2a=False):
        '''
            returns: `trans_dict`, `H2loss`, `root.V_target`
        '''
        if state0 is None:
            state0 = self.prop.randomInitStates(1)
        self.tree.reset_root(state0[0].detach().cpu().numpy(), self.prop.getObss(state0)[0].detach().cpu().numpy())
        H2loss = []
        for i in range(select_itr):
            states, idx_pairs = self.tree.select(select_size)
            states = np.vstack(states)
            states = torch.from_numpy(states).float().to(self.agent.device)
            h1td, h2loss = self.h1Sim(states, h1_explore_eps, train_h2a=train_h2a)
            h1td = D.numpy_dict(h1td)
            tds = D.deBatch_dict(h1td)
            for j in range(select_size):
                self.tree.new_traj(idx_pairs[j][0], idx_pairs[j][1], tds[j])
            H2loss.append(h2loss)
            self.tree.backup()
            self.tree.update_regret_mc()
        trans_dict = self.tree.to_transdict()
        return trans_dict, H2loss, self.tree.root.V_target
        
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
    
    def h2Pretrain(self, epoch:int, episode:int, horizon=1200, states_num=256):
        Loss = []
        with Progress() as progress:
            task = progress.add_task("H2TreeTrainer pretraining:", total=episode)
            for i in range(epoch):
                progress.tasks[task].description = "epoch {0} of {1}".format(i+1, epoch)
                progress.tasks[task].completed = 0
                for _ in range(episode):
                    loss = torch.zeros(1, device=self.agent.device, requires_grad=True)
                    states0 = self.prop.randomInitStates(states_num)
                    h1actions = self.agent.actor.uniSample(states_num)
                    states = states0
                    obss = self.prop.getObss(states)
                    for t in range(horizon):
                        _, h2actions = self.agent.h2act(obss, h1actions)
                        states, obss, _, h2rewards, _, _ = self.prop.propagate(states, h1actions, h2actions, require_grad=True)
                        loss = loss - h2rewards.mean()
                    self.agent.h2a_opt.zero_grad()
                    loss.backward()
                    self.agent.h2a_opt.step()
                    Loss.append(loss.item())
                    progress.update(task, advance=1)
        return Loss


from env.dynamic import matrix
class H2TreeTrainerAlter(H2TreeTrainer):
    def __init__(self, prop: hirearchicalPropagator.H2CWDePropagator, agent: agent.H2Agent, tutor:agent.planTrackAgent, 
                 conGramMat_file=None, h2a_bound=0.06) -> None:
        super().__init__(prop, agent)

        self.transfer_time = prop.dt*prop.h2_step
        if conGramMat_file is None:
            conGramMat_file = f"../model/conGramMat{int(self.transfer_time)}.npy"
        try:
            conGramMat = np.load(conGramMat_file)
        except:
            conGramMat = matrix.CW_ConGramMat(self.transfer_time, prop.orbit_rad) # need numerical integral, time consumming!
            np.save(conGramMat_file, conGramMat)
        W_ = np.linalg.inv(conGramMat)
        self.W_ = torch.from_numpy(W_).float().to(self.agent.device)
        self.B = torch.vstack([torch.zeros([3,3]), torch.eye(3)]).to(self.agent.device)
        Phi0 = matrix.CW_TransMat(0, self.transfer_time, prop.orbit_rad)
        self.Phi0 = torch.from_numpy(Phi0).float().to(self.agent.device)

        self.tutor = tutor

        self.h2a_bound = h2a_bound


    def transfer_u(self, states0, targets, step):
        tau = self.prop.dt*step
        Phi1 = matrix.CW_TransMat(tau, self.transfer_time, self.prop.orbit_rad)
        Phi1 = torch.from_numpy(Phi1).float().to(self.device)
        temp = -self.B.T@Phi1.T@self.W_
        return (states0@self.Phi0.T-targets)@temp.T
    
    def tutor_obs(self, states):
        batch_size = states.shape[0]
        decoded = self.prop.statesDecode(states)
        primal = decoded["primal"].squeeze(dim=1)
        forecast_time = decoded["forecast_time"].reshape((batch_size, -1))
        forecast_pos = decoded["forecast_pos"].reshape((batch_size, -1))
        forecast_vel = decoded["forecast_vel"].reshape((batch_size, -1))
        obss = torch.cat((primal, forecast_time, forecast_pos, forecast_vel), dim=1)
        return obss

    def h2Sim(self, states0:torch.Tensor, h1actions:torch.Tensor=None, h1noise=True, prop_with_grad=True):
        '''
            returns: `h2transdict`, `next_states`, `h2rewards`(with grad if `prop_with_grad`), `h1actions`, `h1rewards`, `dones`, `terminal_rewards`.
        '''
        batch_size = states0.shape[0]
        trans_dict = D.init_transDictBatch(self.prop.h2_step, batch_size, self.prop.state_dim, self.prop.obs_dim, self.prop.h2_action_dim,
                                           struct="torch", device=self.agent.device)
        step = 0
        done = False
        done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.agent.device)
        obss = self.prop.getObss(states0, require_grad=True)
        
        if h1actions is None:
            _, h1actions = self.agent.act(obss)
        if h1noise:
            try:
                noise = self.agent.actor.uniSample(batch_size)/10
                h1actions = h1actions + noise
                h1actions = self.agent.actor.clip(h1actions)
            except:
                pass
        targets = (h1actions+states0[:,:6]).detach() # increment control
        
        primals0 = states0[:,:6]
        states = states0.clone()
        next_states = states0.clone()
        h1rewards = torch.zeros(batch_size, device=self.agent.device)
        h2rewards = torch.zeros(batch_size, device=self.agent.device, requires_grad=prop_with_grad)
        terminal_rewards = torch.zeros(batch_size, device=self.agent.device)
        while not done and step<self.prop.h2_step:
            trans_dict["states"][step,...] = states[...].detach()
            trans_dict["obss"][step,...] = obss[...].detach()
            actions = self.transfer_u(primals0, targets, step)
            actions = self.h2a_clip(actions)
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
            done = torch.all(dones)
            step += 1
        delta_states = next_states - states0
        return trans_dict, next_states, h2rewards, h1actions, h1rewards, done_flags, terminal_rewards
    
    def tutorSim(self, states0, states_num=1):
        if states0 is None:
            states0 = self.prop.randomInitStates(states_num)
        batch_size = states0.shape[0]
        h1td = D.init_transDictBatch(self.prop.h1_step, batch_size, self.prop.state_dim, self.prop.obs_dim, self.prop.h1_action_dim,
                                     items=("terminal_rewards",),
                                     struct="torch", device=self.agent.device)
        h2Loss = []
        step = 0
        done = False
        done_flags = torch.zeros(batch_size, dtype=torch.bool, device=self.agent.device)
        states = states0
        obss = self.prop.getObss(states)
        tutor_obss = self.tutor_obs(states)
        tutor_targets, _ = self.tutor.track_target(tutor_obss)
        while not done and step<self.prop.h1_step:
            h1td["states"][step, ...] = states[...].detach()
            h1td["obss"][step, ...] = obss[...].detach()

            decoded = self.prop.statesDecode(states)
            forecast_time = decoded["forecast_time"]
            is_approaching = torch.sum(forecast_time>0, dim=1).to(torch.bool)
            targets = torch.where(is_approaching, tutor_targets, torch.zeros_like(tutor_targets, device=self.device))
            h1actions = targets - states[:,:6] # increment control
            
            _, states, h2rs, h1as, h1rs, dones, trs = self.h2Sim(states, h1actions, h1noise=False, prop_with_grad=False)
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
        return h1td, h2Loss

    
    def treeSim(self, select_itr:int, select_size:int, state0:torch.Tensor=None, h1_explore_eps=0., train_h2a=False, tutor_on=False):
        '''
            returns: `trans_dict`, `H2loss`, `root.V_target`
        '''
        if state0 is None:
            state0 = self.prop.randomInitStates(1)
        self.tree.reset_root(state0[0].detach().cpu().numpy(), self.prop.getObss(state0)[0].detach().cpu().numpy())
        
        # imitate learning
        if tutor_on:
            h1td, _ = self.tutorSim(state0)
            h1td = D.numpy_dict(h1td)
            tds = D.deBatch_dict(h1td)
            for d in tds:
                self.tree.new_traj(0, 0, d)
        
        H2loss = []
        for i in range(select_itr):
            states, idx_pairs = self.tree.select(select_size)
            states = np.vstack(states)
            states = torch.from_numpy(states).float().to(self.agent.device)
            h1td, h2loss = self.h1Sim(states, h1_explore_eps, train_h2a=train_h2a)
            h1td = D.numpy_dict(h1td)
            tds = D.deBatch_dict(h1td)
            for j in range(select_size):
                self.tree.new_traj(idx_pairs[j][0], idx_pairs[j][1], tds[j])
            H2loss.append(h2loss)
            self.tree.backup()
            self.tree.update_regret_mc()
        trans_dict = self.tree.to_transdict()
        return trans_dict, H2loss, self.tree.root.V_target