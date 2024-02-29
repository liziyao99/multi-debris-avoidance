from agent.agent import rlAgent
from env.env import dummyEnv, treeEnv, singleEnv
from tree.tree import stateDict
import data.dicts as dicts

from rich.progress import Progress
import numpy as np
import torch

class dummyTrainer:
    def __init__(self, env:dummyEnv, agent:rlAgent) -> None:
        self.env = env
        self.agent = agent

    def new_state(self) -> stateDict:
        raise(NotImplementedError)

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
    
    def train(self, n_episode=10, n_sim=100):
        raise(NotImplementedError)

    def test(self, **kwargs):
        raise(NotImplementedError)
    
    @property
    def propagator(self):
        return self.env.propagator

class treeTrainer(dummyTrainer):
    def __init__(self, env:treeEnv, agent:rlAgent, gamma=1.) -> None:
        self.env = env
        self.agent = agent
        self.env.tree.gamma = gamma # discount factor
        self.testEnv = singleEnv.from_propagator(env.propagator, env.tree.max_gen)

    @property
    def tree(self):
        return self.env.tree

    def new_state(self) -> stateDict:
        # for debug, to be overloaded.
        sd = stateDict(self.env.state_dim, self.env.obs_dim, self.env.action_dim)
        sd.state[:3] = np.random.uniform(low=-5, high=5, size=3)
        sd.state[3:] = np.random.uniform(low=-0.1, high=0.1, size=3)
        obs = self.env.propagator.getObss(np.expand_dims(sd.state, axis=0))
        sd.obs[:] = obs.flatten()
        return sd

    def simulate(self, dicts_redundant=False):
        self.reset_env()
        while not self.env.step(self.agent):
            pass
        # self.tree.backup()
        dicts = self.tree.get_transDicts(redundant=dicts_redundant)
        return dicts

    def train(self, n_episode=10, n_sim=100, batch_size=1280):
        true_values = []
        actor_loss = []
        critic_loss = []
        with Progress() as progress:
            task = progress.add_task("epoch{0}".format(0), total=n_sim)
            for i in range(n_episode):
                progress.tasks[task].description = "episode {0} of {1}".format(i+1, n_episode)
                progress.tasks[task].completed = 0
                for _ in range(n_sim):
                    al, cl = [], []
                    trans_dicts = self.simulate(dicts_redundant=False)
                    trans_dict = dicts.concat_dicts(trans_dicts)
                    dict_batches = dicts.batch_dict(trans_dict, batch_size=batch_size)
                    for d in dict_batches:
                        al_, cl_ = self.agent.update(d)
                        al.append(al_)
                        cl.append(cl_)
                    # al, cl = self.update(trans_dict)

                    total_rewards = []
                    trans_dicts_r = self.tree.get_transDicts(redundant=True)
                    for d in trans_dicts_r:
                        total_reward = d["rewards"].sum()
                        total_rewards.append(total_reward)

                    progress.update(task, advance=1)
                    true_values.append(np.mean(total_rewards))
                    actor_loss.append(np.mean(al))
                    critic_loss.append(np.mean(cl))
                self.agent.save("../model/check_point{0}.ptd".format(i))
                np.savez("../model/log.npz", 
                         true_values = np.array(true_values),
                         actor_loss = np.array(actor_loss),
                         critic_loss = np.array(critic_loss)
                        )
                
    def test(self, t_max=0.02, pick_mode="descendants"):
        '''
            args:
                `t_max`: max searching time each step, second.
                `pick_mode`: see `GST.pick`.
        '''
        trans_dict = dicts.init_transDict(self.tree.max_gen+1, self.env.state_dim, self.env.obs_dim, self.env.action_dim)
        sd = self.new_state()
        self.testEnv.reset(sd.state)
        done = False
        while not done:
            sd = self.tree.decide(sd, self.agent, self.env.propagator, t_max=t_max, pick_mode=pick_mode)
            transit = self.testEnv.step(sd.action)
            done = transit[-1]
            self.testEnv.fill_dict(trans_dict, transit)
        total_rewards = trans_dict["rewards"].sum()
        return total_rewards, trans_dict
        

class singleTrainer(dummyTrainer):
    def __init__(self, env:singleEnv, agent:rlAgent) -> None:
        self.env = env
        self.agent = agent

    @property
    def obs(self):
        state = np.expand_dims(self.env.state, axis=0)
        obs = self.env.propagator.getObss(state)
        return obs.flatten()

    def new_state(self) -> stateDict:
        # for debug, to be overloaded.
        sd = stateDict(self.env.state_dim, self.env.obs_dim, self.env.action_dim)
        sd.state[:3] = np.random.uniform(low=-5, high=5, size=3)
        sd.state[3:] = np.random.uniform(low=-0.1, high=0.1, size=3)
        obs = self.env.propagator.getObss(np.expand_dims(sd.state, axis=0))
        sd.obs[:] = obs.flatten()
        return sd
     
    def simulate(self):
        trans_dict = dicts.init_transDict(self.env.max_episode+1, self.env.state_dim, self.env.obs_dim, self.env.action_dim,
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
    
    def train(self, n_episode=10, n_sim=100):
        true_values = []
        actor_loss = []
        critic_loss = []
        with Progress() as progress:
            task = progress.add_task("epoch{0}".format(0), total=n_sim)
            for i in range(n_episode):
                progress.tasks[task].description = "episode {0} of {1}".format(i+1, n_episode)
                progress.tasks[task].completed = 0
                for _ in range(n_sim):
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