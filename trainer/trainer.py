from agent.agent import rlAgent
from env.env import treeEnv, singleEnv
from tree.tree import stateDict
import data.dicts as dicts

from rich.progress import Progress
import numpy as np

class treeTrainer:
    def __init__(self, env:treeEnv, agent:rlAgent, gamma=1.) -> None:
        self.env = env
        self.agent = agent
        self.env.tree.gamma = gamma # discount factor
        self.test_env = singleEnv.from_propagator(env.propagator, env.tree.max_gen)

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

    def reset_env(self):
        sd = self.new_state()
        self.env.reset(sd)

    def simulate(self):
        self.reset_env()
        while not self.env.step(self.agent):
            pass
        dicts = self.tree.get_transDicts()
        return dicts
    
    def update(self, trans_dict:dict):
        '''
            returns: `actor_loss`, `critic_loss`.
        '''
        return self.agent.update(trans_dict)

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
                    trans_dicts = self.simulate()
                    trans_dict = dicts.concat_dicts(trans_dicts)
                    al, cl = self.update(trans_dict)

                    total_rewards = []
                    for d in trans_dicts:
                        total_reward = d["rewards"].sum()
                        total_rewards.append(total_reward)

                    progress.update(task, advance=1)
                    true_values.append(np.mean(total_rewards))
                    actor_loss.append(al)
                    critic_loss.append(cl)
                self.agent.save("../model/check_point{0}.ptd".format(i))
                np.savez("../model/log.npz", 
                         true_values = np.array(true_values),
                         actor_loss = np.array(actor_loss),
                         critic_loss = np.array(critic_loss)
                        )
                
    def test(self, t_max=0.01):
        trans_dict = dicts.init_transDict(self.tree.max_gen+1, self.env.state_dim, self.env.obs_dim, self.env.action_dim)
        sd = self.new_state()
        self.test_env.reset(sd.state)
        done = False
        while not done:
            sd = self.tree.decide(sd, self.agent, self.env.propagator, t_max)
            transit = self.test_env.step(sd.action)
            done = transit[-1]
            self.test_env.fill_dict(trans_dict, transit)
        total_rewards = trans_dict["rewards"].sum()
        return total_rewards, trans_dict
        