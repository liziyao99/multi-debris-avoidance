import gymnasium as gym
from agent.agent import rlAgent
import agent.agent as A
from data.buffer import replayBuffer
import data.dicts as D

import numpy as np
import torch
from rich.progress import Progress

class gymTrainer:
    def __init__(self, env_name:str, agent:rlAgent, buffer:replayBuffer, max_episode_step=200) -> None:
        self.name = env_name
        self.env = gym.make(env_name, max_episode_steps=max_episode_step, render_mode="human")
        self.agent = agent
        self.buffer = buffer
        self.max_episode_step = max_episode_step
        self.loss_keys = []

    def reset(self):
        return self.env.reset()
    
    @property
    def obs_dim(self):
        return self.env.observation_space.shape[0]
    
    @property
    def action_dim(self):
        return self.env.action_space.shape[0]
    
    def simulate(self, off_policy_train=True, render=False):
        trans_dict = D.init_transDict(length=self.max_episode_step, state_dim=0, obs_dim=self.obs_dim, action_dim=self.action_dim)
        trans_dict = D.numpy_dict(trans_dict)
        obs, info = self.reset()
        done = False
        step = 0
        while not done:
            if render:
                self.env.render()
            trans_dict["obss"][step,:] = obs[:]
            _, action = self.agent.act(self.toTorch(obs))
            action = self.toNumpy(action)
            trans_dict["actions"][step,:] = action[:]
            obs, reward, terminated, truncated, info = self.env.step(action)
            reward = self.reward_normalize(reward)
            done = terminated or truncated
            trans_dict["rewards"][step] = reward
            trans_dict["next_obss"][step,:] = obs[:]
            trans_dict["dones"][step] = done
            step += 1
            if off_policy_train and self.buffer.size>self.buffer.minimal_size:
                self.agent.update(self.buffer.sample())
        return trans_dict
    
    def train(self, n_epoch:int, n_episode:int):
        keys = ["total_rewards"] + self.loss_keys
        log_dict = dict(zip( keys, [[] for _ in range(len(keys))] ))
        with Progress() as progress:
            task = progress.add_task("epoch{0}".format(0), total=n_episode)
            for i in range(n_epoch):
                progress.tasks[task].description = "epoch {0} of {1}".format(i+1, n_epoch)
                progress.tasks[task].completed = 0
                for _ in range(n_episode):
                    trans_dict = self.simulate()
                    Loss = self.agent.update(trans_dict)
                    total_reward = trans_dict["rewards"].sum().item()
                    log_dict["total_rewards"].append(total_reward)
                    for i in range(len(self.loss_keys)):
                        log_dict[self.loss_keys[i]].append(Loss[i])
                    self.buffer.from_dict(trans_dict)
                    progress.update(task, advance=1)
            self.agent.save(f"../model/check_point_{self.name}.ptd")
        # return log_dict # TODO: substitute tuple returns by dict
        return tuple(log_dict.values())
    
    def reward_normalize(self, reward:np.ndarray):
        return reward

    def toTorch(self, x:np.ndarray):
        if x.ndim == 1:
            x = x.reshape(1,-1)
        return torch.from_numpy(x).float().to(self.agent.device)
    
    def toNumpy(self, x:torch.Tensor):
        return x.squeeze(dim=0).detach().cpu().numpy()
    
class gymSAC(gymTrainer):
    def __init__(self, env_name: str,  max_episode_step=200, actor_hiddens=[128]*2, critic_hiddens=[128]*2) -> None:
        self.name = env_name
        self.env = gym.make(env_name, max_episode_steps=max_episode_step, render_mode="human")
        self.max_episode_step = max_episode_step
        action_bounds = list(self.env.action_space.high)
        sigma_bounds = [10]*self.action_dim
        agent = A.SAC(self.obs_dim, self.action_dim, 
                      actor_hiddens=actor_hiddens,
                      critic_hiddens=critic_hiddens,
                      action_bounds=action_bounds, 
                      sigma_upper_bounds=sigma_bounds,
                      actor_lr=1e-4,
                      critic_lr=2e-3,
                      alpha_lr=1e-4)
        buffer = replayBuffer(keys=("obss", "actions", "rewards", "next_obss", "dones"))
        self.agent = agent
        self.buffer = buffer
        self.loss_keys = ["critic_loss", "actor_loss", "alpha_loss"]
    
class gymDDPG(gymTrainer):
    def __init__(self, env_name: str, max_episode_step=200, actor_hiddens=[128]*2, critic_hiddens=[128]*2) -> None:
        self.name = env_name
        self.env = gym.make(env_name, max_episode_steps=max_episode_step, render_mode="human")
        self.max_episode_step = max_episode_step
        action_upper_bounds = list(self.env.action_space.high)
        action_lower_bounds = list(self.env.action_space.low)
        agent = A.DDPG(self.obs_dim, self.action_dim, 
                       actor_hiddens=actor_hiddens,
                       critic_hiddens=critic_hiddens,
                       action_upper_bounds=action_upper_bounds, 
                       action_lower_bounds=action_lower_bounds,
                       actor_lr=1e-4,
                       critic_lr=2e-3)
        buffer = replayBuffer(keys=("obss", "actions", "rewards", "next_obss", "dones"))
        self.agent = agent
        self.buffer = buffer
        self.loss_keys = ["critic_loss", "actor_loss"]

