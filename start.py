import matplotlib.pyplot as plt
import torch
import numpy as np
from env.propagators.hierarchicalPropagator import H2PlanTrackPropagator
prop = H2PlanTrackPropagator(8, device="cuda")

from agent.hierarchicalAgent import trackCW_SAC
action_bounds = [1., 1., 1.]
# sigma_bounds=  [1e2]*3
sigma_bounds=  [torch.inf]*3
h2out_ub = [ 0.06]*3
h2out_lb = [-0.06]*3
hAgent = trackCW_SAC(obs_dim=prop.obs_dim,
                    action_bounds=action_bounds,
                    sigma_upper_bounds=sigma_bounds,
                    h1a_hiddens=[512]*8, 
                    h2a_hiddens=[512]*4, 
                    h1c_hiddens=[512]*8,
                    h2out_ub=h2out_ub, 
                    h2out_lb=h2out_lb, 
                    h2a_lr=2e-3,
                    device="cuda")

import data.buffer
buffer_keys = ["states", "obss", "actions", "rewards", "next_states", "next_obss",
               "dones", "terminal_rewards"]
buffer = data.buffer.replayBuffer(buffer_keys, capacity=10000, batch_size=640)

from trainer.hierarchicalTrainer import H2Trainer, H2PlanTrackTrainer
loss_keys = ["critic_loss", "actor_loss"]
T = H2PlanTrackTrainer(prop, hAgent, buffer, loss_keys)
T.hAgent[1].load("../model/plan_track_1/h1.ptd")

log = T.offPolicyTrain(20, 100, states_num=256, h1_explore_eps=0)