from trainer.trainer import treeTrainer
from agent.agent import normalDistAgent
from env.env import debugTreeEnv

import numpy as np

def debug1(pop=20):
    max_gen = 200
    env = debugTreeEnv(pop, max_gen)
    agent = normalDistAgent(actor_hiddens=[256]*4, critic_hiddens=[256]*4, actor_lr=1E-4, critic_lr=2E-3)
    T = treeTrainer(env, agent, gamma=1)
    T.agent.load("../model/check_point3.ptd")
    # ds = T.simulate()
    # _ = T.test(1)
    T.train(n_episode=1, n_sim=1)
    # _ = T.test(t_max=1,)
    return

def debug2():
    from trainer.trainer import singleTrainer
    from agent.agent import PPOClipAgent
    from env.env import debugSingleEnv

    env = debugSingleEnv(200)
    agent = PPOClipAgent(actor_hiddens=[256]*4, critic_hiddens=[256]*4, actor_lr=1E-4, critic_lr=2E-3)
    T = singleTrainer(env, agent)

    d = T.simulate()
    T.agent.update(d)

def debug3():
    from trainer.trainer import treeTrainer
    from agent.agent import normalDistAgent
    from env.env import treeEnvB
    from env.propagatorB import motionSystemB, CWPropagatorB

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    space_dim = 3
    state_dim = space_dim*2
    A0 = np.eye(state_dim)
    A0 += np.hstack((np.zeros((state_dim,space_dim)), 
                    np.vstack((np.eye(space_dim),np.zeros((space_dim,space_dim))))
                    ))
    A1 = np.eye(state_dim) + np.random.uniform(low=-0.1, high=0.1, size=(state_dim, state_dim))
    A2 = A0 + np.random.uniform(low=-0.1, high=0.1, size=(state_dim, state_dim))

    max_dist=10.

    # propB = motionSystemB(A0, max_dist=max_dist, device="cuda")
    propB = CWPropagatorB(device="cuda")

    pop = 1024
    max_gen = 200
    env = treeEnvB.from_propagator(propB, population=pop, max_gen=max_gen, device="cuda")
    agent = normalDistAgent(obs_dim=propB.obs_dim, action_dim=propB.action_dim,
        actor_hiddens=[512]*8, critic_hiddens=[512]*8, actor_lr=1E-5, critic_lr=1E-4)
    T = treeTrainer(env, agent, gamma=0.995)
    T.train(4)


if __name__ == "__main__":
    from trainer.example import CWTreeTrainer
    from trainer.mpTrainer import mpTreeTrainer

    mpt = mpTreeTrainer(2, trainerType=CWTreeTrainer)
    mpt.train(1, 4, folder="../model/")
