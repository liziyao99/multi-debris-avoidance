from trainer.trainer import treeTrainer
from agent.agent import normalDistAgent
from env.env import debugTreeEnv


import rich.progress
import numpy as np

def debug1():
    from trainer.trainer import treeTrainer
    from agent.agent import normalDistAgent
    from env.env import treeEnvB
    import env.propagatorB as pB

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # propB = pB.CWPropagatorB(device="cuda")
    propB = pB.CWDebrisPropagatorB(device="cuda", n_debris=2)

    pop = 512
    max_gen = 3600
    action_bound = 6e-2
    env = treeEnvB.from_propagator(propB, population=pop, max_gen=max_gen, device="cuda")
    agent = normalDistAgent(obs_dim=propB.obs_dim, action_dim=propB.action_dim,
        actor_hiddens=[512]*4, critic_hiddens=[512]*4, 
        action_lower_bound=-action_bound, action_upper_bound=action_bound, 
        actor_lr=1E-5, critic_lr=5E-4)
    T = treeTrainer(env, agent, gamma=0.99)
    n = 3
    batch_size = 512
    horizon = 3600
    loss_list = []
    with rich.progress.Progress() as pbar:
        task = pbar.add_task("sequential optimize", total=n)
        for _ in range(n):
            states = T.propagator.T.randomInitStates(batch_size).to(agent.device)
            loss = T.propagator.seqOpt(states, agent, horizon)
            loss_list.append(loss.item())
            pbar.update(task, advance=1)

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
    debug1()
