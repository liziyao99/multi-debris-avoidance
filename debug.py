from trainer.trainer import treeTrainer
from agent.agent import normalDistAgent
from env.env import debugTreeEnv


import torch
import rich.progress
import numpy as np
import matplotlib.pyplot as plt

def debug1(n_debris, seq=True):
    from trainer.trainer import treeTrainer
    from agent.agent import normalDistAgent
    from env.env import treeEnvB
    import env.propagatorB as pB
    
    propB = pB.CWDebrisPropagatorB(device="cuda", n_debris=n_debris)

    pop = 512
    max_gen = 3600
    action_bound = 6e-2
    env = treeEnvB.from_propagator(propB, population=pop, max_gen=max_gen, device="cuda")
    agent = normalDistAgent(obs_dim=propB.obs_dim, action_dim=propB.action_dim,
        actor_hiddens=[512]*4, critic_hiddens=[512]*4, 
        action_lower_bound=-action_bound, action_upper_bound=action_bound, 
        actor_lr=1E-5, critic_lr=5E-4)
    T = treeTrainer(env, agent, gamma=0.99)

    if seq:
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
    else: # N
        _, d = T.test(decide_mode="determined", t_max=.01, g_max=3)
    return d, T.testEnv.stage
        

def debug2(n_debris, dummy=False):
    from trainer.trainer import treeTrainer
    from agent.agent import normalDistAgent
    from env.env import treeEnvB
    import env.propagatorB as pB
    from plotting import analyze

    propB = pB.CWDebrisPropagatorB(device="cuda", n_debris=n_debris)

    pop = 512
    max_gen = 3600
    action_bound = 6e-2
    env = treeEnvB.from_propagator(propB, population=pop, max_gen=max_gen, device="cuda")
    agent = normalDistAgent(obs_dim=propB.obs_dim, action_dim=propB.action_dim,
        actor_hiddens=[512]*4, critic_hiddens=[512]*4, 
        action_lower_bound=-action_bound, action_upper_bound=action_bound, 
        actor_lr=1E-5, critic_lr=5E-4)
    T = treeTrainer(env, agent, gamma=0.99)
    plt.close('all')
    _, d = T.test(decide_mode="determined", t_max=.01, g_max=3)
    fig, _ = analyze.historyFile(d, T.agent, stage=T.testEnv.stage, n_debris=T.propagator.N.n_debris)
    return d, T.testEnv.stage



if __name__ == "__main__":
    d, end = debug2(2,False)
    plt.close("all")
    plt.plot(d["rewards"][:end])
    plt.show()
