from trainer.trainer import treeTrainer
from agent.agent import normalDistAgent
from env.env import debugTreeEnv
import data.dicts as D

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

def debug3(batch_size=256, horizon=600, episode=100):
    from trainer.trainer import treeTrainer
    from agent.agent import normalDistAgent
    from env.env import treeEnvB
    import env.propagatorB as pB
    import rich.progress

    propB = pB.CWPropagatorB(device="cuda")

    action_bound = 6e-2
    agent = normalDistAgent(obs_dim=propB.obs_dim, action_dim=propB.action_dim,
                            actor_hiddens=[512]*10, critic_hiddens=[512]*10, 
                            action_lower_bound=-action_bound, action_upper_bound=action_bound, 
                            actor_lr=1E-5, critic_lr=5E-4)
    
    from agent import net
    import torch
    state_dim = propB.state_dim
    action_dim = propB.action_dim
    ub = agent.actor.obc.upper_bounds[:action_dim]
    lb = agent.actor.obc.lower_bounds[:action_dim]
    sw = torch.tensor([1.,1.,1.,0.1,0.1,0.1])
    cw = torch.tensor([0.01,0.01,0.01])
    tracker = net.trackNet(state_dim, action_dim,
                            n_hiddens=[512]*4, 
                            state_weights=sw, control_weights=cw, 
                            upper_bounds=ub, lower_bounds=lb).to(device=propB.T.device)
    opt = torch.optim.Adam(tracker.parameters(), lr=1e-4)
    
    loss_list = []
    with rich.progress.Progress() as pbar:
        task = pbar.add_task("sequential optimize", total=episode)
        for _ in range(episode):
            states = propB.T.randomInitStates(batch_size).to(agent.device)
            targets = propB.T.randomInitStates(batch_size).to(agent.device)
            targets_seq = torch.tile(targets, (horizon,1,1))
            loss = propB.T.seqTrack(states, targets_seq, tracker)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_list.append(loss.item())
            pbar.update(task, advance=1)
    return loss_list
    

if __name__ == "__main__":
    ll = debug3()
    plt.close("all")
    plt.plot(ll)
    plt.show()
