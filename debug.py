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
    import env.propagators.propagatorB as pB
    
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
    import env.propagators.propagatorB as pB
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
    from agent.agent import planTrackAgent
    from env.propagators.propagator import CWPlanTrackPropagator

    prop = CWPlanTrackPropagator(2)
    sw = torch.tensor([1, 1, 1, 0.1, 0.1, 0.1])
    cw = torch.tensor([0.01, 0.01, 0.01])
    cub = torch.tensor([0.06]*3)
    clb = torch.tensor([-0.06]*3)
    agent = planTrackAgent(prop.state_dim, prop.obs_dim, prop.action_dim, sw, cw, control_upper_bounds=cub, control_lower_bounds=clb)
    states = prop.randomInitStates(10)
    obss = prop.getObss(states)

if __name__ == "__main__":
    from env.propagators.hirearchicalPropagator import H2CWDePropagator
    from agent.agent import H2Agent
    from trainer.trainer import H2TreeTrainer
    p = H2CWDePropagator(3, device="cuda")
    h1out_ub = [ 1000]*3 + [ 3.6]*3
    h1out_lb = [-1000]*3 + [-3.6]*3
    h2out_ub = [ 0.06]*3
    h2out_lb = [-0.06]*3
    agent = H2Agent(obs_dim=p.obs_dim,
                    h1obs_dim=p.obs_dim,
                    h2obs_dim=6, 
                    h1out_dim=p.h1_action_dim, 
                    h2out_dim=p.h2_action_dim, 
                    h1a_hiddens=[512], 
                    h2a_hiddens=[512]*8, 
                    h1c_hiddens=[512]*8,
                    h1out_ub=h1out_ub, 
                    h1out_lb=h1out_lb, 
                    h2out_ub=h2out_ub, 
                    h2out_lb=h2out_lb, 
                    device="cuda")
    T = H2TreeTrainer(p, agent,)
    # T.h2Pretrain(10,10,512)
    T.agent.load("../model/h2.ptd")
    T.train(1,1,10,32)