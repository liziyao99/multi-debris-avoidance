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

def gym():
    from trainer.myGym.gymTrainer import gymSAC, gymDDPG
    gs = gymSAC('Pendulum-v1')
    gd = gymDDPG('Pendulum-v1')
    # tr, cl ,al = gd.train(1, 10)
    tr, cl, al, apl = gs.train(1, 200)



if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    gamma = 0.99

    from env.propagators.variableDebris import vdPropagator
    vdp = vdPropagator(10, 0.06, dt=0.3, device="cuda", safe_dist=100, p_new_debris=1e-3, gamma=gamma)
    vdp.collision_reward = -1.

    from torch.utils.tensorboard import SummaryWriter
    from env.dynamic import cwutils
    import data.dicts as D

    from agent.agent import lstmDDPG_V, dualLstmDDPG
    LD = lstmDDPG_V(main_obs_dim=6, 
                sub_obs_dim=6, 
                sub_feature_dim=32, 
                lstm_num_layers=1, 
                action_dim=3,
                sigma=0.1,
                gamma=gamma,
                actor_hiddens=[512]*4, 
                critic_hiddens=[512]*4,
                action_upper_bounds=[ 1]*3, 
                action_lower_bounds=[-1]*3,
                actor_lr = 5e-5,
                critic_lr = 5e-5,
                partial_dim=None,
                device=vdp.device)
    LD.load("../model/LD_V_2.ptd")

    from data.buffer import replayBuffer
    buffer_keys = ["primal_states", "debris_states", "primal_obss", "debris_obss", "values"]
    update_batchsize = 4096
    buffer = replayBuffer(buffer_keys, capacity=100000, minimal_size=2*update_batchsize, batch_size=update_batchsize)

    from rich.progress import Progress
    n_epoch = 1
    n_episode = 1
    n_step = 400
    update_n_step = 1

    Np = 1
    Nd = 4
    total_rewards = []
    critic_loss = []

    collide = False
    sp = vdp.randomPrimalStates(Np)
    sd = vdp.randomDebrisStates(Nd)
    op, od = vdp.getObss(sp, sd)
    td_sim = dict(zip(buffer_keys, [[] for _ in buffer_keys]))
    done_flags = torch.zeros(Np, dtype=torch.bool)
    done_steps = (n_step-1)*torch.ones(Np, dtype=torch.int32)
    total_rewards = []
    Datas = []
    Actions = torch.zeros((n_step, Np, 3))
    Rewards = torch.zeros((n_step, Np))
    Critics = torch.zeros((n_step, Np))

    LD.init_OU_noise(Np)
    OU_noise = []
    for step in range(n_step):
        OU_noise.append(LD.OU_noise.detach().clone())
        actions, _ = LD.act(op, od)
        # actions = LD.tree_act(sp, sd, vdp, 40, 4, max_gen=20)
        (next_sp, next_sd), rewards, dones, (next_op, next_od) = vdp.propagate(sp, sd, actions,
                                                                            discard_leaving=True,
                                                                            new_debris=True)
        data = (sp, torch.stack([sd]*Np, dim=0), op, od)
        data = [d[~done_flags].detach().cpu() for d in data] + [None] # values
        Datas.append(data)

        Rewards[step,...] = rewards.detach().cpu()
        Critics[step,...] = LD._critic(op, od).detach().cpu()
        Actions[step,...] = actions.detach().cpu()
        sp, sd, op, od = next_sp, next_sd, next_op, next_od
        done_steps[dones.cpu()&~done_flags] = step
        done_flags |= dones.cpu()
        if done_flags.all():
            collide = True
            break