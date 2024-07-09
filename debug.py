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

def alter():
    from trainer.mpTrainer_ import mpH2TreeTrainer
    import data.buffer
    h1out_ub = [ 1000]*3 + [ 3.6]*3
    h1out_lb = [-1000]*3 + [-3.6]*3
    h2out_ub = [ 0.06]*3
    h2out_lb = [-0.06]*3
    agentArgs = {"h1a_hiddens": [1024]*6, 
                "h2a_hiddens": [512]*8, 
                "h1c_hiddens": [1024]*6,
                "h1out_ub": h1out_ub, 
                "h1out_lb": h1out_lb, 
                "h2out_ub": h2out_ub, 
                "h2out_lb": h2out_lb, }
    buffer_keys = ["states", "obss", "actions", "rewards", "next_states", "next_obss",
                "dones", "Q_targets", "V_targets", "regret_mc", "terminal_rewards"]
    buffer = data.buffer.replayBuffer(buffer_keys, capacity=10000, batch_size=640)
    mt = mpH2TreeTrainer(n_process=16, buffer=buffer, n_debris=3, agentArgs=agentArgs, 
                    select_itr=1, select_size=1, batch_size=1024, main_device="cuda", mode="alter")
    mt.debug()

def sac():
    from trainer.mpTrainer_ import mpH2TreeTrainer
    import data.buffer
    action_bounds = [1000, 1000, 1000, 3.6, 3.6, 3.6]
    sigma_bounds=  [1e2]*6
    agentArgs = {"actor_hiddens": [640]*14, 
                "critic_hiddens": [640]*14,
                "action_bounds": action_bounds,
                "sigma_upper_bounds": sigma_bounds }
    buffer_keys = ["states", "obss", "actions", "rewards", "next_states", "next_obss",
                "dones", "Q_targets", "V_targets", "regret_mc", "terminal_rewards"]
    buffer = data.buffer.replayBuffer(buffer_keys, capacity=10000, batch_size=640)
    mt = mpH2TreeTrainer(n_process=16, buffer=buffer, n_debris=3, agentArgs=agentArgs, 
                    select_itr=2, select_size=100, batch_size=1024, main_device="cuda", mode="SAC")
    mt.debug()

if __name__ == "__main__":
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    from env.propagators.variableDebris import vdPropagator
    vdp = vdPropagator(10, 0.06, dt=0.2, device="cuda", p_new_debris=1e-3)
    vdp.collision_reward = -1.

    from torch.utils.tensorboard import SummaryWriter
    from env.dynamic import cwutils
    import data.dicts as D

    from agent.agent import lstmDDPG_V
    LD = lstmDDPG_V(main_obs_dim=6, 
                sub_obs_dim=6, 
                sub_feature_dim=32, 
                lstm_num_layers=1, 
                action_dim=3,
                sigma=0.1,
                actor_hiddens=[512]*4, 
                critic_hiddens=[512]*4,
                action_upper_bounds=[ 1]*3, 
                action_lower_bounds=[-1]*3,
                partial_dim=None,
                device=vdp.device)

    from data.buffer import replayBuffer
    buffer_keys = ["primal_states", "debris_states", "primal_obss", "debris_obss", "values"]
    buffer = replayBuffer(buffer_keys, capacity=100000, batch_size=512)

    n_epoch = 1
    n_episode = 1
    n_step = 3000

    Np = 64
    Nd = 2
    total_rewards = []
    critic_loss = []

    for episode in range(n_episode):
        sp = vdp.randomPrimalStates(Np)
        sd = vdp.randomDebrisStates(Nd)
        op, od = vdp.getObss(sp, sd)
        td_sim = dict(zip(buffer_keys, [[] for _ in buffer_keys]))
        done_flags = torch.zeros(Np, dtype=torch.bool)
        done_steps = (n_step-1)*torch.ones(Np, dtype=torch.int32)
        Datas = []
        Rewards = torch.zeros((n_step, Np))
        for step in range(n_step):        
            _, actions = LD.act(op, od)
            (next_sp, next_sd), rewards, dones, (next_op, next_od) = vdp.propagate(sp, sd, actions)
            data = (sp, torch.stack([sd]*Np, dim=0), op, od)
            data = [d[~done_flags].detach().cpu() for d in data] + [None] # values
            Datas.append(data)

            # if buffer.size > buffer.minimal_size:
            #     td_buffer = buffer.sample(stack=False)
            #     LD.update(td_buffer)

            Rewards[step,...] = rewards.detach().cpu()
            sp, sd, op, od = next_sp, next_sd, next_op, next_od
            done_steps[dones.cpu()&~done_flags] = step
            done_flags |= dones.cpu()
            if done_flags.all():
                break

        Values = torch.zeros((n_step, Np))
        for step in range(n_step-1, -1, -1):
            if (step>done_steps).all():
                continue
            if step==n_step-1:
                Values[step, step==done_steps] = Rewards[step, step==done_steps] \
                    + LD._critic(next_op[step==done_steps], next_od[step==done_steps]).detach().cpu().squeeze(dim=-1)*LD.gamma
            else:
                Values[step, step==done_steps] = Rewards[step, step==done_steps]
                Values[step, step<done_steps] = Rewards[step, step<done_steps] + Values[step+1, step<done_steps]*LD.gamma
            Datas[step][-1] = Values[step, step<=done_steps]
        total_rewards.append(Rewards.sum(dim=0).mean().item())
        for d in Datas:
            [td_sim[buffer_keys[i]].extend(d[i]) for i in range(len(d))]
        # cl, al, _ = LD.update(td_sim)
        # critic_loss.append(cl)
    
    dicts = D.split_dict(td_sim, 2048)
    cl, al, _ = LD.update(dicts[-1], vdp)