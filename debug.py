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

def setTransTest():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    gamma = 0.99

    from env.propagators.variableDebris import vdPropagator, vdPropagatorPlane
    # vdp = vdPropagator(10, 0.06, dt=0.3, device="cuda", safe_dist=100, p_new_debris=1e-3, gamma=None)
    vdp = vdPropagatorPlane(10, 0.06, dt=0.3, device="cuda", safe_dist=100, p_new_debris=5e-2, gamma=None)
    vdp.collision_reward = -1.
    vdp.beta_action = 1/vdp.max_dist

    from torch.utils.tensorboard import SummaryWriter
    from env.dynamic import cwutils
    import data.dicts as D

    
    from data.buffer import replayBuffer
    buffer_keys = ["primal_states", "debris_states", "primal_obss", "debris_obss", "values"]
    update_batchsize = 2048
    buffer = replayBuffer(buffer_keys, capacity=200000, minimal_size=2*update_batchsize, batch_size=update_batchsize)

    from agent.agent import setTransDDPG
    LD = setTransDDPG(main_obs_dim=20, 
                    sub_obs_dim=9, 
                    sub_feature_dim=48, 
                    action_dim=3, 
                    num_heads=3, 
                    encoder_fc_hiddens=[128],
                    pma_fc_hiddens=[128],
                    pma_mab_fc_hiddens=[128],
                    initial_fc_output=12,
                    initial_fc_hiddens=[512]*2,
                    encoder_depth=1, 
                    device=vdp.device, 
                    gamma=gamma,
                    critic_hiddens=[256]*2,
                    actor_hiddens=[256]*2,
                    action_upper_bounds=[ 1]*3,
                    action_lower_bounds=[-1]*3,
                    )
    
    from rich.progress import Progress
    import numpy as np
    n_epoch = 1
    n_episode = 1
    n_step = 1200
    mpc_horizon = 2

    Np = 128
    Nd = 1
    total_rewards = []

    writer = SummaryWriter("./tblogs/mpc")
    sp = vdp.randomPrimalStates(Np)
    sd = vdp.randomDebrisStates(Nd)
    op, od = vdp.getObss(sp, sd)

    with Progress() as pbar:
        task = pbar.add_task(total=n_episode, description="episode")
        Nd = np.random.randint(1, 4)
        for episode in range(n_episode):
            LD._OU = True
            LD.init_OU_noise(Np, scale=1.)
            critic_loss = []
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
                (next_sp, next_sd), rewards, dones, (next_op, next_od) = vdp.propagate(sp, sd, actions,
                                                                                    discard_leaving=True,
                                                                                    new_debris=True)
                data = (sp, torch.stack([sd]*Np, dim=0), op, od)
                data = [d[~done_flags].detach().cpu() for d in data] + [None] # values
                Datas.append(data)

                if buffer.size > buffer.minimal_size:
                    td_buffer = buffer.sample(stack=False)
                    LD._OU = False
                    LD.update(td_buffer, vdp, horizon=mpc_horizon, n_update=1)
                    LD._OU = True

                Rewards[step,...] = rewards.detach().cpu()
                sp, sd, op, od = next_sp, next_sd, next_op, next_od
                done_steps[dones.cpu()&~done_flags] = step
                done_flags |= dones.cpu()
                if done_flags.all():
                    break

            LD._OU = False

            Values = torch.zeros((n_step, Np))
            for step in range(n_step-1, -1, -1):
                if (step>done_steps).all():
                    continue
                if step==n_step-1:
                    Values[step, step==done_steps] = Rewards[step, step==done_steps] \
                        + LD._critic(next_op[step==done_steps], next_od[step==done_steps]).detach().cpu().squeeze(dim=-1)*LD.gamma
                else:
                    Values[step, step==done_steps] = Rewards[step, step==done_steps]
                    Values[step, step< done_steps] = Rewards[step, step< done_steps] + Values[step+1, step<done_steps]*LD.gamma
                Datas[step][-1] = Values[step, step<=done_steps]
            total_rewards.append(Rewards.sum(dim=0).mean().item())
            for _data in Datas:
                [td_sim[buffer_keys[i]].extend(_data[i]) for i in range(len(_data))]
            dicts = D.split_dict(td_sim, update_batchsize)
            for _dict in dicts:
                cl, al, _ = LD.update(_dict, vdp, horizon=mpc_horizon, n_update=1)
                critic_loss.append(cl)
            buffer.from_dict(td_sim)

            writer.add_scalar("reward", total_rewards[-1], episode)
            writer.add_scalar("critic_loss", np.mean(critic_loss), episode)

            for name, param in LD.setTrans.named_parameters():
                if param.grad is None:
                    continue
                writer.add_histogram("setTrans_"+name+"_data", param.data, episode)
                writer.add_histogram("setTrans_"+name+"_grad", param.grad, episode)
            for name, param in LD.critic.named_parameters():
                if param.grad is None:
                    continue
                writer.add_histogram("critic_"+name+"_data", param.data, episode)
                writer.add_histogram("critic_"+name+"_grad", param.grad, episode)
            for name, param in LD.actor.named_parameters():
                if param.grad is None:
                    continue
                writer.add_histogram("actor_"+name+"_data", param.data, episode)
                writer.add_histogram("actor_"+name+"_grad", param.grad, episode)
            pbar.advance(task, 1)


if __name__ == "__main__":
    setTransTest()