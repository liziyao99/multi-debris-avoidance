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
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from agent.setTransfomer import setTransformer
    from torch.utils.tensorboard import SummaryWriter

    st = setTransformer(n_feature=6, num_heads=3, encoder_fc_hiddens=[128], encoder_depth=1, 
                        n_output=48, pma_fc_hiddens=[128], pma_mab_fc_hiddens=[128], 
                        mhAtt_dropout=0.2, fc_dropout=0.5).to("cuda")
    linear = torch.nn.Linear(48, 3).to("cuda")
    model = torch.nn.Sequential(st, linear)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    from utils import lineProj, dotEachRow
    from env.propagators.variableDebris import vdPropagator, vdPropagatorPlane
    # vdp = vdPropagator(10, 0.06, dt=0.3, device="cuda", safe_dist=100, p_new_debris=1e-3, gamma=None)
    vdp = vdPropagatorPlane(10, 0.06, dt=0.3, device="cuda", safe_dist=100, p_new_debris=5e-2, gamma=None)

    def _gen_data(batch_size=1024, n_debris=None):
        if n_debris is None:
            n_debris = np.random.randint(1, vdp.max_n_debris+1)
        sp = vdp.randomPrimalStates(batch_size)
        sd = vdp.randomDebrisStatesTime(n_debris)
        op, od = vdp.getObss(sp, sd)
        pos = od[:,:,:3]
        vel = od[:,:,3:]
        dr = dotEachRow(pos, vel)
        _, closet_approach = lineProj(torch.zeros_like(pos), pos, vel)
        closet_dist = closet_approach.norm(dim=-1) # (batch_size, n_debris)
        closet_dist = torch.where(dr<0, closet_dist, torch.inf)
        _, min_idx = closet_dist.min(dim=1)
        data = od
        label = torch.zeros((batch_size, 3), device=vdp.device)
        for i in range(batch_size):
            label[i,...] = closet_approach[i, min_idx[i],:]
        return data, label
    
    test_data, test_label = _gen_data(10000, 1)
    print([(test_data[:,:,i]>0).float().mean().item() for i in range(6)])
    print([(test_label[:,i]>0).float().mean().item() for i in range(3)])
