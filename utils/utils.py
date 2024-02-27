import torch

def affine(x, s0, t0, s1, t1):
    return ((x-s0)*t1 + (t0-x)*s1)/(t0-s0)

def compute_advantage(gamma, lmd, td_delta:torch.Tensor):
    '''
        `td_delta`: Torch.Tensor of shape (step_num, copy_num).
    '''
    td_delta = torch.flip(td_delta, dims=[0]).detach()
    advantage_list = []
    advantage = 0.
    for delta in td_delta:
        advantage = gamma * lmd * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    advantage_list = torch.vstack(advantage_list)
    return advantage_list